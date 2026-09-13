"""Demonstration initialization and PPO for continuous Cartesian arm actions."""
import sys
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import torch
from stable_baselines3 import PPO
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rlbot.arm_env import ArmEnv, validate_arm_config
from rlbot.connectome import ConnectomeFeatures
from train_connectome import RehearsalPPO, Progress


class Teacher:
    """Training data only; deliberately absent from ArmEnv and policy inference."""
    def __init__(self, speed=None):
        self.phase = 0
        self.count = 0
        self.speed = speed

    def action(self, env):
        cube = env.cube
        pick = env.start + [0,0,env.cube_width/2+.008]
        place = env.goal + [0,0,.008]
        targets = [pick, pick, pick+[0,0,.15], place+[0,0,.15], place, place, place+[0,0,.12]]
        shut = env.closed_action
        grip = [1,shut,shut,shut,shut,1,1][self.phase]
        target = targets[self.phase]
        # Ease through the phases that carry the cube. Driving the target at the
        # full 80 mm/s steps the commanded velocity, and a pincer's angled pads
        # flick the cube out sideways when that happens.
        rate = (self.speed or env.carry_speed)[self.phase]
        close = np.linalg.norm(env.grip_point-target)<.006
        self.count = self.count+1 if close else 0
        wait = [8,20,8,8,12,20,30][self.phase]
        if self.count >= wait and self.phase<6:
            self.phase += 1
            self.count = 0
        return np.r_[np.clip((target-env.target)/.004,-rate,rate),grip].astype(np.float32)


class ReleaseCorrection:
    def __init__(self):
        self.active = False
        self.settled = 0
        self.opening = False
        self.clear = 0

    def action(self, env):
        distance = np.linalg.norm(env.cube[:2]-env.goal[:2])
        self.active = self.active or (distance < .025 and env.cube[2] < env.rest_height+.020
                                     and env.max_lift > .08 and env.held > .5)
        if not self.active:
            return None
        height = env.rest_height+.008
        seated = abs(env.grip_point[2]-height) < .006 and env.cube[2] < env.rest_height+.003
        self.settled = self.settled+1 if seated else 0
        self.opening = self.opening or self.settled >= 12
        released = self.opening and env.contacts() == 0 and env.data.qpos[env.gripq] > env.released_qpos
        self.clear = self.clear+1 if released else 0
        action = np.array([0,0,np.clip((height-env.target[2])/.004,-1,1),env.closed_action],dtype=np.float32)
        if self.opening:
            action[2:] = [0,1]
        if self.clear >= 8:
            action[2] = np.clip((height+.12-env.target[2])/.004,-.5,.5)
        return action


def collect_release_corrections(model, env, count, start=30):
    observations=[]; actions=[]; corrected=[]; runs=[]
    for seed in range(start,start+count):
        state,_=env.reset(seed=seed)
        teacher=ReleaseCorrection()
        episode_obs=[]; episode_actions=[]; episode_corrected=[]
        for _ in range(env.horizon):
            with torch.no_grad():
                tensor,_=model.policy.obs_to_tensor(state)
                learned=model.policy.get_distribution(tensor).distribution.mean[0].cpu().numpy()
            correction=teacher.action(env)
            action=learned if correction is None else correction
            episode_obs.append(state.copy())
            episode_actions.append(action.copy())
            episode_corrected.append(correction is not None)
            state,_,t,tr,info=env.step(action)
            if t or tr: break
        if info['success']:
            observations.extend(episode_obs)
            actions.extend(episode_actions)
            corrected.extend(episode_corrected)
        runs.append(dict(seed=seed,correction_states=sum(episode_corrected),**info))
        print('Release correction rollout',runs[-1],flush=True)
    if not any(corrected):
        raise RuntimeError('No successful release corrections to imitate')
    return (np.asarray(observations,dtype=np.float32),np.asarray(actions,dtype=np.float32),
            np.asarray(corrected,dtype=bool),runs)


def calibrate_jaw(model, gain):
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Jaw output gain must be finite and positive')
    with torch.no_grad():
        model.policy.action_net.weight[3].mul_(gain)
        model.policy.action_net.bias[3].mul_(gain)
    previous = (getattr(model,'arm_calibration',None) or {}).get('jaw_output_gain',1.)
    model.arm_calibration = dict(jaw_output_gain=previous*gain)


def evaluate(model, env, count, start=1000):
    runs=[]
    for seed in range(start,start+count):
        obs,_=env.reset(seed=seed)
        teacher=Teacher() if model is None else None
        for _ in range(env.horizon):
            action=teacher.action(env) if model is None else model.predict(obs,deterministic=True)[0]
            obs,r,t,tr,info=env.step(action)
            if t or tr: break
        runs.append(dict(seed=seed,**info))
    return dict(successes=sum(r['success'] for r in runs),episodes=count,runs=runs)


def distill_release(model, env, output, count, updates, learning_rate, demonstrations=None):
    torch.manual_seed(7)
    output.mkdir(parents=True,exist_ok=True)
    if demonstrations is None:
        observations,actions,corrected,runs=collect_release_corrections(model,env,count)
        np.savez_compressed(output/'release_corrections.npz',observations=observations,actions=actions,
                            corrected=corrected,runs=json.dumps(runs),
                            configuration=json.dumps(env.configuration,sort_keys=True))
    else:
        with np.load(demonstrations,allow_pickle=False) as dataset:
            validate_arm_config(json.loads(str(dataset['configuration'])),env.configuration)
            observations=dataset['observations'].copy(); actions=dataset['actions'].copy()
            corrected=dataset['corrected'].copy(); runs=json.loads(str(dataset['runs']))
    prefix_ids=np.flatnonzero(~corrected)
    release_ids=np.flatnonzero(corrected)
    if not len(prefix_ids) or not len(release_ids):
        raise ValueError('Release distillation requires both learned-prefix and corrected transitions')
    targets=torch.tensor(actions)
    gain=(getattr(model,'arm_calibration',None) or {}).get('jaw_output_gain',1.)
    targets[corrected,3]*=gain
    with torch.no_grad():
        features=torch.cat([model.policy.extract_features(torch.tensor(batch))
                            for batch in np.array_split(observations,max(1,len(observations)//128))])
    trainable=[*model.policy.mlp_extractor.policy_net.parameters(),*model.policy.action_net.parameters()]
    optimizer=torch.optim.Adam(trainable,lr=learning_rate)
    for index in range(updates):
        ids=np.r_[prefix_ids[torch.randint(len(prefix_ids),(128,)).numpy()],
                  release_ids[torch.randint(len(release_ids),(128,)).numpy()]]
        predicted=model.policy.action_net(model.policy.mlp_extractor.forward_actor(features[ids]))
        loss=(predicted-targets[ids]).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable,1.)
        optimizer.step()
        if (index+1)%250==0:
            print('Release distillation',index+1,float(loss.detach()),flush=True)
    model.save(output/'policy')
    score=evaluate(model,env,10,2000)
    report=dict(operation='training-only release correction with learned-prefix distillation',
                environment=env.configuration,ppo_steps=model.num_timesteps,
                imitation_updates=updates,bc_learning_rate=learning_rate,training_seed=7,
                feature_extractor_frozen=True,training_examples=len(observations),
                corrected_examples=int(corrected.sum()),correction_rollouts=runs,
                demonstrations=str(demonstrations) if demonstrations else None,
                source_calibration=getattr(model,'arm_calibration',None),evaluation=score,
                checkpoint_sha256=hashlib.sha256((output/'policy.zip').read_bytes()).hexdigest(),
                graph_sha256=model.policy.features_extractor.graph_sha256)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print('Distilled evaluation',score,flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--teacher-check',action='store_true')
    p.add_argument('--steps',type=int,default=8192)
    p.add_argument('--updates',type=int,default=3000)
    p.add_argument('--bc-lr',type=float,default=5e-4)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--dagger-rounds',type=int,default=0)
    p.add_argument('--dagger-episodes',type=int,default=5)
    p.add_argument('--dagger-updates',type=int,default=1000)
    p.add_argument('--demonstrations',type=Path,help='Reuse a collected demonstration dataset')
    p.add_argument('--resume',type=Path,help='Initialize from an existing arm checkpoint')
    p.add_argument('--calibrate-jaw',type=float,help='Only scale a saved policy jaw output and evaluate it; requires --resume')
    p.add_argument('--correct-release',action='store_true',help='Collect training-only corrections on resumed policy states')
    p.add_argument('--distill-release',action='store_true',help='Fine-tune only the actor on successful release corrections and raw learned-prefix targets; requires --resume and --steps 0')
    p.add_argument('--correction-episodes',type=int,default=5)
    p.add_argument('--station',choices=['pick','cubes'],default='pick')
    p.add_argument('--history',type=int,default=1,help='Number of causal observations per action (1 to 64)')
    p.add_argument('--motion-deadband',type=float,help='Cartesian output deadband; defaults to zero for training or the saved value for calibration')
    p.add_argument('--gripper',choices=['padded','urdf','parallel'],default='padded',
        help="Gripper model: 'padded' is the supplied hooked gripper with contact pads; 'urdf' is that gripper untouched, which does not grasp; 'parallel' is the sliding-jaw substitution the recorded RL results used")
    p.add_argument('--output',type=Path,default=Path('out/rl/arm_observable'))
    a=p.parse_args()
    torch.set_num_threads(2)
    env=ArmEnv(gripper=a.gripper,station=a.station,history=a.history,
               motion_deadband=0. if a.motion_deadband is None else a.motion_deadband)
    if a.teacher_check:
        print(json.dumps(evaluate(None,env,3),indent=2),flush=True); return
    if a.distill_release:
        if not a.resume or a.steps != 0 or a.correct_release or a.calibrate_jaw is not None or a.dagger_rounds:
            p.error('--distill-release requires --resume and --steps 0, without other training/calibration modes')
        if a.correction_episodes < 1 or a.updates < 1 or not np.isfinite(a.bc_lr) or a.bc_lr <= 0:
            p.error('Release distillation requires positive episodes, updates, and learning rate')
        model=PPO.load(a.resume,device='cpu')
        validate_arm_config(getattr(model,'arm_config',None),env.configuration)
        report=distill_release(model,env,a.output,a.correction_episodes,a.updates,a.bc_lr,a.demonstrations)
        report.update(resume=str(a.resume),resume_sha256=hashlib.sha256(a.resume.read_bytes()).hexdigest())
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        return
    if a.calibrate_jaw is not None:
        if not a.resume or not np.isfinite(a.calibrate_jaw) or a.calibrate_jaw <= 0:
            p.error('--calibrate-jaw requires --resume and a finite positive gain')
        model=PPO.load(a.resume,device='cpu')
        prior_config=getattr(model,'arm_config',None)
        prior_deadband=prior_config.get('motion_deadband',0.) if isinstance(prior_config,dict) else 0.
        prior_env=ArmEnv(gripper=a.gripper,station=a.station,history=a.history,motion_deadband=prior_deadband)
        validate_arm_config(prior_config,prior_env.configuration)
        if a.motion_deadband is None:
            env=prior_env
        calibrate_jaw(model,a.calibrate_jaw)
        model.arm_config=env.configuration
        a.output.mkdir(parents=True,exist_ok=True)
        model.save(a.output/'policy')
        score=evaluate(model,env,10,2000)
        report=dict(operation='post-training jaw-output and motion-deadband calibration',resume=str(a.resume),
                    environment=model.arm_config,calibration=model.arm_calibration,
                    ppo_steps=model.num_timesteps,evaluation=score,
                    graph_sha256=model.policy.features_extractor.graph_sha256)
        (a.output/'report.json').write_text(json.dumps(report,indent=2))
        print('Calibrated evaluation',score,flush=True)
        return
    a.output.mkdir(parents=True,exist_ok=True)
    demos=a.demonstrations or a.output/'demonstrations.npz'
    if not demos.exists():
        observations=[]; actions=[]; kept=0
        for seed in range(a.episodes):
            obs,_=env.reset(seed=seed); teacher=Teacher()
            episode_obs=[]; episode_actions=[]
            for _ in range(env.horizon):
                action=teacher.action(env)
                episode_obs.append(obs); episode_actions.append(action)
                obs,r,t,tr,info=env.step(action)
                if t or tr: break
            # Only imitate episodes that actually placed the cube. The teacher is a
            # scripted opener, not an oracle, and a failed run teaches the policy to
            # shove the cube off the table.
            if info['success']:
                observations+=episode_obs; actions+=episode_actions; kept+=1
            print('demonstration',seed,'kept' if info['success'] else 'discarded',info,flush=True)
        if not kept:
            raise RuntimeError('No successful demonstrations to imitate')
        print(f'kept {kept}/{a.episodes} demonstrations, {len(observations)} transitions',flush=True)
        np.savez_compressed(demos,observations=observations,actions=actions,
                            configuration=json.dumps(env.configuration,sort_keys=True))
    d=np.load(demos,allow_pickle=False)
    validate_arm_config(json.loads(str(d['configuration'])) if 'configuration' in d else None,
                        env.configuration)
    obs=torch.tensor(d['observations']); actions=torch.tensor(d['actions'])
    if a.resume and (a.resume.parent/'release_corrections.npz').exists():
        previous=np.load(a.resume.parent/'release_corrections.npz',allow_pickle=False)
        validate_arm_config(json.loads(str(previous['configuration'])) if 'configuration' in previous else None,
                            env.configuration)
        selected=previous['corrected'] if 'corrected' in previous else np.ones(len(previous['actions']),dtype=bool)
        obs=torch.cat([obs,torch.tensor(previous['observations'][selected]).repeat(8,1)])
        actions=torch.cat([actions,torch.tensor(previous['actions'][selected]).repeat(8,1)])
    model=RehearsalPPO('MlpPolicy',env,policy_kwargs=dict(features_extractor_class=ConnectomeFeatures,
        net_arch=dict(pi=[128,128],vf=[128,128]),ortho_init=False),learning_rate=1e-5,
        n_steps=512,batch_size=128,n_epochs=3,vf_coef=.01,ent_coef=0.,seed=7,verbose=0)
    if a.resume:
        prior=PPO.load(a.resume,device='cpu')
        validate_arm_config(getattr(prior,'arm_config',None),env.configuration)
        model.policy.load_state_dict(prior.policy.state_dict())
    model.arm_config=env.configuration
    if a.correct_release:
        if not a.resume:
            p.error('--correct-release requires --resume')
        extra_obs=[]; extra_actions=[]
        for seed in range(30,30+a.correction_episodes):
            state,_=env.reset(seed=seed)
            correction_teacher=ReleaseCorrection()
            episode_obs=[]; episode_actions=[]
            for _ in range(env.horizon):
                action=model.predict(state,deterministic=True)[0]
                correction=correction_teacher.action(env)
                if correction is not None:
                    episode_obs.append(state.copy()); episode_actions.append(correction)
                    # Execute corrective demonstrations only in this training data pass.
                    action=correction
                state,_,t,tr,info=env.step(action)
                if t or tr: break
            if info['success']:
                extra_obs.extend(episode_obs); extra_actions.extend(episode_actions)
            print('Release correction',seed,'kept' if info['success'] else 'discarded',info,flush=True)
        if not extra_obs:
            raise RuntimeError('No successful release corrections to imitate')
        np.savez_compressed(a.output/'release_corrections.npz',observations=extra_obs,actions=extra_actions,
                            configuration=json.dumps(env.configuration,sort_keys=True))
        obs=torch.cat([obs,torch.tensor(np.array(extra_obs)).repeat(8,1)])
        actions=torch.cat([actions,torch.tensor(np.array(extra_actions)).repeat(8,1)])
        print('Release correction states',len(extra_obs),flush=True)
    with torch.no_grad(): model.policy.log_std.fill_(-3.5)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=a.bc_lr)
    for i in range(a.updates):
        ids=torch.randint(len(obs),(256,))
        loss=(model.policy.get_distribution(obs[ids]).distribution.mean-actions[ids]).square().mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.0)
        optimizer.step()
        if (i+1)%250==0: print('BC',i+1,float(loss.detach()),flush=True)
    dagger=[]
    rng=np.random.default_rng(7)
    for round_index in range(a.dagger_rounds):
        extra_obs=[]; extra_actions=[]
        for seed in range(100+round_index*a.dagger_episodes,100+(round_index+1)*a.dagger_episodes):
            state,_=env.reset(seed=seed); teacher=Teacher()
            for _ in range(env.horizon):
                expert=teacher.action(env)
                learned=model.predict(state,deterministic=True)[0]
                extra_obs.append(state.copy()); extra_actions.append(expert)
                action=expert if rng.random()<.5/(round_index+1) else learned
                state,_,t,tr,info=env.step(action)
                if t or tr: break
            print('DAgger rollout',round_index,seed,info,flush=True)
        obs=torch.cat([obs,torch.tensor(np.array(extra_obs))])
        actions=torch.cat([actions,torch.tensor(np.array(extra_actions))])
        for _ in range(a.dagger_updates):
            ids=torch.randint(len(obs),(256,))
            loss=(model.policy.get_distribution(obs[ids]).distribution.mean-actions[ids]).square().mean()
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.0)
            optimizer.step()
        score=evaluate(model,env,3)
        dagger.append(score)
        print('DAgger validation',round_index,score,flush=True)
        model.save(a.output/f'dagger_{round_index}')
    model.save(a.output/'imitation')
    before=evaluate(model,env,5)
    print('Before PPO',before,flush=True)
    model.demo_observations=obs; model.demo_actions=actions
    model.rehearsal_updates=32
    model.rehearsal_optimizer=torch.optim.Adam(model.policy.parameters(),lr=1e-4)
    model.learn(a.steps,callback=Progress())
    model.save(a.output/'policy')
    after=evaluate(model,env,10,2000)
    report=dict(ppo_steps=model.num_timesteps,imitation_updates=a.updates,before_ppo=before,after_ppo=after,
                environment=model.arm_config,dagger=dagger,
                resume=str(a.resume) if a.resume else None,bc_learning_rate=a.bc_lr,
                release_corrections=a.correct_release,training_examples=len(obs),
                graph_sha256=model.policy.features_extractor.graph_sha256,
                ppo_metrics={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/')})
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print('After PPO',after,flush=True)


if __name__=='__main__': main()
