"""Demonstration initialization and PPO for continuous Cartesian arm actions."""
import sys
from pathlib import Path
import argparse
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


def evaluate(model, env, count, start=1000):
    runs=[]
    for seed in range(start,start+count):
        obs,_=env.reset(seed=seed)
        teacher=Teacher()
        for _ in range(env.horizon):
            action=teacher.action(env) if model is None else model.predict(obs,deterministic=True)[0]
            obs,r,t,tr,info=env.step(action)
            if t or tr: break
        runs.append(dict(seed=seed,**info))
    return dict(successes=sum(r['success'] for r in runs),episodes=count,runs=runs)


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
    p.add_argument('--correct-release',action='store_true',help='Collect training-only corrections on resumed policy states')
    p.add_argument('--correction-episodes',type=int,default=5)
    p.add_argument('--station',choices=['pick','cubes'],default='pick')
    p.add_argument('--gripper',choices=['padded','urdf','parallel'],default='padded',
        help="Gripper model: 'padded' is the supplied hooked gripper with contact pads; 'urdf' is that gripper untouched, which does not grasp; 'parallel' is the sliding-jaw substitution the recorded RL results used")
    p.add_argument('--output',type=Path,default=Path('out/rl/arm_observable'))
    a=p.parse_args()
    torch.set_num_threads(2)
    env=ArmEnv(gripper=a.gripper,station=a.station)
    if a.teacher_check:
        print(json.dumps(evaluate(None,env,3),indent=2),flush=True); return
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
        obs=torch.cat([obs,torch.tensor(previous['observations']).repeat(8,1)])
        actions=torch.cat([actions,torch.tensor(previous['actions']).repeat(8,1)])
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
            for _ in range(env.horizon):
                action=model.predict(state,deterministic=True)[0]
                distance=np.linalg.norm(env.cube[:2]-env.goal[:2])
                if distance<.025 and env.cube[2]<.744 and env.max_lift>.08:
                    correction=np.array([0,0,0,1],dtype=np.float32)
                    if env.data.qpos[env.gripq]>env.released_qpos:
                        correction[2]=np.clip((.852-env.target[2])/.004,-1,1)
                    extra_obs.append(state.copy()); extra_actions.append(correction)
                    # Execute corrective demonstrations only in this training data pass.
                    action=correction
                state,_,t,tr,_=env.step(action)
                if t or tr: break
        if not extra_obs:
            raise RuntimeError('No release correction states reached')
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
        optimizer.zero_grad(); loss.backward(); optimizer.step()
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
            optimizer.zero_grad(); loss.backward(); optimizer.step()
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
