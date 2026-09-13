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
from rlbot.arm_env import ArmEnv
from rlbot.connectome import ConnectomeFeatures
from train_connectome import RehearsalPPO, Progress


class Teacher:
    """Training data only; deliberately absent from ArmEnv and policy inference."""
    def __init__(self):
        self.phase = 0
        self.count = 0

    def action(self, env):
        cube = env.cube
        pick = env.start + [0,0,.032]
        place = env.goal + [0,0,.008]
        targets = [pick, pick, pick+[0,0,.15], place+[0,0,.15], place, place, place+[0,0,.12]]
        grip = [1,-1,-1,-1,-1,1,1][self.phase]
        target = targets[self.phase]
        close = np.linalg.norm(env.arm.grip_pos-target)<.006
        self.count = self.count+1 if close else 0
        wait = [8,20,8,8,12,20,30][self.phase]
        if self.count >= wait and self.phase<6:
            self.phase += 1
            self.count = 0
        return np.r_[np.clip((target-env.target)/.004,-1,1),grip].astype(np.float32)


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
    p.add_argument('--demonstrations',type=Path,help='Reuse a collected demonstration dataset')
    p.add_argument('--resume',type=Path,help='Initialize from an existing arm checkpoint')
    p.add_argument('--correct-release',action='store_true',help='Collect training-only corrections on resumed policy states')
    p.add_argument('--correction-episodes',type=int,default=5)
    p.add_argument('--output',type=Path,default=Path('out/rl/arm_friction3'))
    a=p.parse_args()
    torch.set_num_threads(2)
    env=ArmEnv()
    if a.teacher_check:
        print(json.dumps(evaluate(None,env,3),indent=2),flush=True); return
    a.output.mkdir(parents=True,exist_ok=True)
    demos=a.demonstrations or a.output/'demonstrations.npz'
    if not demos.exists():
        observations=[]; actions=[]
        for seed in range(a.episodes):
            obs,_=env.reset(seed=seed); teacher=Teacher()
            for _ in range(env.horizon):
                action=teacher.action(env)
                observations.append(obs); actions.append(action)
                obs,r,t,tr,info=env.step(action)
                if t or tr: break
            print('demonstration',seed,info,flush=True)
        np.savez_compressed(demos,observations=observations,actions=actions)
    d=np.load(demos)
    obs=torch.tensor(d['observations']); actions=torch.tensor(d['actions'])
    if a.resume and (a.resume.parent/'release_corrections.npz').exists():
        previous=np.load(a.resume.parent/'release_corrections.npz')
        obs=torch.cat([obs,torch.tensor(previous['observations']).repeat(8,1)])
        actions=torch.cat([actions,torch.tensor(previous['actions']).repeat(8,1)])
    model=RehearsalPPO('MlpPolicy',env,policy_kwargs=dict(features_extractor_class=ConnectomeFeatures,
        net_arch=dict(pi=[128,128],vf=[128,128]),ortho_init=False),learning_rate=1e-5,
        n_steps=512,batch_size=128,n_epochs=3,vf_coef=.01,ent_coef=0.,seed=7,verbose=0)
    if a.resume:
        model.policy.load_state_dict(PPO.load(a.resume,device='cpu').policy.state_dict())
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
                    if env.data.qpos[env.gripq]>.037:
                        correction[2]=np.clip((.852-env.target[2])/.004,-1,1)
                    extra_obs.append(state.copy()); extra_actions.append(correction)
                    # Execute corrective demonstrations only in this training data pass.
                    action=correction
                state,_,t,tr,_=env.step(action)
                if t or tr: break
        if not extra_obs:
            raise RuntimeError('No release correction states reached')
        np.savez_compressed(a.output/'release_corrections.npz',observations=extra_obs,actions=extra_actions)
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
                resume=str(a.resume) if a.resume else None,bc_learning_rate=a.bc_lr,
                release_corrections=a.correct_release,training_examples=len(obs),
                graph_sha256=model.policy.features_extractor.graph_sha256,
                ppo_metrics={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/')})
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    print('After PPO',after,flush=True)


if __name__=='__main__': main()
