"""Run a saved arm policy alone, with optional live viewer or recorded replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import webbrowser
import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rlbot.arm_env import ArmEnv
from pick_place import camera, jpeg


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='out/rl/arm_release/policy.zip')
    p.add_argument('--seed',type=int,default=2000)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--view',action='store_true')
    p.add_argument('--record',action='store_true')
    p.add_argument('--open',action='store_true')
    p.add_argument('--output',type=Path,default=Path('out/arm_rl_demo'))
    a=p.parse_args()
    torch.set_num_threads(2)
    policy=PPO.load(a.checkpoint,device='cpu')
    env=ArmEnv()
    frames=[]; gifs=[]; reports=[]
    options=mujoco.MjvOption(); options.geomgroup[3]=0
    overview=camera([-.26,-1.60,1.0],1.7,-60,-25)
    detail=camera([-.28,-1.72,.79],.65,100,-30)
    env.model.vis.map.znear=.003/env.model.stat.extent
    renderer=mujoco.Renderer(env.model,height=480,width=640) if a.record else None
    viewer=None
    if a.view:
        import mujoco.viewer as mjviewer
        viewer=mjviewer.launch_passive(env.model,env.data)
        viewer.cam.lookat[:]=overview.lookat
        viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation=1.7,-60,-25
        viewer.opt.geomgroup[3]=0
    try:
        for seed in range(a.seed,a.seed+a.episodes):
            obs,_=env.reset(seed=seed)
            started=time.monotonic()
            for step in range(env.horizon):
                action,_=policy.predict(obs,deterministic=True)
                obs,reward,terminated,truncated,info=env.step(action)
                if renderer and (step%2==0 or terminated or truncated):
                    images=[]
                    for cam in (overview,detail):
                        renderer.update_scene(env.data,camera=cam,scene_option=options)
                        scene=renderer.scene
                        for axis in range(2):
                            for sign in (-1,1):
                                pos=env.goal.copy(); pos[2]=.703; pos[axis]+=sign*.038
                                size=np.array([.038,.038,.001]); size[axis]=.001
                                mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_BOX,
                                    size,pos,np.eye(3).ravel(),np.array([.2,1,.6,1],dtype=np.float32))
                                scene.ngeom+=1
                        images.append(renderer.render().copy())
                    activity=policy.policy.features_extractor.last_activity[0].cpu().tolist()
                    frames.append(dict(**info,time=float(env.data.time),overview=jpeg(images[0]),
                        detail=jpeg(images[1]),action=action.tolist(),activity=activity))
                    gifs.append(Image.fromarray(images[0]))
                if viewer:
                    if not viewer.is_running(): break
                    viewer.sync()
                    time.sleep(max(0,env.data.time-(time.monotonic()-started)))
                if terminated or truncated: break
            reports.append(dict(seed=seed,**info))
            print(reports[-1],flush=True)
    finally:
        if renderer: renderer.close()
        if viewer: viewer.close()
    a.output.mkdir(parents=True,exist_ok=True)
    report=dict(checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                graph_sha256=policy.policy.features_extractor.graph_sha256,
                controller='learned continuous Cartesian and gripper policy',pad_sliding_friction=3.0,
                successes=sum(x['success'] for x in reports),
                episodes=len(reports),runs=reports)
    (a.output/'report.json').write_text(json.dumps(report,indent=2))
    if a.record:
        with np.load(policy.policy.features_extractor.graph_path) as graph:
            positions=graph['positions'].copy()
            positions=(positions-(positions.min(0)+positions.max(0))/2)/max(float(np.ptp(positions,axis=0).max()),1)
            edges=np.stack([graph['pre'][::4],graph['post'][::4]],axis=1).tolist()
        payload=json.dumps(dict(report=report,frames=frames,positions=positions.tolist(),edges=edges),separators=(',',':'))
        template=(ROOT/'demo/arm_rl.html').read_text(encoding='utf-8')
        (a.output/'index.html').write_text(template.replace('__ARM_DATA__',payload),encoding='utf-8')
        gifs[0].save(a.output/'demo.gif',save_all=True,append_images=gifs[1:],duration=100,loop=0)
        gifs[-1].save(a.output/'preview.png')
        if a.open: webbrowser.open((a.output/'index.html').resolve().as_uri())


if __name__=='__main__': main()
