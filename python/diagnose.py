# python/diagnose.py
import numpy as np, pybullet as p
from drone_pretrain_env import PyBulletDroneEnv

env = PyBulletDroneEnv(render=False)
env.reset(seed=0)
print("mass/inertia:", p.getDynamicsInfo(env.drone, -1)[0], p.getDynamicsInfo(env.drone, -1)[2])

# 全モーター均等のホバリング推力(mg/4 ≒ 2.45N → 指令0.41 → action≒-0.18)
hover = np.full(4, -0.18, dtype=np.float32)
for name, noise in [("均等", 0.0), ("m0だけ+0.1", 0.1)]:
    env.reset(seed=0)
    for t in range(50):
        a = hover.copy(); a[0] += noise
        obs, r, term, trunc, _ = env.step(a)
        if term: break
    print(name, "終了step:", t + 1, "角速度:", np.round(obs[10:13], 2), "高度差:", round(float(obs[1]), 2))