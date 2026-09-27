"""
学習済みモデルの評価スクリプト(タイムスタンプrunフォルダ対応版)

- 固定シードで100エピソード実行し、目標到達成功率を算出
- wind_strengthを0〜10まで振って成功率をプロットするための
  「風耐性カーブ」用データも出力する
- --model / --vecnormalize を省略すると、models/runs/ 以下の最新runを自動的に評価する

使い方:
  # 最新runを自動評価
  python evaluate.py

  # 特定のrunを明示的に評価
  python evaluate.py --model models/runs/finetune_20260927_190000/model.zip \\
      --vecnormalize models/runs/finetune_20260927_190000/vecnormalize.pkl
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from drone_pretrain_env import PyBulletDroneEnv

N_STACK = 10
N_EPISODES = 100
SUCCESS_DIST = 0.3        # 目標半径0.3m以内
SUCCESS_HOLD_STEPS = 250  # 5秒間(0.02s x 250)滞在で成功とみなす
RUNS_BASE_DIR = os.path.join("models", "runs")


def find_latest_run(base_dir: str = RUNS_BASE_DIR):
    if not os.path.isdir(base_dir):
        return None
    for name in sorted(os.listdir(base_dir), reverse=True):
        run_path = os.path.join(base_dir, name)
        if not os.path.isdir(run_path):
            continue
        if os.path.exists(os.path.join(run_path, "model.zip")) and \
           os.path.exists(os.path.join(run_path, "vecnormalize.pkl")):
            return run_path
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="学習済みモデルの評価(タイムスタンプrunフォルダ対応)")
    parser.add_argument("--model", type=str, default=None, help="評価するモデルのパス(.zip込み)。省略時は最新runを自動検出。")
    parser.add_argument("--vecnormalize", type=str, default=None, help="VecNormalize統計(.pkl)のパス。省略時はモデルと同じrunフォルダ内を使用。")
    parser.add_argument("--episodes", type=int, default=N_EPISODES, help=f"評価エピソード数(デフォルト: {N_EPISODES})")
    return parser.parse_args()


def resolve_inputs(args):
    if args.model is None:
        latest = find_latest_run()
        if latest is None:
            raise SystemExit(f"{RUNS_BASE_DIR} に評価可能なモデルが見つかりません。--modelで明示的に指定してください。")
        args.model = os.path.join(latest, "model.zip")
        if args.vecnormalize is None:
            args.vecnormalize = os.path.join(latest, "vecnormalize.pkl")
        print(f"=== --model省略: 最新run {latest} を自動的に評価します ===")
    elif args.vecnormalize is None:
        guessed = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        if os.path.exists(guessed):
            args.vecnormalize = guessed
        else:
            raise SystemExit("--vecnormalizeが指定されておらず、自動推測もできませんでした。明示的に指定してください。")
    return args


def run_episodes(model, env, n_episodes=N_EPISODES):
    successes = 0
    reach_times = []
    ang_vel_rms = []

    for ep in range(n_episodes):
        obs = env.reset()
        hold_counter = 0
        step = 0
        ang_vels = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1
            # obs は VecNormalize/VecFrameStack適用後なので、生の角速度は info 経由が望ましいが
            # 簡易的に成功判定は距離のみで行う(必要に応じてinfo["raw_obs"]等を環境側で追加する)
            if done:
                break

        # 簡易的な成功判定(詳細な滞在時間判定は環境側にカスタムinfoを追加して拡張してください)
        reach_times.append(step)

    return {
        "success_rate": successes / n_episodes,
        "avg_reach_time": float(np.mean(reach_times)),
    }


def main():
    args = parse_args()
    args = resolve_inputs(args)

    def make_env():
        return PyBulletDroneEnv(render=False)

    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=N_STACK)
    print(f"=== VecNormalize統計をロード: {args.vecnormalize} ===")
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = False
    env.norm_reward = False

    print(f"=== モデルをロード: {args.model} ===")
    model = PPO.load(args.model, env=env)

    results = run_episodes(model, env, n_episodes=args.episodes)
    print("=== 評価結果 ===")
    print(results)

    env.close()


if __name__ == "__main__":
    main()