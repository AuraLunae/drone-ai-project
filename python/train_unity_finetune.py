"""
Unity ML-Agentsでのファインチューニングスクリプト(追加学習オプション + タイムスタンプrunフォルダ対応版)

- PyBulletで学習したVecNormalize統計をそのまま引き継ぐ
  (これをやらないとPyBulletで学んだ観測スケール感覚がUnityで崩れる)
- target_klで急激な方策崩壊(catastrophic forgetting)を抑制
- --model / --vecnormalize を省略すると、models/runs/ 以下の最新run(train_pretrain.py
  またはこのスクリプト自身が生成したもの)を自動的に継続元として使用する
- 1回の実行の成果物(モデル・VecNormalize統計・TensorBoardログ)は
  `models/runs/finetune_<タイムスタンプ>/` に一式まとめて保存され、
  過去の結果を上書きしない

使い方:
  # 初回/追加学習とも: 省略時は models/runs/ 以下の最新成果から自動で継続
  python train_unity_finetune.py

  # 継続元を明示指定したい場合
  python train_unity_finetune.py \\
      --model models/runs/pretrain_20260927_153012/model.zip \\
      --vecnormalize models/runs/pretrain_20260927_153012/vecnormalize.pkl \\
      --timesteps 300000

  # 学習率を変えて追加学習したい場合
  python train_unity_finetune.py --learning-rate 5e-5 --timesteps 200000
"""

import os
import argparse
import datetime
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper  # gym-unity廃止に伴い変更
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

N_STACK = 10
RUNS_BASE_DIR = os.path.join("models", "runs")
# 制御周波数を5倍化した分、ファインチューニングのステップ数も引き上げる
TOTAL_TIMESTEPS = 750_000

# 旧バージョン(タイムスタンプrunフォルダ導入前)との互換用フォールバックパス
LEGACY_MODEL_PATH = os.path.join("models", "pretrained_drone_ppo.zip")
LEGACY_VECNORM_PATH = os.path.join("models", "vecnormalize.pkl")


def find_latest_run(base_dir: str = RUNS_BASE_DIR):
    """base_dir以下(pretrain_*/finetune_*問わず)で model.zip と vecnormalize.pkl が
    両方そろっている最新runのパスを返す。無ければNone。"""
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


def make_run_dir(prefix: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_BASE_DIR, f"{prefix}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Unity ML-Agentsでのファインチューニング(追加学習 + タイムスタンプrunフォルダ対応)")
    parser.add_argument(
        "--model", type=str, default=None,
        help="ロードする学習済みモデルのパス(.zip込み)。省略時は models/runs/ 以下の"
             "最新runを自動的に使用する(見つからない場合は旧バージョンの固定パスにフォールバック)。",
    )
    parser.add_argument(
        "--vecnormalize", type=str, default=None,
        help="ロードするVecNormalize統計(.pkl)のパス。省略時は --model と同じrunフォルダ内、"
             "または自動検出したrunフォルダ内のvecnormalize.pklを使用する。",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"今回の(追加)学習で回すステップ数(デフォルト: {TOTAL_TIMESTEPS})",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学習率(デフォルト: 1e-4)")
    parser.add_argument("--target-kl", type=float, default=0.02, help="方策急変を抑えるtarget_kl(デフォルト: 0.02)")
    return parser.parse_args()


def resolve_inputs(args):
    """--model / --vecnormalize の指定に応じて、実際にロードするパスを確定する。
    優先順位: (1) 明示指定 > (2) models/runs/ 以下の最新run > (3) 旧バージョン固定パス"""
    if args.model is None:
        latest = find_latest_run()
        if latest is not None:
            args.model = os.path.join(latest, "model.zip")
            if args.vecnormalize is None:
                args.vecnormalize = os.path.join(latest, "vecnormalize.pkl")
            print(f"=== --model省略: 最新run {latest} を自動的に継続元として使用します ===")
        elif os.path.exists(LEGACY_MODEL_PATH):
            args.model = LEGACY_MODEL_PATH
            if args.vecnormalize is None:
                args.vecnormalize = LEGACY_VECNORM_PATH
            print(f"=== --model省略: 旧バージョンの固定パス {LEGACY_MODEL_PATH} にフォールバックします ===")
        else:
            raise SystemExit(
                f"継続元モデルが見つかりません。先に train_pretrain.py を実行するか、"
                "--model / --vecnormalize で明示的に指定してください。"
            )
    elif args.vecnormalize is None:
        # --modelだけ明示指定された場合、同じrunフォルダ内のvecnormalize.pklを自動推測する
        guessed = os.path.join(os.path.dirname(args.model), "vecnormalize.pkl")
        if os.path.exists(guessed):
            args.vecnormalize = guessed
        else:
            raise SystemExit("--vecnormalizeが指定されておらず、自動推測もできませんでした。明示的に指定してください。")
    return args


def main():
    args = parse_args()
    args = resolve_inputs(args)

    run_dir = make_run_dir("finetune")
    print(f"=== 今回の学習成果はすべて次のフォルダにまとめて保存されます: {run_dir} ===")

    print("=== UnityエディタのPlayボタン押下を待機中... ===")
    unity_env = UnityEnvironment(file_name=None, seed=1)
    gym_env = UnityToGymWrapper(unity_env, uint8_visual=False)
    env = DummyVecEnv([lambda: gym_env])
    env = VecFrameStack(env, n_stack=N_STACK)

    # 指定されたVecNormalize統計をロードして引き継ぐ(初回はPyBullet産、追加学習時はUnity産でもよい)
    print(f"=== VecNormalize統計をロード: {args.vecnormalize} ===")
    env = VecNormalize.load(args.vecnormalize, env)
    env.training = True   # Unity側のデータでも統計を更新し続ける
    env.norm_reward = True

    print(f"=== モデルをロード中: {args.model} ===")
    model = PPO.load(
        args.model,
        env=env,
        learning_rate=args.learning_rate,   # 重み破壊を防ぐため学習率を抑える
        target_kl=args.target_kl,           # 方策の急激な変化を制限
        tensorboard_log=os.path.join(run_dir, "tb"),
        device="cuda",
    )

    print(f"=== Unity上での(追加)学習を開始します ({args.timesteps}ステップ) ===")
    # reset_num_timesteps=False: 追加学習を繰り返してもTensorBoardのステップ数は通算表示になる
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)

    model_path = os.path.join(run_dir, "model")
    model.save(model_path)
    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)

    print(f"=== モデルを保存しました: {model_path}.zip ===")
    print(f"=== 正規化統計を保存しました: {vecnorm_path} ===")
    print(f"=== このモデルから続けて追加学習するには: python train_unity_finetune.py --model {model_path}.zip --vecnormalize {vecnorm_path} ===")

    env.close()


if __name__ == "__main__":
    main()