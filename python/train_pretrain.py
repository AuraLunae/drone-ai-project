"""
PyBulletでの事前学習スクリプト(追加学習オプション + タイムスタンプrunフォルダ対応版)

- 16並列のヘッドレス(DIRECT)環境で高速化
- VecNormalizeで観測/報酬を正規化し、統計をUnity側に引き継ぐ
- VecFrameStack(10)で時間方向の情報を付与(Unity側 Stacked Vectors=10と一致)
- CheckpointCallback / EvalCallbackで途中経過を必ず保存
- --resume オプションで、既存のモデル(チェックポイント含む)から追加学習できる
- 1回の実行で生成される成果物(モデル・VecNormalize統計・チェックポイント・
  ベストモデル・TensorBoardログ)は、すべて `models/runs/pretrain_<タイムスタンプ>/`
  という1つのフォルダにまとめて保存される。固定ファイル名への上書きを行わないため、
  過去の学習結果が消えたり、実行中のファイルと衝突してエラーになることがない。

使い方:
  # 新規に学習を開始(自動的に models/runs/pretrain_20260927_153012/ 等が作られる)
  python train_pretrain.py

  # チェックポイントから追加学習(パスを明示指定)
  python train_pretrain.py --resume models/runs/pretrain_20260927_153012/checkpoints/pretrain_500000_steps.zip --timesteps 2000000

  # 直近の学習成果(models/runs/以下で最新のもの)から自動的に追加学習
  python train_pretrain.py --resume latest --timesteps 1000000
"""

import os
import argparse
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from drone_pretrain_env import PyBulletDroneEnv

N_ENVS = 16
N_STACK = 10
RUNS_BASE_DIR = os.path.join("models", "runs")
# 制御周波数をUnity(50Hz)に合わせたことで1エピソードあたりのRLステップ数が
# 旧版(10Hz相当)の5倍になったため、目標ステップ数も5倍程度に引き上げる
TOTAL_TIMESTEPS = 5_000_000


def find_latest_run(base_dir: str = RUNS_BASE_DIR):
    """base_dir以下で model.zip と vecnormalize.pkl が両方そろっている
    最新(フォルダ名の降順で最初に見つかった)runのパスを返す。無ければNone。
    フォルダ名の先頭がタイムスタンプ(YYYYMMDD_HHMMSS)なので、文字列ソートの
    降順=時系列の新しい順になる。"""
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
    """models/runs/<prefix>_<タイムスタンプ>/ を作成して返す(checkpoints/best/tb込み)。
    1回の学習で生成される成果物をすべてこのフォルダにまとめることで、
    過去の学習結果を上書き/混在させず、追加学習を安全に繰り返せるようにする。"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RUNS_BASE_DIR, f"{prefix}_{timestamp}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "best"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "tb"), exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="PyBulletでの事前学習(追加学習 + タイムスタンプrunフォルダ対応)")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="追加学習の継続元となるモデルzipのパス。特別な値 'latest' を指定すると "
             f"{RUNS_BASE_DIR} 以下の最新runを自動的に継続元として使用する。省略時は新規学習。",
    )
    parser.add_argument(
        "--vecnormalize", type=str, default=None,
        help="--resume指定時に引き継ぐVecNormalize統計(.pkl)のパス。"
             "省略時は継続元runフォルダ内の vecnormalize.pkl を自動使用する。",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"今回の学習(または追加学習)で回すステップ数(デフォルト: {TOTAL_TIMESTEPS})",
    )
    parser.add_argument("--n-envs", type=int, default=N_ENVS, help="並列環境数(デフォルト: 16)")
    return parser.parse_args()


def make_env(rank: int, seed: int = 0):
    def _init():
        env = PyBulletDroneEnv(render=False)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    args = parse_args()

    # --resume latest: models/runs/ 以下の最新runを自動的に継続元にする
    if args.resume == "latest":
        latest = find_latest_run()
        if latest is None:
            raise SystemExit(
                f"{RUNS_BASE_DIR} に有効な学習成果が見つかりません。"
                "--resumeを外して新規学習するか、明示的にパスを指定してください。"
            )
        args.resume = os.path.join(latest, "model.zip")
        if args.vecnormalize is None:
            args.vecnormalize = os.path.join(latest, "vecnormalize.pkl")
        print(f"=== --resume latest: {latest} を継続元として使用します ===")
    elif args.resume and args.vecnormalize is None:
        # 明示パス指定時、同じrunフォルダ内のvecnormalize.pklを自動推測する
        guessed = os.path.join(os.path.dirname(args.resume), "vecnormalize.pkl")
        if os.path.exists(guessed):
            args.vecnormalize = guessed

    run_dir = make_run_dir("pretrain")
    print(f"=== 今回の学習成果はすべて次のフォルダにまとめて保存されます: {run_dir} ===")

    set_random_seed(0)

    # 1. 学習用: 指定した並列数のヘッドレス環境
    env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)])
    env = VecFrameStack(env, n_stack=N_STACK)

    # 2. 評価用: 学習統計に影響しない別シードの単一環境
    eval_env = DummyVecEnv([make_env(999)])
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    if args.resume:
        if args.vecnormalize is None:
            raise SystemExit("--resume指定時はVecNormalize統計が必要です。--vecnormalizeで明示指定してください。")
        # --- 追加学習モード: 既存のVecNormalize統計を引き継ぐ ---
        print(f"=== 追加学習: VecNormalize統計をロード: {args.vecnormalize} ===")
        env = VecNormalize.load(args.vecnormalize, env)
        env.training = True
        env.norm_reward = True

        eval_env = VecNormalize.load(args.vecnormalize, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    else:
        # --- 新規学習モード ---
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="pretrain",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best"),
        log_path=os.path.join(run_dir, "tb"),
        eval_freq=max(20_000 // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
    )

    if args.resume:
        print(f"=== 追加学習: モデルをロード: {args.resume} ===")
        model = PPO.load(
            args.resume,
            env=env,
            tensorboard_log=os.path.join(run_dir, "tb"),
            device="cuda",
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            clip_range=0.2,
            tensorboard_log=os.path.join(run_dir, "tb"),
            device="cuda",
        )

    mode_label = "追加学習" if args.resume else "新規学習"
    print(f"=== PyBulletでの{mode_label}を開始します (DIRECTモード, {args.n_envs}並列, {args.timesteps}ステップ) ===")
    # 追加学習時はreset_num_timesteps=Falseにして、TensorBoardのステップ数を通算表示にする
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_callback, eval_callback],
        reset_num_timesteps=not args.resume,
    )

    model_path = os.path.join(run_dir, "model")
    model.save(model_path)
    vecnorm_path = os.path.join(run_dir, "vecnormalize.pkl")
    env.save(vecnorm_path)

    print(f"=== モデルを保存しました: {model_path}.zip ===")
    print(f"=== 正規化統計を保存しました: {vecnorm_path} ===")
    print(f"=== このモデルから続けて追加学習するには: --resume {model_path}.zip (または --resume latest) ===")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()