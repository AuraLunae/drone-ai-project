"""
PyBullet ドローン事前学習環境（フルエンドツーエンド / 4モーター出力版）

Unity側 (DecisionRequester: Decision Period = 1, fixedDeltaTime = 0.02s) と
実時間ベースで完全に同期するよう設計。

行動空間の設計方針(本バージョンの核心):
- 旧版は「3軸の合成推力ベクトル」を行動とし、さらに +Y方向にホバリング分の
  推力を自動加算する“補助”が入っていた。これはAIの代わりにこちらが
  重力補償を肩代わりしてしまっており、フルエンドツーエンドとは言えなかった。
- 本版では、行動を「4基のモーターそれぞれの推力指令」に変更する。
  ホバリングに必要な出力バランス、ロール/ピッチ(モーター位置オフセットによる
  トルク)、ヨー(モーターの回転方向の違いによる反トルク)のすべてを、
  AI自身が観測から逆算して出力する必要がある。
- 力とトルクは「モーター位置(機体中心からのオフセット)」と「反トルク係数」を
  使って**自前で**計算する。合力は機体重心(COM)にそのまま加え、
  ロール/ピッチトルクは各モーターのオフセット×力の外積(cross product)、
  ヨートルクは回転方向差の反トルクとして、すべて明示的に合算してから
  applyExternalTorqueで一度に加える(PyBulletの「位置指定によるトルク自動導出」に
  依存すると挙動がバージョン/設定依存になりやすいため、確実性を優先して自前計算にした)。
- 機体質量はURDFの既定値に依存させず、`p.changeDynamics(..., mass=1.0)`で
  明示的に1.0kgへ固定する。`MAX_THRUST_PER_MOTOR`はこの1.0kgを前提に
  決めた値であり、Unity側Rigidbodyの`Mass: 1.0`と必ず一致させること。

その他の方針(旧版から継続):
- 物理シミュレーションの刻み幅(精度)と、行動決定の周期は分離する。
  1回のRLステップ = 5物理サブステップ = 0.02秒 (Unityの1フレームと同じ実時間)。
- 風はエピソード内で時間変化するOrnstein-Uhlenbeck過程（平均回帰する乱気流）。
- 初期位置・機体スケールもエピソードごとにランダム化（ドメインランダム化）。
- 報酬は距離だけでなく、姿勢安定性・角速度・行動の滑らかさも評価する。

注意(物理モデルの簡略化について):
- 見た目・当たり判定は `sphere2.urdf`(球体)のままだが、力の作用点だけを
  仮想的な4隅のモーター位置に設定することで、クアッドコプターのロール/ピッチ/
  ヨー特性を模擬している。球体の慣性テンソルは実機の機体形状と完全には一致
  しないため、厳密な実機再現が必要な場合はURDF自体をクアッドコプター形状に
  差し替えることを推奨する(「まず動くものを一つ完成させる」段階では本簡略化で十分)。
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class PyBulletDroneEnv(gym.Env):

    PHYSICS_DT = 1.0 / 250.0                                # PyBulletネイティブの物理刻み幅
    CONTROL_DT = 0.02                                        # Unity fixedDeltaTime と一致
    SUBSTEPS_PER_ACTION = round(CONTROL_DT / PHYSICS_DT)     # = 5
    MAX_EPISODE_STEPS = 500                                  # 500 x 0.02s = 10秒でタイムアウト

    # --- 4モーター クアッドコプター ダイナミクス パラメータ ---
    ARM_LENGTH = 0.15            # [m] 機体中心からモーターまでの距離
    MAX_THRUST_PER_MOTOR = 6.0   # [N] モーター1基あたりの最大推力(機体重量約1kg基準でTWR≈2.4)
    YAW_TORQUE_COEFF = 0.02      # 推力に対する反トルクの比例係数(モーターの空気抵抗トルクを模擬)

    # モーター配置(ローカル座標: X=右, Y=上, Z=前方) と回転方向
    # 対角のモーター同士が同じ回転方向になるよう配置し、ヨー制御を可能にする
    MOTOR_OFFSETS = np.array([
        [-ARM_LENGTH, 0.0,  ARM_LENGTH],   # m0: 前左 (CW)
        [ ARM_LENGTH, 0.0,  ARM_LENGTH],   # m1: 前右 (CCW)
        [ ARM_LENGTH, 0.0, -ARM_LENGTH],   # m2: 後右 (CW)
        [-ARM_LENGTH, 0.0, -ARM_LENGTH],   # m3: 後左 (CCW)
    ])
    MOTOR_SPIN = np.array([-1.0, 1.0, -1.0, 1.0])  # CW=-1, CCW=+1 (機体ローカルY軸まわりの反トルク符号)

    def __init__(self, render: bool = False):
        super().__init__()
        self._render = render
        self._client = None
        self.drone = None

        # 行動空間: 4基のモーター推力指令 [-1.0 ~ 1.0] (それぞれ独立)
        # -1.0 = 出力0%, +1.0 = 出力100%(MAX_THRUST_PER_MOTOR) にマッピングする
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        # 観測空間: 位置差分(3) + 姿勢(4) + 速度(3) + 角速度(3) = 13次元
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        self.target_pos = np.array([0.0, 2.0, 0.0])
        self._elapsed_steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        # 風パラメータ(OU過程) — reset毎に再抽選される
        self.wind_vec = np.zeros(3)
        self.wind_mean = np.zeros(3)
        self.wind_theta = 1.0
        self.wind_sigma = 1.0

    # ------------------------------------------------------------------ #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._client is None:
            self._client = p.connect(p.GUI if self._render else p.DIRECT)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.resetSimulation()
        p.setTimeStep(self.PHYSICS_DT)
        p.setGravity(0, -9.81, 0)  # Unityに合わせてY軸負の方向に重力
        p.loadURDF("plane.urdf")

        # 初期位置を目標周辺でランダム化
        init_offset = self.np_random.uniform(-0.5, 0.5, size=3)
        init_offset[1] = abs(init_offset[1])
        start_pos = [init_offset[0], 1.0 + init_offset[1], init_offset[2]]

        # 機体スケールを±10%ランダム化(空力特性のばらつきを模擬)
        scale = 1.0 + self.np_random.uniform(-0.1, 0.1)
        self.drone = p.loadURDF("sphere2.urdf", start_pos, globalScaling=0.2 * scale)
        # 質量をUnity側Rigidbody(Mass=1.0)と厳密に一致させる。
        # sphere2.urdfの既定質量やglobalScalingによる質量変化に依存しない。
        p.changeDynamics(self.drone, -1, mass=1.0)

        # 風: ベース風向・強さ・収束速度・乱れ幅をすべてエピソードごとに再抽選
        direction = self.np_random.normal(0, 1, 3)
        direction /= (np.linalg.norm(direction) + 1e-8)
        self.wind_mean = direction * self.np_random.uniform(3.0, 8.0)
        self.wind_theta = self.np_random.uniform(0.5, 2.0)
        self.wind_sigma = self.np_random.uniform(1.0, 4.0)
        self.wind_vec = np.zeros(3)

        self._elapsed_steps = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        return self._get_obs(), {}

    # ------------------------------------------------------------------ #
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang_vel = p.getBaseVelocity(self.drone)
        pos_diff = np.array(pos) - self.target_pos
        return np.concatenate([pos_diff, orn, vel, ang_vel], dtype=np.float32)

    def _update_wind(self, dt):
        """Ornstein-Uhlenbeck過程: wind_meanに向かって滑らかに収束しながら乱れる"""
        noise = self.np_random.normal(0, 1, 3)
        self.wind_vec += (
            self.wind_theta * (self.wind_mean - self.wind_vec) * dt
            + self.wind_sigma * np.sqrt(dt) * noise
        )

    def _apply_rotor_dynamics(self, thrusts):
        """
        4基のモーター推力[N]から、合力とロール/ピッチ/ヨーの合トルクを
        すべて自前で計算して機体に加える(PyBulletの位置指定による自動トルク
        導出には依存しない)。

        - 各モーターの推力ベクトル(機体ローカル+Y方向)をまず計算し、4基分を
          合力としてそのまま重心(COM)に加える(applyExternalForceのposObjに
          機体位置そのものを渡すため、この力自体からは追加のトルクは発生しない)。
        - ロール/ピッチトルクは「モーター位置オフセット × 推力ベクトル」の外積
          (cross product)を4基分合算して明示的に計算する。
        - ヨートルクは各モーターの回転方向(CW/CCW)による反トルクを合算する。
        - 上記すべてを合算したトルクを、力とは別にapplyExternalTorqueで一度だけ加える。

        注意: この`np.cross`による外積計算式自体はPyBullet(右手系)/Unity(左手系)の
        どちらでも同一だが、その先でAddTorque/applyExternalTorqueに渡した際の
        実際の回転方向は物理エンジンの内部実装依存で反転しうる。本環境(PyBullet)を
        「正」の基準とし、Unity側は`DroneAgent.cs`の`TorqueSignCorrection`定数で
        符号を合わせる(検証手順は「Step 3.5: モーター単体テストによる符号検証」参照)。
        """
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        body_up_world = rot_matrix @ np.array([0.0, 1.0, 0.0])

        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        for i in range(4):
            world_offset = rot_matrix @ self.MOTOR_OFFSETS[i]
            force = body_up_world * thrusts[i]
            total_force += force
            # ロール/ピッチ: 位置オフセットと力の外積を自前で合算
            total_torque += np.cross(world_offset, force)
            # ヨー: 回転方向差による反トルクを合算
            total_torque += body_up_world * (self.MOTOR_SPIN[i] * thrusts[i] * self.YAW_TORQUE_COEFF)

        # 合力は重心(機体位置そのもの)に加える → この呼び出し自体からは追加トルクは発生しない
        p.applyExternalForce(self.drone, -1, total_force.tolist(), pos, p.WORLD_FRAME)
        # 合トルクは自前計算した値をまとめて加える
        p.applyExternalTorque(self.drone, -1, total_torque.tolist(), p.WORLD_FRAME)

    # ------------------------------------------------------------------ #
    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        # [-1, 1] -> [0, 1] のモーター指令に変換(0=停止, 1=最大出力)
        # 重力補正などの補助は一切行わない。ホバリングに必要な出力配分は
        # すべてAI自身が学習する。
        motor_cmd = (action + 1.0) / 2.0
        thrusts = motor_cmd * self.MAX_THRUST_PER_MOTOR

        # 250Hzの物理精度を維持したまま、0.02秒(=5サブステップ)ごとに1回行動を反映
        for _ in range(self.SUBSTEPS_PER_ACTION):
            self._apply_rotor_dynamics(thrusts)
            self._update_wind(self.PHYSICS_DT)
            p.applyExternalForce(self.drone, -1, self.wind_vec, [0, 0, 0], p.WORLD_FRAME)
            p.stepSimulation()

        obs = self._get_obs()
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        _, ang_vel = p.getBaseVelocity(self.drone)

        dist = np.linalg.norm(obs[0:3])
        orientation_penalty = 1.0 - abs(orn[3])          # quaternion w成分(傾くほど小さくなる)
        ang_vel_penalty = np.linalg.norm(ang_vel) * 0.05
        action_smooth_penalty = np.linalg.norm(action - self.prev_action) * 0.02
        self.prev_action = action

        reward = (
            1.0
            - (dist / 2.0)
            - orientation_penalty * 0.5
            - ang_vel_penalty
            - action_smooth_penalty
        )

        self._elapsed_steps += 1
        terminated = bool(dist > 4.0 or pos[1] < 0.2)
        truncated = self._elapsed_steps >= self.MAX_EPISODE_STEPS

        if terminated:
            reward = -2.0

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self._client is not None:
            p.disconnect()
            self._client = None