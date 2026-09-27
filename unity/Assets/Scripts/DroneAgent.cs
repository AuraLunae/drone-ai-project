using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

/// <summary>
/// フルエンドツーエンド版ドローンエージェント
/// - Decision Requester の Decision Period = 1 を前提(毎FixedUpdateで意思決定)
/// - 風はOrnstein-Uhlenbeck過程による時間変化する乱気流(PyBullet側と同一モデル)
/// - 行動は「4基のモーターの個別推力指令」。ロール/ピッチトルクは各モーターの
///   位置オフセットと推力の外積(Vector3.Cross)で自前計算し、ヨートルクは回転方向の
///   異なるモーター間の反トルク差で計算する(Unityの AddForceAtPosition による自動トルク
///   導出には依存しない。PyBullet側 drone_pretrain_env.py の _apply_rotor_dynamics と
///   完全に対応させること)。重力補正などの補助は行わず、ホバリングに必要な出力配分は
///   AIが自力で学習する。
/// - 報酬は距離 + 姿勢安定性 + 角速度 + 行動の滑らかさ で構成(PyBullet側と一致)
/// </summary>
public class DroneAgent : Agent
{
    private Rigidbody rb;
    public Transform targetTransform; // Targetオブジェクトをドラッグ&ドロップ

    // 風(OU過程) — OnEpisodeBeginごとに再抽選
    private Vector3 windVec;
    private Vector3 windMean;
    private float windTheta;
    private float windSigma;

    // --- 4モーター クアッドコプター ダイナミクス パラメータ ---
    // PyBullet側 (ARM_LENGTH / MAX_THRUST_PER_MOTOR / YAW_TORQUE_COEFF) と必ず一致させること
    private const float ArmLength = 0.15f;
    private const float MaxThrustPerMotor = 6.0f;
    private const float YawTorqueCoeff = 0.02f;

    // モーター配置(ローカル座標) と回転方向。対角のモーター同士が同じ回転方向。
    private static readonly Vector3[] MotorOffsets = new Vector3[]
    {
        new Vector3(-ArmLength, 0f,  ArmLength), // m0: 前左 (CW)
        new Vector3( ArmLength, 0f,  ArmLength), // m1: 前右 (CCW)
        new Vector3( ArmLength, 0f, -ArmLength), // m2: 後右 (CW)
        new Vector3(-ArmLength, 0f, -ArmLength), // m3: 後左 (CCW)
    };
    private static readonly float[] MotorSpin = { -1f, 1f, -1f, 1f }; // CW=-1, CCW=+1

    // UnityのPhysX(左手系)とPyBulletのBullet(右手系)の内部実装の違いにより、
    // 同じ外積計算式(Vector3.Cross / np.cross)で求めたトルクでも、AddTorqueに
    // 渡した際の実際の回転方向がPyBullet側と反転する可能性がある(理論だけでは
    // 断定できない)。下記「Step 3.5: モーター単体テストによる符号検証」の手順で
    // 実機検証し、もし回転方向がPyBulletと逆だった場合はこの値を -1f に変更するだけで
    // 全軸の符号を一括反転できる。
    private const float TorqueSignCorrection = 1f;

    private Vector4 prevAction;

    public override void Initialize()
    {
        rb = GetComponent<Rigidbody>();
    }

    public override void OnEpisodeBegin()
    {
        Vector3 initOffset = new Vector3(
            Random.Range(-0.5f, 0.5f),
            Mathf.Abs(Random.Range(-0.5f, 0.5f)),
            Random.Range(-0.5f, 0.5f)
        );
        transform.localPosition = new Vector3(initOffset.x, 1f + initOffset.y, initOffset.z);
        transform.localRotation = Quaternion.identity;
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        Vector3 dir = Random.onUnitSphere;
        windMean = dir * Random.Range(3.0f, 8.0f);
        windTheta = Random.Range(0.5f, 2.0f);
        windSigma = Random.Range(1.0f, 4.0f);
        windVec = Vector3.zero;

        prevAction = Vector4.zero;
    }

    // PyBulletと完全に一致する13次元の観測値(行動空間の変更に伴う変更なし)
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.localPosition - targetTransform.localPosition); // (3)
        sensor.AddObservation(transform.localRotation);                                 // (4)
        sensor.AddObservation(rb.linearVelocity);                                       // (3)
        sensor.AddObservation(rb.angularVelocity);                                      // (3)
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        Vector3 bodyUpWorld = transform.rotation * Vector3.up;
        Vector3 totalForce = Vector3.zero;
        Vector3 totalTorque = Vector3.zero;
        Vector4 currentAction = Vector4.zero;

        for (int i = 0; i < 4; i++)
        {
            float a = Mathf.Clamp(actions.ContinuousActions[i], -1f, 1f);
            currentAction[i] = a;

            // [-1, 1] -> [0, 1] のモーター指令に変換(0=停止, 1=最大出力)。
            // 重力補正などの補助は一切行わない。
            float motorCmd = (a + 1f) / 2f;
            float thrust = motorCmd * MaxThrustPerMotor;

            Vector3 worldOffset = transform.rotation * MotorOffsets[i];
            Vector3 force = bodyUpWorld * thrust;
            totalForce += force;
            // ロール/ピッチ: 位置オフセットと力の外積を自前で合算(AddForceAtPositionの
            // 自動トルク導出には頼らない)
            totalTorque += Vector3.Cross(worldOffset, force);
            // ヨー: 回転方向差による反トルクを合算
            totalTorque += bodyUpWorld * (MotorSpin[i] * thrust * YawTorqueCoeff);
        }

        // 合力は重心に加える(AddForceは常に重心に作用するため、これ自体は追加トルクを生まない)
        rb.AddForce(totalForce);
        // 合トルクは自前計算した値をまとめて加える(符号補正あり。詳細はTorqueSignCorrectionの定義を参照)
        rb.AddTorque(totalTorque * TorqueSignCorrection);

        float dist = Vector3.Distance(transform.localPosition, targetTransform.localPosition);
        float orientationPenalty = 1.0f - Mathf.Abs(transform.localRotation.w);
        float angVelPenalty = rb.angularVelocity.magnitude * 0.05f;
        float actionSmoothPenalty = Vector4.Distance(currentAction, prevAction) * 0.02f;
        prevAction = currentAction;

        float reward = 1.0f - (dist / 2.0f) - orientationPenalty * 0.5f - angVelPenalty - actionSmoothPenalty;
        SetReward(reward);

        if (dist > 4.0f || transform.localPosition.y < 0.2f)
        {
            SetReward(-2.0f);
            EndEpisode();
        }
    }

    void FixedUpdate()
    {
        float dt = Time.fixedDeltaTime; // 0.02秒(50Hz)。PyBulletのCONTROL_DTと一致させること。
        Vector3 noise = new Vector3(GaussianRandom(), GaussianRandom(), GaussianRandom());
        windVec += windTheta * (windMean - windVec) * dt + windSigma * Mathf.Sqrt(dt) * noise;
        rb.AddForce(windVec);
    }

    // UnityにはGaussian乱数が無いのでBox-Muller法で生成
    private float GaussianRandom()
    {
        float u1 = 1.0f - Random.value;
        float u2 = 1.0f - Random.value;
        return Mathf.Sqrt(-2.0f * Mathf.Log(u1)) * Mathf.Sin(2.0f * Mathf.PI * u2);
    }
}