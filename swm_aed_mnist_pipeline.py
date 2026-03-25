"""
╔══════════════════════════════════════════════════════════════════════╗
║   SWM-AED 对抗样本检测与过滤 —— MNIST 完整流程                        ║
║   基于论文: "Deep Learning Models Are Vulnerable, But Adversarial    ║
║            Examples Are Even More Vulnerable" (Li et al., 2025)      ║
╠══════════════════════════════════════════════════════════════════════╣
║  流程：                                                               ║
║  1. 加载 MNIST 测试集（干净样本）                                      ║
║  2. 加载 cnn_model.h5（标准CNN）                                      ║
║  3. 用 ART + FGSM 生成对抗样本                                        ║
║  4. 混合干净样本与对抗样本，打乱                                        ║
║  5. SWM-AED：计算每张图像的 SMCE 值                                    ║
║  6. 依据阈值过滤对抗样本，保留干净子集                                  ║
║  7. 加载 cnn_robust.h5（PGD对抗训练模型）                              ║
║  8. 将过滤后数据集输入 cnn_robust.h5，输出最终准确率                    ║
╚══════════════════════════════════════════════════════════════════════╝

运行前请确认：
    pip install adversarial-robustness-toolbox tensorflow numpy
    cnn_model.h5   和  cnn_robust.h5  与本脚本同目录
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ART（Adversarial Robustness Toolbox）
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import FastGradientMethod

# ═══════════════════════════════════════════════════════
# ★  配置区（按需修改）★
# ═══════════════════════════════════════════════════════
CNN_MODEL_PATH    = "cnn_model.h5"    # 标准 CNN（用于 FGSM 白盒攻击 + SMCE 熵计算器）
ROBUST_MODEL_PATH = "cnn_robust.h5"  # PGD 对抗训练模型（最终分类评估）

NUM_CLEAN_SAMPLES = 500    # 从测试集取干净样本数量（建议 200–1000）
NUM_ADV_SAMPLES   = 500    # 生成对抗样本数量（与干净样本等量更公平）
FGSM_EPS          = 0.3    # FGSM 扰动强度（MNIST 常用 0.2–0.3，[0,1] 像素范围）
MASK_SIZE         = 4      # 滑动窗口大小（MNIST 28×28，推荐 4 或 7）
SMCE_THRESHOLD    = 0.1    # SWM-AED 检测阈值（论文推荐默认值 0.1）
BATCH_SIZE        = 64     # 推理批次大小
RANDOM_SEED       = 42
# ═══════════════════════════════════════════════════════

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ──────────────────────────────────────────────────────
# 工具：分隔线打印
# ──────────────────────────────────────────────────────
def section(title):
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


# ──────────────────────────────────────────────────────
# Step 1 · 加载 MNIST 测试集
# ──────────────────────────────────────────────────────
def load_mnist_test(n_samples=None):
    """
    加载 MNIST 测试集，归一化到 [0,1]，保持 (H,W,1) 形状。
    """
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = x_test[..., np.newaxis]   # (N, 28, 28, 1)
    y_test = y_test.astype(np.int32)

    if n_samples is not None and n_samples < len(x_test):
        idx = np.random.choice(len(x_test), n_samples, replace=False)
        x_test, y_test = x_test[idx], y_test[idx]

    return x_test, y_test


# ──────────────────────────────────────────────────────
# Step 2 · 加载 Keras 模型
# ──────────────────────────────────────────────────────
def load_keras_model(path, label="模型"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n[错误] 找不到 {label}: {path}\n"
            "请将对应 .h5 文件放在与本脚本相同的目录后重新运行。"
        )
    print(f"  加载 {label}: {path}")
    model = keras.models.load_model(path)
    return model


# ──────────────────────────────────────────────────────
# Step 3 · 用 ART + FGSM 生成对抗样本
# ──────────────────────────────────────────────────────
def build_art_classifier(keras_model, input_shape, nb_classes=10):
    """
    将 Keras 模型包装为 ART TensorFlowV2Classifier。
    使用 SparseCategoricalCrossentropy 与原始整数标签匹配。
    """
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    art_clf = TensorFlowV2Classifier(
        model=keras_model,
        loss_object=loss_fn,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
    )
    return art_clf


def generate_fgsm_adversarial(art_clf, x_clean, y_clean, eps=0.3, n_samples=500):
    """
    使用 ART FastGradientMethod 生成 FGSM 对抗样本。
    返回对抗样本数组和对应真实标签。
    """
    # 随机采样子集用于攻击
    idx = np.random.choice(len(x_clean), min(n_samples, len(x_clean)), replace=False)
    x_sub = x_clean[idx]
    y_sub = y_clean[idx]

    fgsm = FastGradientMethod(
        estimator=art_clf,
        eps=eps,
        eps_step=eps,       # 单步攻击
        targeted=False,
        num_random_init=0,
        batch_size=BATCH_SIZE,
    )
    print(f"  生成 FGSM 对抗样本（ε={eps}，共 {len(x_sub)} 张）...", end=" ", flush=True)
    x_adv = fgsm.generate(x=x_sub, y=y_sub)
    print("完成 ✓")
    return x_adv.astype(np.float32), y_sub


# ──────────────────────────────────────────────────────
# Step 4 · 混合并打乱
# ──────────────────────────────────────────────────────
def mix_and_shuffle(x_clean, y_clean, x_adv, y_adv):
    """
    合并干净样本与对抗样本，随机打乱，并记录 ground-truth 标记。
    """
    x_mixed     = np.concatenate([x_clean, x_adv],  axis=0)
    y_mixed     = np.concatenate([y_clean, y_adv],  axis=0)
    true_is_adv = np.array([False] * len(x_clean) + [True] * len(x_adv))

    shuf = np.random.permutation(len(x_mixed))
    return (x_mixed[shuf].astype(np.float32),
            y_mixed[shuf],
            true_is_adv[shuf])


# ──────────────────────────────────────────────────────
# Step 5 · SWM-AED：计算 SMCE
# ──────────────────────────────────────────────────────
def apply_mask(image, row, col, msize):
    """在 (row, col) 处覆盖 msize×msize 的零值遮挡块。"""
    masked = image.copy()
    h, w = image.shape[:2]
    masked[row:min(row+msize, h), col:min(col+msize, w), :] = 0.0
    return masked


def shannon_entropy(prob_vec):
    """H = -Σ p·log₂(p)，对应论文公式 (2) 内层计算。"""
    p = np.clip(prob_vec, 1e-12, 1.0)
    return float(-np.sum(p * np.log2(p)))


def compute_smce_single(model, image, msize):
    """
    论文公式 (2):
        H_SMCE(I) = (1/n) · Σᵢ [ -Σⱼ p_ij · log₂(p_ij) ]

    对图像 image 用 msize×msize 滑动窗口（步长=msize，不重叠）
    依次遮挡，批量推理，求平均熵。
    """
    h, w = image.shape[:2]
    masked_imgs = []
    for r in range(0, h, msize):
        for c in range(0, w, msize):
            masked_imgs.append(apply_mask(image, r, c, msize))

    batch  = np.array(masked_imgs, dtype=np.float32)
    probs  = model.predict(batch, verbose=0)   # (n_windows, num_classes)
    return float(np.mean([shannon_entropy(p) for p in probs]))


def compute_smce_batch(model, images, msize, verbose=True):
    """批量计算 SMCE，返回 shape=(N,) 数组。"""
    n = len(images)
    smce = np.zeros(n, dtype=np.float32)
    for i, img in enumerate(images):
        smce[i] = compute_smce_single(model, img, msize)
        if verbose and (i % 50 == 0 or i == n - 1):
            print(f"  SMCE 计算进度: {i+1:>4}/{n}  ", end='\r', flush=True)
    if verbose:
        print(f"  SMCE 计算完成: {n}/{n}  ✓              ")
    return smce


# ──────────────────────────────────────────────────────
# Step 6 · SWM-AED Algorithm 1：检测 + 过滤
# ──────────────────────────────────────────────────────
def swm_aed_filter(images, labels, smce_vals, threshold):
    """
    Algorithm 1 (论文 §3.4.4):
        if SMCE > threshold  →  adversarial (True)
        else                 →  clean       (False)
    返回：过滤后的 (images, labels) 以及检测标记数组。
    """
    is_adv = smce_vals > threshold
    keep   = ~is_adv
    return images[keep], labels[keep], is_adv


def detection_metrics(pred_is_adv, true_is_adv):
    """计算混淆矩阵及 Precision / Recall / F1 / Detection-Acc。"""
    TP = int(np.sum( pred_is_adv &  true_is_adv))
    FP = int(np.sum( pred_is_adv & ~true_is_adv))
    FN = int(np.sum(~pred_is_adv &  true_is_adv))
    TN = int(np.sum(~pred_is_adv & ~true_is_adv))
    prec = TP / (TP + FP + 1e-10)
    rec  = TP / (TP + FN + 1e-10)
    f1   = 2 * prec * rec / (prec + rec + 1e-10)
    acc  = (TP + TN) / (TP + FP + FN + TN + 1e-10)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                Precision=prec, Recall=rec, F1=f1, Det_Acc=acc)


# ──────────────────────────────────────────────────────
# 辅助：分类准确率
# ──────────────────────────────────────────────────────
def clf_accuracy(model, x, y):
    if len(x) == 0:
        return 0.0
    preds = model.predict(x, verbose=0, batch_size=BATCH_SIZE)
    return float(np.mean(np.argmax(preds, axis=1) == y))


# ══════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════
def main():
    section("SWM-AED MNIST 完整流程")
    print("  论文: Li et al. (2025), arXiv:2511.05073")

    # ── Step 1: 加载 MNIST ────────────────────────────
    section("Step 1 · 加载 MNIST 测试集")
    x_test, y_test = load_mnist_test(n_samples=None)
    print(f"  原始测试集: {x_test.shape}，类别数: 10")

    # 采样干净子集
    c_idx  = np.random.choice(len(x_test), NUM_CLEAN_SAMPLES, replace=False)
    x_clean, y_clean = x_test[c_idx], y_test[c_idx]
    print(f"  采样干净样本: {len(x_clean)} 张")

    # ── Step 2: 加载标准 CNN（攻击模型 + 熵计算器）──────
    section("Step 2 · 加载标准 CNN（cnn_model.h5）")
    cnn_model = load_keras_model(CNN_MODEL_PATH, "标准CNN")
    cnn_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    clean_acc_cnn = clf_accuracy(cnn_model, x_clean, y_clean)
    print(f"  标准CNN在干净样本上的准确率: {clean_acc_cnn*100:.2f}%")

    # ── Step 3: FGSM 生成对抗样本（使用 ART）────────────
    section("Step 3 · ART + FGSM 生成对抗样本")
    art_clf = build_art_classifier(
        cnn_model,
        input_shape=x_clean.shape[1:],   # (28, 28, 1)
        nb_classes=10
    )
    x_adv, y_adv = generate_fgsm_adversarial(
        art_clf, x_clean, y_clean,
        eps=FGSM_EPS, n_samples=NUM_ADV_SAMPLES
    )

    # 攻击效果验证
    adv_acc_cnn = clf_accuracy(cnn_model, x_adv, y_adv)
    print(f"  标准CNN在对抗样本上的准确率（攻击后，无防御）: "
          f"{adv_acc_cnn*100:.2f}%  "
          f"（下降幅度: {(clean_acc_cnn - adv_acc_cnn)*100:.2f}%）")

    # ── Step 4: 混合 + 打乱 ───────────────────────────
    section("Step 4 · 混合干净样本与对抗样本")

    # 取与对抗样本等量的干净子集参与混合
    n_adv = len(x_adv)
    mix_c_idx = np.random.choice(len(x_clean), min(n_adv, len(x_clean)), replace=False)
    x_c_mix = x_clean[mix_c_idx]
    y_c_mix = y_clean[mix_c_idx]

    x_mixed, y_mixed, true_is_adv = mix_and_shuffle(x_c_mix, y_c_mix, x_adv, y_adv)

    print(f"  混合后总样本: {len(x_mixed)} 张")
    print(f"    ├─ 干净样本: {int(np.sum(~true_is_adv))} 张")
    print(f"    └─ 对抗样本: {int(np.sum(true_is_adv))} 张")

    # 混合集基线（未过滤）
    acc_mixed_before = clf_accuracy(cnn_model, x_mixed, y_mixed)
    print(f"  标准CNN在混合集的准确率（SWM-AED过滤前）: "
          f"{acc_mixed_before*100:.2f}%")

    # ── Step 5: 计算全部 SMCE 值 ──────────────────────
    section(f"Step 5 · SWM-AED：计算 SMCE（mask={MASK_SIZE}×{MASK_SIZE}）")
    print(f"  共 {len(x_mixed)} 张图像，每张产生"
          f" {(28//MASK_SIZE)**2} 个遮挡窗口（耗时略长，请等待）...")

    smce_all = compute_smce_batch(cnn_model, x_mixed, msize=MASK_SIZE, verbose=True)

    # SMCE 分布统计
    smce_clean = smce_all[~true_is_adv]
    smce_adv   = smce_all[ true_is_adv]
    print(f"\n  SMCE 分布对比:")
    print(f"    干净样本 → 均值={np.mean(smce_clean):.4f}  "
          f"std={np.std(smce_clean):.4f}  "
          f"max={np.max(smce_clean):.4f}")
    print(f"    对抗样本 → 均值={np.mean(smce_adv):.4f}  "
          f"std={np.std(smce_adv):.4f}  "
          f"max={np.max(smce_adv):.4f}")

    # ── Step 6: SWM-AED 过滤 ──────────────────────────
    section(f"Step 6 · SWM-AED 过滤（threshold={SMCE_THRESHOLD}）")
    x_filt, y_filt, pred_is_adv = swm_aed_filter(
        x_mixed, y_mixed, smce_all, threshold=SMCE_THRESHOLD
    )

    # 检测性能
    dm = detection_metrics(pred_is_adv, true_is_adv)
    n_removed = len(x_mixed) - len(x_filt)

    print(f"  过滤前: {len(x_mixed)} 张  →  过滤后: {len(x_filt)} 张")
    print(f"  移除了 {n_removed} 张（{n_removed/len(x_mixed)*100:.1f}%）\n")
    print(f"  ┌──────── SWM-AED 检测混淆矩阵 ─────────┐")
    print(f"  │  TP 正确检测为对抗: {dm['TP']:>5}              │")
    print(f"  │  FP 误报为对抗:     {dm['FP']:>5}              │")
    print(f"  │  FN 漏检对抗:       {dm['FN']:>5}              │")
    print(f"  │  TN 正确保留干净:   {dm['TN']:>5}              │")
    print(f"  ├────────────────────────────────────────┤")
    print(f"  │  Precision:    {dm['Precision']*100:>7.2f}%             │")
    print(f"  │  Recall:       {dm['Recall']*100:>7.2f}%             │")
    print(f"  │  F1 Score:     {dm['F1']:>8.4f}             │")
    print(f"  │  检测准确率:   {dm['Det_Acc']*100:>7.2f}%             │")
    print(f"  └────────────────────────────────────────┘")

    if len(x_filt) == 0:
        print("\n  [警告] 过滤后无剩余样本！请适当降低 SMCE_THRESHOLD 后重新运行。")
        return

    # ── Step 7: 加载 cnn_robust.h5 ────────────────────
    section("Step 7 · 加载 PGD 对抗训练模型（cnn_robust.h5）")
    robust_model = load_keras_model(ROBUST_MODEL_PATH, "PGD对抗训练CNN")
    robust_model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])

    # ── Step 8: 评估最终准确率 ─────────────────────────
    section("Step 8 · 将过滤后数据集输入 cnn_robust.h5 评估")

    # 各对比指标
    robust_acc_clean  = clf_accuracy(robust_model, x_clean,  y_clean)
    robust_acc_adv    = clf_accuracy(robust_model, x_adv,    y_adv)
    robust_acc_mixed  = clf_accuracy(robust_model, x_mixed,  y_mixed)
    robust_acc_final  = clf_accuracy(robust_model, x_filt,   y_filt)

    # ══════════════ 最终汇总报告 ══════════════════════
    section("★  最终结果汇总  ★")
    print(f"  {'指标':<40} {'准确率':>8}")
    print(f"  {'─'*50}")
    print(f"  {'[标准CNN] 干净样本基线准确率':<40} {clean_acc_cnn*100:>7.2f}%")
    print(f"  {'[标准CNN] 对抗样本（无防御）':<40} {adv_acc_cnn*100:>7.2f}%")
    print(f"  {'─'*50}")
    print(f"  {'[鲁棒模型] 干净样本准确率':<40} {robust_acc_clean*100:>7.2f}%")
    print(f"  {'[鲁棒模型] 对抗样本准确率（无过滤）':<40} {robust_acc_adv*100:>7.2f}%")
    print(f"  {'[鲁棒模型] 混合集准确率（过滤前）':<40} {robust_acc_mixed*100:>7.2f}%")
    print(f"  {'─'*50}")
    print(f"  {'★ [鲁棒模型] 混合集准确率（SWM-AED过滤后）':<40} "
          f"{robust_acc_final*100:>7.2f}%  ← 最终结果")
    print(f"  {'─'*50}")
    print(f"  {'准确率提升（过滤前→过滤后）':<40} "
          f"{(robust_acc_final - robust_acc_mixed)*100:>+7.2f}%")
    print(f"  {'SWM-AED 检测准确率':<40} {dm['Det_Acc']*100:>7.2f}%")
    print(f"  {'过滤后保留样本数':<40} {len(x_filt):>6} / {len(x_mixed)}")
    print(f"\n{'═'*65}\n")

    return {
        'clean_acc_cnn':       clean_acc_cnn,
        'adv_acc_no_defense':  adv_acc_cnn,
        'robust_acc_clean':    robust_acc_clean,
        'robust_acc_adv':      robust_acc_adv,
        'robust_acc_mixed':    robust_acc_mixed,
        'final_acc':           robust_acc_final,   # ★ 核心输出
        'detection_metrics':   dm,
        'smce_values':         smce_all,
    }


if __name__ == "__main__":
    results = main()
