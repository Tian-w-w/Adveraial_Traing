from __future__ import annotations

import re
from pathlib import Path

LOG_FILE = "scanner_20260331.log"
LOG_START_LINE = 12  # 从第12行开始解析

# ── 解析单条日志记录 ──────────────────────────────────────────
def parse_log(log_text: str) -> list[dict]:
    """
    解析日志文本，返回每条记录的结构化数据。
    支持格式：
      [1/100] False1.jpg (178.9KB)
      OK | risk=NONE conf=95% 耗时6.01s
    """
    # 匹配文件名行
    header_pattern = re.compile(
        r'\[(\d+)/(\d+)\]\s+([\w\-\.]+\.(jpg|jpeg|png|bmp|webp))\s*\([\d\.]+\w+\)',
        re.IGNORECASE
    )
    # 匹配结果行（兼容 conf = 95% 或 conf=95% 两种写法）
    result_pattern = re.compile(
        r'(OK|ERROR)\s*\|\s*risk\s*=\s*(HIGH|MEDIUM|NONE)\s+conf\s*=\s*(\d+)%\s+耗时([\d\.]+)s',
        re.IGNORECASE
    )

    records = []
    lines = log_text.splitlines()
    i = 0

    while i < len(lines):
        header_match = header_pattern.search(lines[i])
        if header_match:
            filename = header_match.group(3)
            # 向后找结果行（允许中间有空行）
            for j in range(i + 1, min(i + 4, len(lines))):
                result_match = result_pattern.search(lines[j])
                if result_match:
                    records.append({
                        "filename":   filename,
                        "status":     result_match.group(1).upper(),
                        "risk":       result_match.group(2).upper(),
                        "confidence": int(result_match.group(3)),
                        "elapsed":    float(result_match.group(4)),
                    })
                    i = j  # 跳过已处理的结果行
                    break
        i += 1

    return records


# ── 推断真实标签 ──────────────────────────────────────────────
def get_true_label(filename: str) -> bool | None:
    """
    根据文件名判断真实标签：
    - 包含 'true'（不区分大小写） → True（有手机）
    - 包含 'false'（不区分大小写）→ False（无手机）
    - 无法判断 → None（跳过该样本）
    """
    name_lower = filename.lower()
    if "true" in name_lower:
        return True
    if "false" in name_lower:
        return False
    return None  # 无法判断，跳过


# ── 计算指标 ──────────────────────────────────────────────────
def calc_metrics(records: list[dict]) -> dict:
    """
    计算准确率、精确率、召回率、F1。
    预测正类：risk 为 HIGH 或 MEDIUM
    预测负类：risk 为 NONE
    """
    TP = FP = TN = FN = 0
    skipped = []
    detail_rows = []

    for r in records:
        true_label = get_true_label(r["filename"])

        if true_label is None:
            skipped.append(r["filename"])
            continue

        pred_positive = r["risk"] in ("HIGH", "MEDIUM")

        if true_label and pred_positive:
            result = "TP"; TP += 1
        elif true_label and not pred_positive:
            result = "FN"; FN += 1
        elif not true_label and pred_positive:
            result = "FP"; FP += 1
        else:
            result = "TN"; TN += 1

        detail_rows.append({**r, "true_label": true_label, "result": result})

    total     = TP + FP + TN + FN
    accuracy  = (TP + TN) / total          if total  > 0 else 0.0
    precision = TP / (TP + FP)             if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN)             if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    return {
        "total": total, "skipped": skipped,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy":  accuracy,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "detail":    detail_rows,
    }


# ── 打印报告 ──────────────────────────────────────────────────
def print_report(m: dict):
    sep = "─" * 52

    print(f"\n{'═' * 52}")
    print(f"  手机检测系统 · 评估报告")
    print(f"{'═' * 52}")

    # 混淆矩阵
    print(f"\n  混淆矩阵")
    print(f"  {sep}")
    print(f"  {'':16s}  {'预测: 正(H/M)':>12s}  {'预测: 负(N)':>10s}")
    print(f"  {'真实: 正(有手机)':16s}  {'TP = ' + str(m['TP']):>12s}  {'FN = ' + str(m['FN']):>10s}")
    print(f"  {'真实: 负(无手机)':16s}  {'FP = ' + str(m['FP']):>12s}  {'TN = ' + str(m['TN']):>10s}")
    print(f"  {sep}")

    # 核心指标
    print(f"\n  核心指标（共 {m['total']} 条有效样本）")
    print(f"  {sep}")
    print(f"  准确率  Accuracy  : {m['accuracy']  * 100:.2f}%  ({m['TP'] + m['TN']}/{m['total']})")
    print(f"  精确率  Precision : {m['precision'] * 100:.2f}%  ({m['TP']}/{m['TP'] + m['FP']})")
    print(f"  召回率  Recall    : {m['recall']    * 100:.2f}%  ({m['TP']}/{m['TP'] + m['FN']})")
    print(f"  F1 分数 F1-Score  : {m['f1']        * 100:.2f}%")
    print(f"  {sep}")

    # 跳过的样本
    if m["skipped"]:
        print(f"\n  ⚠ 跳过 {len(m['skipped'])} 个无法识别标签的文件：")
        for name in m["skipped"]:
            print(f"    · {name}")

    # 错误样本明细
    errors = [r for r in m["detail"] if r["result"] in ("FP", "FN")]
    if errors:
        print(f"\n  错误样本明细（共 {len(errors)} 条）")
        print(f"  {sep}")
        print(f"  {'类型':4s}  {'文件名':30s}  {'预测':6s}  {'置信度':>6s}")
        print(f"  {sep}")
        for r in errors:
            tag = "漏检FN" if r["result"] == "FN" else "误报FP"
            print(f"  {tag}  {r['filename']:30s}  {r['risk']:6s}  {r['confidence']:>5d}%")
    else:
        print(f"\n  全部分类正确，无错误样本。")

    print(f"\n{'═' * 52}\n")


# ── 主流程 ────────────────────────────────────────────────────
def main():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"[错误] 找不到日志文件：{LOG_FILE}")
        return

    raw = log_path.read_text(encoding="utf-8")

    # 从第 LOG_START_LINE 行开始解析
    lines = raw.splitlines()
    content = "\n".join(lines[LOG_START_LINE - 1:])

    records = parse_log(content)
    if not records:
        print("[错误] 未解析到任何有效记录，请检查日志格式。")
        return

    print(f"[信息] 共解析到 {len(records)} 条记录")

    metrics = calc_metrics(records)
    print_report(metrics)


if __name__ == "__main__":
    main()