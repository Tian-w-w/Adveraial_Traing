from __future__ import annotations

import re
from pathlib import Path

LOG_FILE = "scanner_20260331.log"
LOG_START_LINE = 12

# ── 解析日志 ──────────────────────────────────────────────────
def parse_log(lines: list[str]) -> list[dict]:
    """
    匹配格式：
    头行: 2026-03-31 19:11:43  INFO      [   1/100]  False1.jpg  (178.9 KB)
    结果: OK | risk=NONE conf =95% 耗时6.01s
    """
    # 头行：日期 时间 INFO [序号/总数] 文件名 (大小)
    header_pattern = re.compile(
        r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+INFO\s+'   # 时间戳 + INFO
        r'\[\s*\d+/\d+\]\s+'                                     # [1/100]
        r'([\w\-\.]+\.(?:jpg|jpeg|png|bmp|webp))\s+'             # 文件名（捕获组1）
        r'\([\d\.]+ \w+\)',                                       # (178.9 KB)
        re.IGNORECASE
    )

    # 结果行：OK | risk=HIGH/MEDIUM/NONE conf =95% 耗时6.01s
    result_pattern = re.compile(
        r'(OK|ERROR)\s*\|\s*risk\s*=\s*(HIGH|MEDIUM|NONE)\s+'
        r'conf\s*=\s*(\d+)%\s+耗时\s*([\d\.]+)s',
        re.IGNORECASE
    )

    records = []

    for i, line in enumerate(lines):
        header_match = header_pattern.match(line.strip())
        if not header_match:
            continue

        filename = header_match.group(1)

        # 向后最多查找 3 行寻找结果行
        for j in range(i + 1, min(i + 4, len(lines))):
            result_match = result_pattern.search(lines[j].strip())
            if result_match:
                records.append({
                    "filename":   filename,
                    "status":     result_match.group(1).upper(),
                    "risk":       result_match.group(2).upper(),
                    "confidence": int(result_match.group(3)),
                    "elapsed":    float(result_match.group(4)),
                })
                break

    return records


# ── 推断真实标签 ──────────────────────────────────────────────
def get_true_label(filename: str) -> bool | None:
    """
    文件名含 'true'  → 正样本（有手机）
    文件名含 'false' → 负样本（无手机）
    否则             → None，跳过
    """
    name = filename.lower()
    if "true" in name:
        return True
    if "false" in name:
        return False
    return None


# ── 计算指标 ──────────────────────────────────────────────────
def calc_metrics(records: list[dict]) -> dict:
    TP = FP = TN = FN = 0
    skipped = []
    detail_rows = []

    for r in records:
        true_label = get_true_label(r["filename"])

        if true_label is None:
            skipped.append(r["filename"])
            continue

        pred_pos = r["risk"] in ("HIGH", "MEDIUM")

        if true_label and pred_pos:
            result = "TP"; TP += 1
        elif true_label and not pred_pos:
            result = "FN"; FN += 1
        elif not true_label and pred_pos:
            result = "FP"; FP += 1
        else:
            result = "TN"; TN += 1

        detail_rows.append({**r, "true_label": true_label, "result": result})

    total     = TP + FP + TN + FN
    accuracy  = (TP + TN) / total               if total        > 0 else 0.0
    precision = TP / (TP + FP)                  if (TP + FP)    > 0 else 0.0
    recall    = TP / (TP + FN)                  if (TP + FN)    > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    return {
        "total": total, "skipped": skipped,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1,
        "detail": detail_rows,
    }


# ── 打印报告 ──────────────────────────────────────────────────
def print_report(m: dict):
    sep = "─" * 54

    print(f"\n{'═' * 54}")
    print(f"  手机检测系统 · 评估报告")
    print(f"{'═' * 54}")

    # 混淆矩阵
    print(f"\n  混淆矩阵")
    print(f"  {sep}")
    print(f"  {'':18s}  {'预测正(H/M)':>10s}  {'预测负(N)':>10s}")
    print(f"  {'真实正(有手机)':18s}  {'TP = ' + str(m['TP']):>10s}  {'FN = ' + str(m['FN']):>10s}")
    print(f"  {'真实负(无手机)':18s}  {'FP = ' + str(m['FP']):>10s}  {'TN = ' + str(m['TN']):>10s}")
    print(f"  {sep}")

    # 核心指标
    print(f"\n  核心指标  （有效样本：{m['total']} 条 / 跳过：{len(m['skipped'])} 条）")
    print(f"  {sep}")
    print(f"  准确率  Accuracy  : {m['accuracy']  * 100:6.2f}%"
          f"  ({m['TP'] + m['TN']}/{m['total']})")
    print(f"  精确率  Precision : {m['precision'] * 100:6.2f}%"
          f"  ({m['TP']}/{m['TP'] + m['FP']})")
    print(f"  召回率  Recall    : {m['recall']    * 100:6.2f}%"
          f"  ({m['TP']}/{m['TP'] + m['FN']})")
    print(f"  F1 分数 F1-Score  : {m['f1']        * 100:6.2f}%")
    print(f"  {sep}")

    # 风险等级分布
    risks = {"HIGH": 0, "MEDIUM": 0, "NONE": 0}
    for r in m["detail"]:
        risks[r["risk"]] += 1
    print(f"\n  预测分布")
    print(f"  {sep}")
    print(f"  HIGH   : {risks['HIGH']:>4d} 条")
    print(f"  MEDIUM : {risks['MEDIUM']:>4d} 条")
    print(f"  NONE   : {risks['NONE']:>4d} 条")
    print(f"  {sep}")

    # 跳过样本
    if m["skipped"]:
        print(f"\n  跳过样本（无法识别标签，共 {len(m['skipped'])} 条）")
        for name in m["skipped"]:
            print(f"    · {name}")

    # 错误明细
    errors = [r for r in m["detail"] if r["result"] in ("FP", "FN")]
    if errors:
        print(f"\n  错误样本明细（共 {len(errors)} 条）")
        print(f"  {sep}")
        print(f"  {'类型':<6}  {'文件名':<28}  {'预测':^6}  {'置信度':>6}")
        print(f"  {sep}")
        for r in errors:
            tag = "漏检FN" if r["result"] == "FN" else "误报FP"
            print(f"  {tag:<6}  {r['filename']:<28}  {r['risk']:^6}  {r['confidence']:>5d}%")
    else:
        print(f"\n  全部分类正确，无错误样本。")

    print(f"\n{'═' * 54}\n")


# ── 主流程 ────────────────────────────────────────────────────
def main():
    log_path = Path(LOG_FILE)
    if not log_path.exists():
        print(f"[错误] 找不到日志文件：{LOG_FILE}")
        return

    all_lines = log_path.read_text(encoding="utf-8").splitlines()

    # 从第 LOG_START_LINE 行开始（1-based → 转为 0-based 索引）
    lines = all_lines[LOG_START_LINE - 1:]

    records = parse_log(lines)
    if not records:
        print("[错误] 未解析到任何记录，请确认日志格式是否匹配。")
        print("  期望格式（头行）：2026-03-31 19:11:43  INFO  [1/100]  False1.jpg  (178.9 KB)")
        print("  期望格式（结果）：OK | risk=NONE conf =95% 耗时6.01s")
        return

    print(f"[信息] 解析完成，共 {len(records)} 条记录")
    metrics = calc_metrics(records)
    print_report(metrics)


if __name__ == "__main__":
    main()