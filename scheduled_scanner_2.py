"""
防偷拍定时文件夹扫描器
────────────────────────────────────────────────
功能：
  • 仅在指定时间窗口内运行（默认 20:00 → 次日 06:00）
  • 递归扫描图片文件夹，逐张调用 qwen-vl-plus 检测
  • 风险等级 > MEDIUM（即 HIGH）的图片复制到 alert 文件夹
  • 每张结果写入结构化日志（同时输出控制台）
  • 扫描结束后打印总耗时 + 进程资源占用摘要

依赖安装：
  pip install openai opencv-python-headless pillow numpy psutil

用法：
  # 守护模式（20:00-06:00 自动运行）
  python scheduled_scanner.py --input ./images

  # 自定义时间窗口
  python scheduled_scanner.py --input ./images --start 22:00 --end 07:00

  # 立即扫描一次（忽略时间窗口，用于调试）
  python scheduled_scanner.py --input ./images --run-once
"""

from __future__ import annotations

import os
import re
import io
import json
import time
import shutil
import signal
import base64
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime, time as dtime
from logging.handlers import RotatingFileHandler

import cv2
import numpy as np
import psutil
from PIL import Image
from openai import OpenAI


# ══════════════════════════════════════════════════════════════
#  ①  全局配置
# ══════════════════════════════════════════════════════════════

DASHSCOPE_API_KEY  = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY_HERE")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME         = "qwen-vl-plus"

ACTIVE_START = dtime(20, 0)   # 20:00
ACTIVE_END   = dtime(6,  0)   # 06:00（次日）

# 严格大于 MEDIUM → 仅 HIGH 进入告警
ALERT_RISK_LEVELS = {"HIGH"}

IMAGE_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
API_CALL_INTERVAL = 1.0          # 两次 API 调用最小间隔（秒）
FRAME_RESIZE      = (640, 480)   # 发送给模型的尺寸


# ══════════════════════════════════════════════════════════════
#  ②  日志配置
# ══════════════════════════════════════════════════════════════

def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"scanner_{datetime.now().strftime('%Y%m%d')}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        log_file, maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)

    logger = logging.getLogger("AntiSpyScanner")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ══════════════════════════════════════════════════════════════
#  ③  提示词
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """你是专业安防视觉分析 AI，专门识别图像中是否存在手机偷拍行为。

偷拍行为特征（需全面考虑）：
1. 手持手机，摄像头朝向他人隐私部位（裙底、胸部、更衣室方向等）
2. 将手机藏于包内、夹缝、桌下等隐蔽处进行拍摄
3. 设备以非自然角度放置（倒置、横置于地面、贴近遮挡物）
4. 屏幕显示拍照/录像界面但持机者试图隐藏或回避视线
5. 在人群中悄无声息地举机对准他人

不属于偷拍：正常自拍、拍摄风景/物体、视频通话、浏览手机。

请严格输出 JSON，不包含任何额外文字。"""

DETECTION_PROMPT = """请分析这张图片，判断是否存在手机偷拍行为。

严格以 JSON 格式返回（不要加 markdown 代码块，不要有任何前后缀文字）：
{
  "is_spying":   <true|false>,
  "confidence":  <0-100 整数，置信度>,
  "risk_level":  <"HIGH"|"MEDIUM"|"LOW"|"NONE">,
  "evidence":    [<最多3条视觉证据字符串>],
  "description": <50字以内的画面描述>,
  "suggestion":  <针对当前风险的处置建议>
}"""


# ══════════════════════════════════════════════════════════════
#  ④  工具函数
# ══════════════════════════════════════════════════════════════

def is_active_window(start: dtime = ACTIVE_START, end: dtime = ACTIVE_END) -> bool:
    """判断当前时刻是否在活跃窗口内，支持跨午夜区间（如 20:00→06:00）"""
    now = datetime.now().time()
    if start > end:          # 跨午夜
        return now >= start or now < end
    return start <= now < end


def seconds_until_active(start: dtime = ACTIVE_START) -> int:
    """距下一个活跃窗口开始还有多少秒"""
    now = datetime.now()
    target = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    if target <= now:
        from datetime import timedelta
        target += timedelta(days=1)
    return int((target - now).total_seconds())


def image_to_b64(path: Path, size: tuple = FRAME_RESIZE) -> str:
    """读取图片 → 压缩至 size → base64 JPEG"""
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"cv2 无法解码: {path}")
    resized = cv2.resize(frame, size)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil     = Image.fromarray(rgb)
    buf     = io.BytesIO()
    pil.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    m = re.search(r"\{.*\}", clean, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fmt_seconds(s: float) -> str:
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


# ══════════════════════════════════════════════════════════════
#  ⑤  核心扫描器
# ══════════════════════════════════════════════════════════════

class ScheduledScanner:

    def __init__(
        self,
        input_dir:  Path,
        alert_dir:  Path,
        log_dir:    Path,
        api_key:    str   = DASHSCOPE_API_KEY,
        start_time: dtime = ACTIVE_START,
        end_time:   dtime = ACTIVE_END,
    ):
        self.input_dir  = input_dir
        self.alert_dir  = alert_dir
        self.start_time = start_time
        self.end_time   = end_time

        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.log    = setup_logger(log_dir)
        self.client = OpenAI(api_key=api_key, base_url=DASHSCOPE_BASE_URL)
        self._stop  = False

        signal.signal(signal.SIGINT,  self._sig_handler)
        signal.signal(signal.SIGTERM, self._sig_handler)

        self.log.info("=" * 64)
        self.log.info("防偷拍定时扫描器  启动")
        self.log.info(f"  输入目录 : {self.input_dir.resolve()}")
        self.log.info(f"  告警目录 : {self.alert_dir.resolve()}")
        self.log.info(f"  日志目录 : {log_dir.resolve()}")
        self.log.info(
            f"  活跃窗口 : {start_time.strftime('%H:%M')} → {end_time.strftime('%H:%M')}"
        )
        self.log.info(f"  模型     : {MODEL_NAME}")
        self.log.info("=" * 64)

    def _sig_handler(self, sig, frame):
        self.log.warning("收到退出信号，将在当前文件处理完毕后安全停止...")
        self._stop = True

    # ── API 调用（无限重试，直到成功） ───────────────────────

    # 重试等待策略（秒）：先快后慢，最长不超过 RETRY_WAIT_CAP
    RETRY_BASE_WAIT = 5      # 首次失败等待秒数
    RETRY_BACKOFF   = 2.0    # 指数退避倍率
    RETRY_WAIT_CAP  = 300    # 最长单次等待上限（5 分钟）

    def _call_api_with_retry(self, b64: str, img_name: str) -> dict:
        """
        无限重试直到 API 返回合法结果。
        针对不同错误类型采用不同等待策略：
          • 频率限制 (429)      → 等待较长时间（60 s 起步）
          • 认证失败 (401/403)  → 等待并警告，不会自动修复，需人工干预
          • 网络 / 超时         → 指数退避
          • 其他服务端错误      → 指数退避
        按 Ctrl-C（SIGINT）或 SIGTERM 设置 self._stop=True 后会中止等待并抛出。
        """
        attempt   = 0
        wait_secs = self.RETRY_BASE_WAIT

        while not self._stop:
            attempt += 1
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{b64}"
                                    },
                                },
                                {"type": "text", "text": DETECTION_PROMPT},
                            ],
                        },
                    ],
                    max_tokens=512,
                    temperature=0.05,
                )
                # ── 调用成功 ──────────────────────────────────
                result = parse_json(resp.choices[0].message.content.strip())
                if attempt > 1:
                    self.log.info(
                        f"    ✓ 第 {attempt} 次调用成功（{img_name}）"
                    )
                return result

            except Exception as exc:
                exc_str  = str(exc)
                exc_type = type(exc).__name__

                # ── 分类识别错误 ──────────────────────────────
                status_code = getattr(exc, "status_code", None)

                if status_code == 429 or "rate limit" in exc_str.lower() or "429" in exc_str:
                    # 频率超限：等待时间从 60 s 起步
                    wait_secs = max(60, wait_secs)
                    level     = "WARNING"
                    reason    = f"API 频率限制 (429)"

                elif status_code in (401, 403) or "auth" in exc_str.lower() or "401" in exc_str or "403" in exc_str:
                    # 认证失败：同样无限等待，但给出明确人工干预提示
                    wait_secs = 60
                    level     = "ERROR"
                    reason    = f"认证/权限失败 ({status_code}) — 请检查 API Key"

                elif any(k in exc_str.lower() for k in ("timeout", "timed out", "connect")):
                    # 网络超时/连接失败：指数退避
                    level  = "WARNING"
                    reason = "网络超时或连接失败"

                elif status_code and status_code >= 500:
                    # 服务端错误：指数退避
                    level  = "WARNING"
                    reason = f"服务端错误 ({status_code})"

                else:
                    level  = "ERROR"
                    reason = f"{exc_type}: {exc_str[:120]}"

                # ── 记录日志 ──────────────────────────────────
                log_fn = getattr(self.log, level.lower())
                log_fn(
                    f"    ✗ [{img_name}] 第 {attempt} 次调用失败 — {reason}  "
                    f"→ {wait_secs}s 后重试..."
                )

                # ── 可中断式等待 ──────────────────────────────
                deadline = time.monotonic() + wait_secs
                while time.monotonic() < deadline:
                    if self._stop:
                        raise RuntimeError("检测到退出信号，终止重试")
                    time.sleep(min(1.0, deadline - time.monotonic()))

                # ── 计算下一轮等待（指数退避，上限封顶）────────
                wait_secs = min(
                    int(wait_secs * self.RETRY_BACKOFF),
                    self.RETRY_WAIT_CAP
                )

        # 走到这里说明 _stop=True
        raise RuntimeError("扫描器收到停止信号，终止 API 调用")

    def _detect_one(self, path: Path) -> dict:
        b64    = image_to_b64(path)
        result = self._call_api_with_retry(b64, path.name)

        result.setdefault("is_spying",   False)
        result.setdefault("confidence",  0)
        result.setdefault("risk_level",  "NONE")
        result.setdefault("evidence",    [])
        result.setdefault("description", "")
        result.setdefault("suggestion",  "")
        result["source_file"] = str(path)
        result["timestamp"]   = datetime.now().isoformat(timespec="seconds")
        return result

    # ── 复制告警文件 ──────────────────────────────────────────

    def _copy_to_alert(self, src: Path, result: dict) -> Path:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        risk = result["risk_level"]
        conf = result["confidence"]
        # 文件名：风险等级_置信度_时间戳_原文件名
        dst  = self.alert_dir / f"{risk}_{conf}pct_{ts}_{src.name}"
        shutil.copy2(src, dst)
        # 同名 JSON 报告
        with open(dst.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return dst

    # ── 单次完整扫描 ──────────────────────────────────────────

    def _run_scan(self):
        proc       = psutil.Process(os.getpid())
        t_start    = time.perf_counter()
        mem_before = proc.memory_info().rss

        images = collect_images(self.input_dir)
        total  = len(images)

        if total == 0:
            self.log.warning(f"在 {self.input_dir} 未找到任何图片，跳过本次扫描")
            return

        self.log.info(f"▶ 扫描开始 | 共发现 {total} 张图片")
        self.log.info("─" * 64)

        stats = {
            "total": total, "processed": 0,
            "alert": 0, "safe": 0, "error": 0,
            "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NONE": 0,
        }
        last_api_call = 0.0

        for idx, img_path in enumerate(images, 1):
            if self._stop:
                self.log.warning("扫描中途收到退出信号，提前结束")
                break

            # 限速：与上次 API 调用之间至少间隔 API_CALL_INTERVAL 秒
            gap = time.perf_counter() - last_api_call
            if gap < API_CALL_INTERVAL:
                time.sleep(API_CALL_INTERVAL - gap)

            file_size = img_path.stat().st_size
            self.log.info(
                f"[{idx:>4}/{total}]  {img_path.name}  ({fmt_bytes(file_size)})"
            )

            try:
                t0      = time.perf_counter()
                result  = self._detect_one(img_path)
                elapsed = time.perf_counter() - t0
                last_api_call = time.perf_counter()
                stats["processed"] += 1
            except Exception:
                self.log.error(f"    ✗ 处理异常:\n{traceback.format_exc()}")
                stats["error"] += 1
                continue

            risk = result["risk_level"]
            conf = result["confidence"]
            desc = result["description"] or "—"

            stats[risk] = stats.get(risk, 0) + 1

            # 可读日志行
            flag = "⚠️  ALERT" if risk in ALERT_RISK_LEVELS else "   OK   "
            self.log.info(
                f"    {flag} | risk={risk:<6}  conf={conf:>3}%"
                f"  耗时={elapsed:.2f}s  |  {desc}"
            )
            if result["evidence"]:
                self.log.debug("    证据: " + " | ".join(result["evidence"]))

            # 告警：复制文件
            if risk in ALERT_RISK_LEVELS:
                stats["alert"] += 1
                dst = self._copy_to_alert(img_path, result)
                self.log.warning(f"    → 已复制至告警目录: {dst.name}")
            else:
                stats["safe"] += 1

        # ══ 汇总 ═══════════════════════════════════════════════
        t_total     = time.perf_counter() - t_start
        mem_after   = proc.memory_info().rss
        cpu_pct     = proc.cpu_percent(interval=0.5)
        has_io      = hasattr(proc, "io_counters")
        io_info     = proc.io_counters() if has_io else None

        self.log.info("")
        self.log.info("╔" + "═" * 62 + "╗")
        self.log.info("║{:^62}║".format("  扫描完成摘要  "))
        self.log.info("╠" + "═" * 62 + "╣")
        self.log.info("║  {:<30}{:>28}  ║".format("总图片数",      str(stats["total"])))
        self.log.info("║  {:<30}{:>28}  ║".format("实际处理",      str(stats["processed"])))
        self.log.info("║  {:<30}{:>28}  ║".format("处理错误",      str(stats["error"])))
        self.log.info("╠" + "─" * 62 + "╣")
        self.log.info("║  {:<30}{:>28}  ║".format("⚠️  HIGH  (已复制到 alert)",  str(stats["alert"])))
        self.log.info("║  {:<30}{:>28}  ║".format("MEDIUM",        str(stats["MEDIUM"])))
        self.log.info("║  {:<30}{:>28}  ║".format("LOW",           str(stats["LOW"])))
        self.log.info("║  {:<30}{:>28}  ║".format("NONE (安全)",   str(stats["NONE"])))
        self.log.info("╠" + "─" * 62 + "╣")
        self.log.info("║  {:<30}{:>28}  ║".format("总检测时间",    fmt_seconds(t_total) + f"  ({t_total:.1f}s)"))
        self.log.info("║  {:<30}{:>28}  ║".format("平均每张耗时",  f"{t_total/max(stats['processed'],1):.2f} 秒"))
        self.log.info("╠" + "─" * 62 + "╣")
        self.log.info("║  {:<30}{:>28}  ║".format("内存（起始）",  fmt_bytes(mem_before)))
        self.log.info("║  {:<30}{:>28}  ║".format("内存（结束）",  fmt_bytes(mem_after)))
        self.log.info("║  {:<30}{:>28}  ║".format("内存增量",      fmt_bytes(mem_after - mem_before)))
        self.log.info("║  {:<30}{:>28}  ║".format("CPU 占用",      f"{cpu_pct:.1f}%"))
        if io_info:
            self.log.info("║  {:<30}{:>28}  ║".format("磁盘读取",  fmt_bytes(io_info.read_bytes)))
            self.log.info("║  {:<30}{:>28}  ║".format("磁盘写入",  fmt_bytes(io_info.write_bytes)))
        self.log.info("╚" + "═" * 62 + "╝")

    # ── 守护模式 ──────────────────────────────────────────────

    def run_daemon(self):
        """
        持续运行：
          进入活跃时间窗口 → 执行扫描 → 等待窗口结束 → 再次等待 → 循环
        """
        self.log.info("守护模式就绪，等待活跃时间窗口...")
        while not self._stop:
            if is_active_window(self.start_time, self.end_time):
                self.log.info("★ 进入活跃时间窗口，开始扫描任务")
                try:
                    self._run_scan()
                except Exception:
                    self.log.error("扫描异常:\n" + traceback.format_exc())

                self.log.info("本轮扫描完成，等待时间窗口结束后进入下一轮休眠...")
                # 等到窗口结束
                while is_active_window(self.start_time, self.end_time) and not self._stop:
                    time.sleep(60)
            else:
                wait = seconds_until_active(self.start_time)
                self.log.info(
                    f"当前不在活跃时段，下次启动时间: "
                    f"{self.start_time.strftime('%H:%M')}  "
                    f"（还需等待 {fmt_seconds(wait)}）"
                )
                slept = 0
                while slept < wait and not self._stop:
                    chunk = min(60, wait - slept)
                    time.sleep(chunk)
                    slept += chunk

        self.log.info("守护进程已安全停止。")

    # ── 单次运行（调试）──────────────────────────────────────

    def run_once(self):
        self.log.info("【单次运行模式】忽略时间窗口，立即执行扫描")
        self._run_scan()


# ══════════════════════════════════════════════════════════════
#  ⑥  CLI
# ══════════════════════════════════════════════════════════════

def parse_hhmm(s: str) -> dtime:
    try:
        h, m = s.split(":")
        return dtime(int(h), int(m))
    except Exception:
        raise argparse.ArgumentTypeError(
            f"时间格式应为 HH:MM，例如 20:00，实际收到: {s!r}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="防偷拍定时文件夹扫描器 (qwen-vl-plus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",    type=Path,      default=Path("./images"),
                        metavar="DIR",  help="扫描目录（默认 ./images）")
    parser.add_argument("--alert",    type=Path,      default=Path("./alerts"),
                        metavar="DIR",  help="告警输出目录（默认 ./alerts）")
    parser.add_argument("--log",      type=Path,      default=Path("./logs"),
                        metavar="DIR",  help="日志目录（默认 ./logs）")
    parser.add_argument("--start",    type=parse_hhmm, default=ACTIVE_START,
                        metavar="HH:MM", help="活跃窗口开始（默认 20:00）")
    parser.add_argument("--end",      type=parse_hhmm, default=ACTIVE_END,
                        metavar="HH:MM", help="活跃窗口结束（默认 06:00）")
    parser.add_argument("--api-key",  default=DASHSCOPE_API_KEY,
                        help="DashScope API Key（也可设环境变量 DASHSCOPE_API_KEY）")
    parser.add_argument("--run-once", action="store_true",
                        help="忽略时间窗口，立即扫描一次后退出")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[错误] 输入目录不存在: {args.input}")
        raise SystemExit(1)

    scanner = ScheduledScanner(
        input_dir  = args.input,
        alert_dir  = args.alert,
        log_dir    = args.log,
        api_key    = args.api_key,
        start_time = args.start,
        end_time   = args.end,
    )

    if args.run_once:
        scanner.run_once()
    else:
        scanner.run_daemon()


if __name__ == "__main__":
    main()
