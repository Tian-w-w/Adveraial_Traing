"""
防偷拍检测系统 - Web API 服务端
核心逻辑：检测画面中是否存在手机，有手机即判定为偷拍嫌疑
"""

import os
import base64
import json
import io
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from PIL import Image

app = Flask(__name__)
CORS(app)

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-882d2641bcea422c99dd818988f2f6c0")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"

client = OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)

# ── Prompt ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
你是一个专业的手机识别视觉分析系统。
你的唯一任务是：判断画面中是否存在手机（smartphone）。
判定标准（满足以下任意两项即可确认是手机）：
1. 外形轮廓：长方形薄片状物体
2. 屏幕或机背特征：玻璃/金属质感的平整表面，有镜面反光
3. 摄像头模组：物体表面存在圆形镜头开孔、摄像头凸起或闪光灯（即使只看到机背也算）
4. 手持姿态：有手部明确握持该物体
重要原则：
- 只要能看到手机的任意一面（正面屏幕或背面摄像头），均判定为手机
- 看到摄像头模组是最强证据，单独即可判定为手机
- 宁可误报，不可漏报
以下物体即使形状相似也必须排除：
- 记事本、书本（无镜面反光，无摄像头）
- 钱包（皮革纹理，厚度不均）
- 充电宝（无摄像头模组，无屏幕）
- 遥控器（按键突出，无完整玻璃面板）
只返回 JSON，不返回任何其他文字。
"""


DETECTION_PROMPT = """
请仔细分析这张图片，判断画面中是否存在手机。

严格按照以下 JSON 格式返回，不要输出任何 JSON 以外的内容：
{
  "is_phone_detected": true或false,
  "confidence": 0到100的整数,
  "risk_level": "HIGH"或"MEDIUM"或"LOW"或"NONE",
  "phone_location": "手机在画面中的位置描述，如果没有手机则填null",
  "key_evidence": "判断依据，例如：看到清晰的摄像头模组和玻璃屏幕反光",
  "exclusion_reason": "如果is_phone_detected为false，说明排除原因；否则填null"
}

判断规则：
- 看到摄像头模组（圆形镜头、镜头开孔）→ 直接判定 true，这是最强证据
- 只有在完全确认是非手机物体时，才判定 false
- 如果看不清时，根据手部姿势进行判断。
- is_phone_detected = true 时，risk_level 直接认定为 HIGH 
"""
# - is_phone_detected = true 时，risk_level 根据摄像头朝向判断：
#     朝向他人 → HIGH
#     朝向不明 → MEDIUM
#     朝向摄像头 → MEDIUM

# ── 图像分析核心函数 ──────────────────────────────────────────────────────────

def analyze_image(image_data: bytes) -> dict:
    """调用 qwen3-vl-plus 分析图片是否存在手机"""
    b64 = base64.b64encode(image_data).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        },
                        {
                            "type": "text",
                            "text": DETECTION_PROMPT
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.1  # 低温度保证输出稳定
        )

        raw = response.choices[0].message.content.strip()

        # 提取 JSON 块（兼容模型在 JSON 前后输出多余文字的情况）
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # 统一补充字段
            result["timestamp"] = datetime.now().isoformat()
            result["raw_response"] = raw  # 保留原始响应，方便调试
            return result

    except json.JSONDecodeError as e:
        return _error_result(f"JSON解析失败: {e}")
    except Exception as e:
        return _error_result(f"API调用异常: {e}")

    return _error_result("未能从响应中提取有效JSON")


def _error_result(reason: str) -> dict:
    """统一错误返回结构"""
    return {
        "is_phone_detected": False,
        "confidence": 0,
        "risk_level": "NONE",
        "phone_location": None,
        "key_evidence": None,
        "exclusion_reason": reason,
        "timestamp": datetime.now().isoformat()
    }


# ── 图像预处理 ────────────────────────────────────────────────────────────────

def preprocess_image(raw_bytes: bytes) -> bytes:
    """压缩图片到合理尺寸，减少API传输量"""
    pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    # 保持比例缩放，最大边不超过 1280
    pil.thumbnail((1280, 1280), Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ── HTTP 接口 ─────────────────────────────────────────────────────────────────

@app.route("/detect", methods=["POST"])
def detect():
    """
    POST /detect
    支持两种方式传图：
      1. multipart/form-data，字段名 'image'
      2. JSON body: { "image_base64": "<base64字符串>" }

    返回示例：
    {
      "is_phone_detected": true,
      "confidence": 92,
      "risk_level": "HIGH",
      "phone_location": "画面右下角，有人手持",
      "key_evidence": "可见清晰摄像头模组与玻璃屏幕反光",
      "exclusion_reason": null,
      "timestamp": "2025-06-10T12:00:00"
    }
    """
    try:
        if request.files.get("image"):
            raw_bytes = request.files["image"].read()
        elif request.json and request.json.get("image_base64"):
            raw_bytes = base64.b64decode(request.json["image_base64"])
        else:
            return jsonify({"error": "请提供 image 文件或 image_base64 字段"}), 400

        img_bytes = preprocess_image(raw_bytes)
        result = analyze_image(img_bytes)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "time": datetime.now().isoformat()
    })


# ── 启动 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("防偷拍检测 API 服务启动中...")
    print(f"  模型: {MODEL_NAME}")
    print(f"  检测接口: POST http://localhost:5000/detect")
    print(f"  健康检查: GET  http://localhost:5000/health")
    app.run(host="0.0.0.0", port=5000, debug=False)