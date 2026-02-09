import os
import uuid
import logging
import requests
import asyncio
import tempfile
import base64
from typing import Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import dashscope
from dashscope import Generation

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

if not TELEGRAM_BOT_TOKEN or not DASHSCOPE_API_KEY:
    raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN or DASHSCOPE_API_KEY in environment variables.")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise EnvironmentError("Missing YANDEX_API_KEY or YANDEX_FOLDER_ID in environment variables.")

MODEL_LLM = 'qwen-plus'

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

TEMP_DIR = tempfile.mkdtemp()

# ------------------ LANGUAGE DETECTION ------------------
def is_chinese(text: str) -> bool:
    """Simple check: if text contains Chinese characters, treat as Chinese"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

# ------------------ PROMPT REWRITING ------------------
async def rewrite_prompt_with_qwen(user_phrase: str) -> str:
    """For Chinese input: enhance with Chengdu context"""
    system_prompt = (
        "你是一个专为图像生成模型设计的提示词工程师。"
        "请将用户给出的中文短语（可能抽象）转化为一个具体、生动、视觉化的场景描述。"
        "要求："
        "1. 场景必须设定在中国成都；"
        "2. 主角是年轻的中国人（避免敏感或浪漫化描述）；"
        "3. 包含具体地点（如茶馆、公园、图书馆、街道）、活动、表情、天气、光线、物品等细节；"
        "4. 语言简洁，用中文输出，不要解释，只输出改写后的描述。"
    )

    user_prompt = f"短语：{user_phrase}"

    try:
        response = Generation.call(
            model=MODEL_LLM,
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=200,
            temperature=0.6
        )
        if response.status_code == 200:
            rewritten = response.output['text'].strip()
            return rewritten.replace('"', '').replace('"', '').replace('"', '')
        else:
            logging.error(f"Qwen error: {response.code} - {response.message}")
            return f"成都场景中，人们正在体验'{user_phrase}'，真实生活，细节丰富"
    except Exception as e:
        logging.error(f"Qwen exception: {e}")
        return f"成都街头，年轻人正在体现'{user_phrase}'的概念，自然光线，日常环境"

# ------------------ YANDEX IMAGE GENERATION ------------------
async def generate_image_with_yandex(prompt: str, update: Update) -> Optional[str]:
    """Generate image using Yandex Art API (asynchronous workflow)"""
    try:
        # Step 1: Send generation request
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync"
        headers = {
            "Authorization": f"Api-Key {YANDEX_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "modelUri": f"art://{YANDEX_FOLDER_ID}/yandex-art/latest",
            "generationOptions": {
                "seed": 42,
                "aspectRatio": {
                    "widthRatio": 1,
                    "heightRatio": 1
                }
            },
            "messages": [
                {"text": prompt}
            ]
        }
        
        logging.info(f"Sending Yandex image request for prompt: {prompt}")
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if resp.status_code != 200:
            logging.error(f"Yandex API error: {resp.status_code} - {resp.text}")
            return None
        
        data = resp.json()
        operation_id = data["id"]
        logging.info(f"Yandex operation ID: {operation_id}")
        
        # Step 2: Poll for result
        result_url = f"https://llm.api.cloud.yandex.net:443/operations/{operation_id}"
        
        max_wait = 180  # 3 minutes max
        poll_interval = 5
        elapsed = 0
        notification_sent = False
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            try:
                result_resp = requests.get(result_url, headers=headers, timeout=30)
            except Exception as e:
                logging.error(f"Yandex status check exception: {e}")
                continue
            
            if result_resp.status_code != 200:
                logging.error(f"Yandex status check error: {result_resp.status_code}")
                continue
            
            result_data = result_resp.json()
            
            # Send waiting message after 30 seconds
            if elapsed >= 30 and not notification_sent and not result_data.get("done"):
                try:
                    await update.message.reply_text("⏳ 图像生成队列较长，请继续等待...")
                    notification_sent = True
                except:
                    pass
            
            # Check if generation is complete
            if result_data.get("done"):
                if "error" in result_data:
                    error_msg = result_data["error"].get("message", "Unknown error")
                    logging.error(f"Yandex generation failed: {error_msg}")
                    return None
                
                # Extract base64 image
                if "response" in result_data and "image" in result_data["response"]:
                    image_b64 = result_data["response"]["image"]
                    logging.info("Yandex image generation successful")
                    
                    # Save base64 to temporary file
                    img_name = f"{uuid.uuid4().hex[:8]}.png"
                    img_path = os.path.join(TEMP_DIR, img_name)
                    
                    image_bytes = base64.b64decode(image_b64)
                    with open(img_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    return img_path  # Return local path instead of URL
                else:
                    logging.error("Yandex response missing image data")
                    return None
        
        # Timeout
        logging.error(f"Yandex generation timed out after {max_wait} seconds")
        await update.message.reply_text("⏱️ 生成超时。服务器队列可能繁忙，请稍后重试。")
        return None
        
    except Exception as e:
        logging.error(f"Yandex image generation exception: {e}")
        return None

# ------------------ TELEGRAM HANDLERS ------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        '🖼️ Send me text in Chinese or English:\n'
        '• Chinese: I will create a scene set in Chengdu\n'
        '• English: I will generate exactly what you describe\n\n'
        '⚠️ Note: Image generation may take 1-3 minutes.'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith('/'):
        return

    # Detect language and set prompt accordingly
    if is_chinese(text):
        # Chinese mode: enhance with Chengdu context
        await update.message.reply_text(f'🧠 正在理解"{text}"...')
        enhanced_prompt = await rewrite_prompt_with_qwen(text)
        logging.info(f"Enhanced prompt (Chinese): {enhanced_prompt}")
        await update.message.reply_text('🎨 正在生成图像，请稍候...')
    else:
        # English mode: use directly as prompt
        await update.message.reply_text(f'🎨 Generating: "{text}"...')
        enhanced_prompt = text
        logging.info(f"Direct prompt (English): {enhanced_prompt}")

    # Generate image using Yandex Art API
    img_path = await generate_image_with_yandex(enhanced_prompt, update)
    if not img_path:
        await update.message.reply_text("❌ 图像生成失败，请重试。如果持续失败，可能是服务器繁忙。")
        return

    try:
        # Send the image
        with open(img_path, 'rb') as photo:
            if is_chinese(text):
                caption = f"✅ 原词: {text}\n🎨 场景: {enhanced_prompt[:200]}"
            else:
                caption = f"✅ Prompt: {text}"
            await update.message.reply_photo(photo=photo, caption=caption)

        # Clean up temp file
        os.remove(img_path)
        logging.info(f"Image sent successfully for: {text}")
    except Exception as e:
        logging.error(f"Send image exception: {e}")
        await update.message.reply_text(f"⚠️ 发送图片失败: {str(e)}")
        # Clean up on error too
        if os.path.exists(img_path):
            os.remove(img_path)

# ------------------ MAIN ------------------
def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Bot is running with Yandex Art API...")
    print("⚠️ Note: Image generation may take 1-3 minutes due to queue times")
    app.run_polling()

if __name__ == '__main__':
    main()