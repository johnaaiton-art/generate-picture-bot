import os
import uuid
import logging
import requests
import asyncio
import tempfile
from typing import Optional
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import dashscope
from dashscope import Generation, ImageSynthesis

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')

if not TELEGRAM_BOT_TOKEN or not DASHSCOPE_API_KEY:
    raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN or DASHSCOPE_API_KEY in environment variables.")

MODEL_IMAGE = 'wan2.2-t2i-flash'
SIZE = '1024*1024'
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

# ------------------ IMAGE GENERATION ------------------
async def generate_image_from_prompt(prompt: str, update: Update) -> Optional[str]:
    try:
        resp = ImageSynthesis.async_call(
            model=MODEL_IMAGE,
            prompt=prompt,
            size=SIZE,
            n=1
        )
        
        if resp.status_code != 200:
            logging.error(f"Image API error: {resp.code} - {resp.message}")
            return None
        
        task_id = resp.output['task_id']
        logging.info(f"Task created: {task_id}")
        
        max_wait = 180
        poll_interval = 4
        elapsed = 0
        last_status = None
        notification_sent = False
        
        while elapsed < max_wait:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            
            try:
                status_resp = ImageSynthesis.fetch(task_id)
            except Exception as e:
                logging.error(f"Status check exception: {e}")
                continue
            
            if status_resp.status_code != 200:
                logging.error(f"Status check error: {status_resp.code} - {status_resp.message}")
                continue
            
            task_status = status_resp.output.get('task_status', 'UNKNOWN')
            
            if task_status != last_status:
                logging.info(f"Task {task_id} status: {task_status}")
                last_status = task_status
            
            if elapsed >= 30 and not notification_sent and task_status == 'PENDING':
                try:
                    await update.message.reply_text("⏳ 图像生成队列较长，请继续等待...")
                    notification_sent = True
                except:
                    pass
            
            if task_status == 'SUCCEEDED':
                return status_resp.output['results'][0]['url']
            elif task_status == 'FAILED':
                error_msg = status_resp.output.get('message', 'Unknown error')
                logging.error(f"Task failed: {error_msg}")
                return None
        
        logging.error(f"Task timed out after {max_wait} seconds (status: {last_status})")
        await update.message.reply_text("⏱️ 生成超时。服务器队列可能繁忙，请稍后重试。")
        return None
        
    except Exception as e:
        logging.error(f"Image generation exception: {e}")
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

    img_url = await generate_image_from_prompt(enhanced_prompt, update)
    if not img_url:
        await update.message.reply_text("❌ 图像生成失败，请重试。如果持续失败，可能是服务器繁忙。")
        return

    try:
        img_name = f"{uuid.uuid4().hex[:8]}.png"
        img_path = os.path.join(TEMP_DIR, img_name)
        with requests.get(img_url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(img_path, 'wb') as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)

        with open(img_path, 'rb') as photo:
            if is_chinese(text):
                caption = f"✅ 原词: {text}\n🎨 场景: {enhanced_prompt[:200]}"
            else:
                caption = f"✅ Prompt: {text}"
            await update.message.reply_photo(photo=photo, caption=caption)

        os.remove(img_path)
        logging.info(f"Image sent successfully for: {text}")
    except Exception as e:
        logging.error(f"Send image exception: {e}")
        await update.message.reply_text(f"⚠️ 发送图片失败: {str(e)}")

# ------------------ MAIN ------------------
def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("✅ Bot is running...")
    print("⚠️ Note: Image generation may take 1-3 minutes due to queue times")
    app.run_polling()

if __name__ == '__main__':
    main()
