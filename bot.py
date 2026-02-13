import os
import uuid
import logging
import requests
import asyncio
import tempfile
import base64
import re
from typing import Optional, List, Tuple
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import dashscope
from dashscope import Generation
import gspread
from google.oauth2.service_account import Credentials

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

# Google Sheets setup
GOOGLE_CREDS_FILE = os.path.join(os.path.dirname(__file__), 'google-creds.json')
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1H-ezqh5Vcl3_6YWJIy9KvgpDCsM3V6N4LJq0dqNbJS0/edit?gid=0#gid=0"
SHEET_NAME = "Chinese"

if not TELEGRAM_BOT_TOKEN or not DASHSCOPE_API_KEY:
    raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN or DASHSCOPE_API_KEY in environment variables.")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise EnvironmentError("Missing YANDEX_API_KEY or YANDEX_FOLDER_ID in environment variables.")

MODEL_LLM = 'qwen-plus'

dashscope.api_key = DASHSCOPE_API_KEY
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'

TEMP_DIR = tempfile.mkdtemp()

# Cache for collocations (like Hebrew bot's LAST_RESULTS)
COLLOCATION_CACHE = {}

# Initialize Google Sheets client
def get_google_sheets_client():
    """Initialize Google Sheets API client"""
    try:
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(GOOGLE_CREDS_FILE, scopes=scopes)
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        logging.error(f"Failed to initialize Google Sheets client: {e}")
        return None

# ------------------ LANGUAGE DETECTION ------------------
def is_chinese(text: str) -> bool:
    """Simple check: if text contains Chinese characters, treat as Chinese"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def extract_collocation_request(text: str) -> Optional[str]:
    """
    Check if text is a collocation request.
    Formats: "途径 col" or "途径 collocation"
    Returns the Chinese word if it's a collocation request, None otherwise.
    """
    text = text.strip()
    
    # Pattern: Chinese characters followed by space and "col" or "collocation"
    pattern = r'^([\u4e00-\u9fff]+)\s+(col|collocation)$'
    match = re.match(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1)  # Return the Chinese word
    return None

# ------------------ COLLOCATION GENERATION ------------------
async def generate_collocations(chinese_word: str) -> List[Tuple[str, str]]:
    """
    Generate typical collocations for a Chinese word.
    Returns list of (chinese_collocation, english_translation) tuples.
    """
    system_prompt = (
        "你是一个中文搭配词专家。请为给定的中文词提供5个最常用的搭配短语。\n"
        "CRITICAL要求：\n"
        "1. 每个搭配必须是2-4个汉字的短语（不要完整句子！）\n"
        "2. 每个搭配必须包含原词\n"
        "3. 提供简短的英文翻译（1-3个英文词）\n"
        "4. 输出格式：每行一个搭配，格式为'中文搭配|英文翻译'\n"
        "5. 只输出搭配列表，不要例句或解释\n\n"
        "正确示例：\n"
        "途径：\n"
        "有效途径|effective means\n"
        "法律途径|legal channel\n"
        "外交途径|diplomatic channel\n\n"
        "错误示例（太长）：\n"
        "加强实践教学是提升学生动手能力的有效途径 ❌\n"
        "通过途径解决问题 ❌"
    )

    user_prompt = f"请为'{chinese_word}'生成5个常用搭配短语（必须2-4个汉字）。"

    try:
        response = Generation.call(
            model=MODEL_LLM,
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=200,
            temperature=0.3
        )
        
        if response.status_code == 200:
            result_text = response.output['text'].strip()
            
            # Parse the response
            collocations = []
            lines = result_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remove numbering if present (1. 2. etc.)
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                
                # Split by | or common separators
                parts = re.split(r'[|：:\-—]', line, maxsplit=1)
                
                if len(parts) == 2:
                    chinese = parts[0].strip()
                    english = parts[1].strip()
                    
                    # Clean up quotes
                    chinese = chinese.replace('"', '').replace('"', '').replace('"', '')
                    english = english.replace('"', '').replace('"', '').replace('"', '')
                    
                    # CRITICAL: Only accept SHORT collocations (2-6 characters max)
                    if chinese and english and 2 <= len(chinese) <= 6:
                        collocations.append((chinese, english))
                    else:
                        logging.warning(f"Rejected long collocation: {chinese} ({len(chinese)} chars)")
            
            # If we got at least some collocations, return them
            if collocations:
                return collocations[:5]  # Max 5
            else:
                # Fallback with short default
                logging.warning(f"Failed to parse collocations from: {result_text}")
                return [(f"{chinese_word}用法", "usage")]
        else:
            logging.error(f"API error: {response.code} - {response.message}")
            return [(f"{chinese_word}用法", "usage")]
            
    except Exception as e:
        logging.error(f"Collocation generation exception: {e}")
        return [(f"{chinese_word}用法", "usage")]

# ------------------ GOOGLE SHEETS OPERATIONS ------------------
def save_collocation_to_sheet(chinese: str, english: str) -> bool:
    """Save a collocation to Google Sheets (like Hebrew bot)"""
    try:
        client = get_google_sheets_client()
        if not client:
            logging.error("Google Sheets client not initialized")
            return False
        
        # Open the spreadsheet and worksheet (like Hebrew bot)
        spreadsheet = client.open_by_url(SPREADSHEET_URL)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        
        # Append the row (like Hebrew bot: just append, no duplicate check)
        worksheet.append_row([chinese, english], value_input_option="USER_ENTERED")
        logging.info(f"Saved to sheet: {chinese} | {english}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save to sheet: {e}")
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
        '• English: I will generate exactly what you describe\n'
        '• Chinese word + "col": Get collocations (e.g., "途径 col")\n\n'
        '⚠️ Note: Image generation may take 1-3 minutes.'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith('/'):
        return

    # Check if this is a collocation request
    chinese_word = extract_collocation_request(text)
    
    if chinese_word:
        # Collocation mode
        await update.message.reply_text(f'📚 正在查找 "{chinese_word}" 的常用搭配...')
        
        collocations = await generate_collocations(chinese_word)
        
        if not collocations:
            await update.message.reply_text("❌ 未能生成搭配，请重试。")
            return
        
        # Store collocations in cache (like Hebrew bot)
        chat_id = update.message.chat_id
        COLLOCATION_CACHE[chat_id] = collocations
        
        # Create inline keyboard with index-based callback_data
        keyboard = []
        for idx, (chinese, english) in enumerate(collocations):
            button_text = f"{chinese} {english}"
            # Trim if too long for display
            if len(button_text) > 60:
                button_text = button_text[:57] + "..."
            # Use only index (like Hebrew bot: f"save:{i}")
            callback_data = f"save:{idx}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"✅ 找到 {len(collocations)} 个常用搭配：\n点击按钮保存到表格",
            reply_markup=reply_markup
        )
        return
    
    # Image generation mode (original functionality)
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

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks for collocation saving (like Hebrew bot)"""
    query = update.callback_query
    await query.answer()
    
    # Parse callback data
    if not query.data.startswith("save:"):
        return
    
    try:
        # Get index from callback data (like Hebrew bot)
        idx = int(query.data.split(":")[1])
        
        # Retrieve collocations from global cache (like Hebrew bot)
        chat_id = query.message.chat_id
        cached = COLLOCATION_CACHE.get(chat_id)
        
        if not cached or idx >= len(cached):
            await query.edit_message_text("❌ 数据已过期，请重新请求搭配")
            return
        
        chinese, english = cached[idx]
        
    except (ValueError, IndexError, TypeError) as e:
        logging.error(f"Button callback error: {e}")
        await query.edit_message_text("❌ 数据格式错误")
        return
    
    # Save to Google Sheets
    success = save_collocation_to_sheet(chinese, english)
    
    if success:
        await query.edit_message_text(
            f"✅ 已保存:\n中文: {chinese}\n英文: {english}\n\n已添加到表格！"
        )
    else:
        await query.edit_message_text(
            f"❌ 保存失败，请检查 Google Sheets 配置\n\n搭配: {chinese} | {english}"
        )

# ------------------ MAIN ------------------
def main():
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Check Google credentials file exists
    if not os.path.exists(GOOGLE_CREDS_FILE):
        logging.warning(f"Google credentials file not found: {GOOGLE_CREDS_FILE}")
        logging.warning("Collocation saving will not work until you add google-creds.json")
    
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_callback))
    
    print("✅ Bot is running with Yandex Art API and Collocation feature...")
    print("📚 Collocation format: '途径 col' or '途径 collocation'")
    print("⚠️ Note: Image generation may take 1-3 minutes due to queue times")
    app.run_polling()

if __name__ == '__main__':
    main()