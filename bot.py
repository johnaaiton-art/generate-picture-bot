import os
import uuid
import logging
import requests
import asyncio
import tempfile
import base64
import re
from typing import Optional, List, Tuple, Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

# ------------------ CONFIG ------------------
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

# Google Sheets setup
GOOGLE_CREDS_FILE = os.path.join(os.path.dirname(__file__), 'google-creds.json')
SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1H-ezqh5Vcl3_6YWJIy9KvgpDCsM3V6N4LJq0dqNbJS0/edit?gid=0#gid=0"
SHEET_NAME = "Chinese"

if not TELEGRAM_BOT_TOKEN or not DEEPSEEK_API_KEY:
    raise EnvironmentError("Missing TELEGRAM_BOT_TOKEN or DEEPSEEK_API_KEY in environment variables.")

if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise EnvironmentError("Missing YANDEX_API_KEY or YANDEX_FOLDER_ID in environment variables.")

TEMP_DIR = tempfile.mkdtemp()

# Cache for collocations
COLLOCATION_CACHE: Dict[int, List[Tuple[str, str]]] = {}

# Initialize DeepSeek client
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

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

# ------------------ DEFINITION GENERATION ------------------
async def generate_definition(chinese_word: str) -> str:
    """Generate definition and examples for a Chinese word using Yandex API"""
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "你是一个中文词汇专家。用户会给你一个中文词语，"
        "请提供简洁的定义和2-3个例句。"
        "格式："
        "定义：[简短定义]\n"
        "例句：\n"
        "1. [例句1]\n"
        "2. [例句2]"
    )
    
    data = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.3,
            "maxTokens": 500
        },
        "messages": [
            {
                "role": "system",
                "text": system_prompt
            },
            {
                "role": "user",
                "text": chinese_word
            }
        ]
    }
    
    try:
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if 'result' in result and 'alternatives' in result['result']:
            definition = result['result']['alternatives'][0]['message']['text'].strip()
            return definition
        return f"定义：{chinese_word} 的词义"
    except Exception as e:
        logging.error(f"Yandex definition error: {e}")
        return f"定义：{chinese_word}"

# ------------------ COLLOCATION GENERATION ------------------
async def generate_collocations(chinese_word: str) -> List[Tuple[str, str]]:
    """
    Generate typical collocations using DeepSeek.
    Returns list of (chinese_collocation, english_translation) tuples.
    """
    system_prompt = """You are a Chinese collocation expert.

CRITICAL FORMAT REQUIREMENT:
Every line MUST use this EXACT format: 中文搭配|English translation
The pipe symbol | is MANDATORY between Chinese and English.

RULES:
1. Each collocation must be 2-4 Chinese characters (NOT full sentences)
2. Each collocation must contain the original word
3. Provide SHORT English translation (1-3 words)
4. Give EXACTLY 5 collocations
5. Output ONLY the list, no numbering, no explanations

CORRECT EXAMPLE for 途径:
有效途径|effective means
法律途径|legal channel
外交途径|diplomatic channel
和平途径|peaceful means
正式途径|official channel

WRONG (missing pipe or English):
有效途径 ❌
途径之一 ❌"""

    user_prompt = f"Generate 5 collocations for: {chinese_word}"

    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        
        result_text = response.choices[0].message.content.strip()
        logging.info(f"DeepSeek raw response: {result_text}")
        
        # Parse the response
        collocations = []
        lines = result_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            
            # MUST have pipe
            if '|' not in line:
                logging.warning(f"Skipping line without pipe: {line}")
                continue
            
            # Split by pipe
            parts = line.split('|', 1)
            
            if len(parts) == 2:
                chinese = parts[0].strip()
                english = parts[1].strip()
                
                # Clean quotes
                chinese = chinese.replace('"', '').replace('"', '').replace('"', '')
                english = english.replace('"', '').replace('"', '').replace('"', '')
                
                # Only SHORT collocations (2-6 chars)
                if chinese and english and 2 <= len(chinese) <= 6:
                    collocations.append((chinese, english))
                else:
                    logging.warning(f"Rejected (too long): {chinese} ({len(chinese)} chars)")
        
        if collocations:
            return collocations[:5]
        else:
            logging.error(f"No valid collocations parsed from: {result_text}")
            return [(f"{chinese_word}用法", "usage")]
            
    except Exception as e:
        logging.error(f"DeepSeek collocation error: {e}")
        return [(f"{chinese_word}用法", "usage")]

# ------------------ GOOGLE SHEETS OPERATIONS ------------------
def save_collocation_to_sheet(chinese: str, english: str) -> bool:
    """Save a collocation to Google Sheets with timestamp"""
    try:
        from datetime import datetime
        
        client = get_google_sheets_client()
        if not client:
            logging.error("Google Sheets client not initialized")
            return False
        
        # Open by URL then get worksheet by name
        spreadsheet = client.open_by_url(SPREADSHEET_URL)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Row: Chinese | English | Timestamp
        row = [chinese, english, timestamp]
        worksheet.append_row(row, value_input_option="USER_ENTERED")
        logging.info(f"Saved to sheet: {chinese} | {english} | {timestamp}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save to sheet: {e}")
        return False

# ------------------ IMAGE PROMPT GENERATION ------------------
async def generate_image_prompt(user_phrase: str) -> str:
    """For Chinese input: create scene description in Russian for Yandex API"""
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        f"Создай краткое, конкретное описание сцены для генерации изображения на основе китайской фразы: '{user_phrase}'. "
        "Требования: "
        "1. Действие происходит в Китае (город Чэнду); "
        "2. Главный персонаж - молодой китаец или китаянка; "
        "3. Включи конкретное место (парк, чайная, библиотека, улица), действие, погоду, освещение; "
        "4. Опиши сцену кратко и визуально, одним-двумя предложениями. "
        "Выведи ТОЛЬКО описание сцены по-русски, без объяснений и без повтора исходной фразы."
    )
    
    data = {
        "modelUri": f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-lite",
        "completionOptions": {
            "stream": False,
            "temperature": 0.7,
            "maxTokens": 200
        },
        "messages": [
            {
                "role": "system",
                "text": system_prompt
            },
            {
                "role": "user",
                "text": f"Фраза: {user_phrase}"
            }
        ]
    }
    
    try:
        response = requests.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        
        if 'result' in result and 'alternatives' in result['result']:
            prompt = result['result']['alternatives'][0]['message']['text'].strip()
            logging.info(f"Generated image prompt: {prompt}")
            return prompt
        return f"成都街头，年轻人正在体现'{user_phrase}'的概念，自然光线，日常环境"
    except Exception as e:
        logging.error(f"Yandex prompt generation error: {e}")
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
                    
                    return img_path
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
        '🖼️ 中文学习助手\n\n'
        '功能：\n'
        '1. 发送中文词语 → 定义 + 图片\n'
        '2. 发送"中文词 col" → 搭配按钮(保存到表格)\n'
        '3. 发送英文描述 → 直接生成图片\n\n'
        '例如:\n'
        '• "激发" → 定义和图片\n'
        '• "激发 col" → 搭配列表\n'
        '• "a tired donkey" → 生成图片\n\n'
        '⚠️ 图片生成需要1-3分钟'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if not text or text.startswith('/'):
        return

    # Check if this is a collocation request
    chinese_word = extract_collocation_request(text)
    
    if chinese_word:
        # MODE 2: Collocation request
        await update.message.reply_text(f'📚 正在查找 "{chinese_word}" 的常用搭配...')
        
        collocations = await generate_collocations(chinese_word)
        
        if not collocations:
            await update.message.reply_text("❌ 未能生成搭配，请重试。")
            return
        
        # Store collocations in cache
        chat_id = update.message.chat_id
        COLLOCATION_CACHE[chat_id] = collocations
        
        # Create inline keyboard
        keyboard = []
        for idx, (chinese, english) in enumerate(collocations):
            button_text = f"{chinese} {english}"
            if len(button_text) > 60:
                button_text = button_text[:57] + "..."
            callback_data = f"save:{idx}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"✅ 找到 {len(collocations)} 个常用搭配：\n点击按钮保存到表格",
            reply_markup=reply_markup
        )
        return
    
    # Check if Chinese-only (MODE 1) or non-Chinese (MODE 3)
    if is_chinese(text):
        # MODE 1: Chinese word only → Definition + Picture
        await update.message.reply_text(f'📖 正在查找 "{text}" 的定义...')
        
        # Generate definition
        definition = await generate_definition(text)
        await update.message.reply_text(f"📝 {definition}")
        
        # Generate image prompt
        await update.message.reply_text('🎨 正在生成图像...')
        image_prompt = await generate_image_prompt(text)
        logging.info(f"Image prompt for Chinese: {image_prompt}")
        
        # Generate image
        img_path = await generate_image_with_yandex(image_prompt, update)
        if not img_path:
            await update.message.reply_text("❌ 图像生成失败，请重试。")
            return
        
        try:
            with open(img_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=f"🖼️ {text}")
            os.remove(img_path)
            logging.info(f"Sent definition + image for: {text}")
        except Exception as e:
            logging.error(f"Send image exception: {e}")
            await update.message.reply_text(f"⚠️ 发送图片失败: {str(e)}")
            if os.path.exists(img_path):
                os.remove(img_path)
    else:
        # MODE 3: Non-Chinese description → Direct image generation
        await update.message.reply_text(f'🎨 Generating: "{text}"...')
        
        # Use text directly as prompt
        img_path = await generate_image_with_yandex(text, update)
        if not img_path:
            await update.message.reply_text("❌ Image generation failed. Please try again.")
            return
        
        try:
            with open(img_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=f"✅ Prompt: {text}")
            os.remove(img_path)
            logging.info(f"Sent image for English prompt: {text}")
        except Exception as e:
            logging.error(f"Send image exception: {e}")
            await update.message.reply_text(f"⚠️ Failed to send image: {str(e)}")
            if os.path.exists(img_path):
                os.remove(img_path)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks for collocation saving"""
    query = update.callback_query
    await query.answer()
    
    # Parse callback data
    if not query.data.startswith("save:"):
        await query.edit_message_text("❌ 无效的按钮数据")
        return
    
    try:
        # Get index from callback data
        idx = int(query.data.split(":")[1])
        
        # Retrieve collocations from cache
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
    
    print("✅ Bot is running with 3 modes:")
    print("   1. Chinese word → Definition + Picture")
    print("   2. Chinese word + 'col' → Collocations to Google Sheets")
    print("   3. English description → Direct picture generation")
    print("⚠️ Note: Image generation may take 1-3 minutes")
    app.run_polling()

if __name__ == '__main__':
    main()
