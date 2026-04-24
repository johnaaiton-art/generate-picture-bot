import os
import uuid
import logging
import requests
import asyncio
import tempfile
import base64
import re
import json
import hashlib
import zipfile
import random
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes, ConversationHandler
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
from google.cloud import texttospeech

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

# Chirp3-HD voices for Chinese TTS
CHINESE_CHIRP_VOICES = [
    "cmn-CN-Chirp3-HD-Aoede",
    "cmn-CN-Chirp3-HD-Leda",
    "cmn-CN-Chirp3-HD-Puck",
    "cmn-CN-Chirp3-HD-Fenrir",
]

# ConversationHandler state for /anki
ANKI_WAITING_DAYS = 1

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

def extract_dif_request(text: str) -> Optional[List[str]]:
    """
    Check if text is a 'dif' request.
    Formats: "指引 导致 dif" or "引导 指引 导致 dif" (2-3 Chinese words + 'dif')
    Returns list of Chinese words if valid, else None.
    """
    text = text.strip()
    if not re.search(r'\bdif\b\s*$', text, re.IGNORECASE):
        return None

    # Remove trailing 'dif'
    cleaned = re.sub(r'\bdif\b\s*$', '', text, flags=re.IGNORECASE).strip()

    # Split on whitespace or Chinese punctuation
    parts = re.split(r'[\s,、]+', cleaned)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) < 2 or len(parts) > 3:
        return None

    # All parts must be Chinese characters only
    chinese_pattern = re.compile(r'^[\u4e00-\u9fff]+$')
    for part in parts:
        if not chinese_pattern.match(part):
            return None

    return parts

# ------------------ DEFINITION GENERATION ------------------
async def generate_definition(chinese_word: str) -> str:
    """Generate definition and examples for a Chinese word using Yandex API"""
    headers = {
        "Authorization": f"Api-Key {YANDEX_API_KEY}",
        "Content-Type": "application/json"
    }
    
    system_prompt = (
        "你是一个中文词汇专家。用户会给你一个中文词语，"
        "请用简体中文提供简洁的定义和2-3个例句。"
        "要求：必须使用简体中文，不要使用繁体中文。"
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
6. Use SIMPLIFIED Chinese characters only (简体中文)

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

# ------------------ DIF GENERATION ------------------
async def generate_dif_content(words: List[str]) -> dict:
    """
    Call DeepSeek to explain differences between words and get 3 collocations each.
    Returns dict with 'explanation' and 'words' list.
    """
    words_str = '、'.join(words)

    # Build the words array section dynamically
    words_json_example = '\n    '.join(
        f'{{"word": "{w}", "collocations": [{{"chinese": "搭配例子", "english": "translation"}}, {{"chinese": "搭配例子", "english": "translation"}}, {{"chinese": "搭配例子", "english": "translation"}}]}}'
        for w in words
    )

    prompt = f"""You are an expert Chinese language teacher. Explain the differences between these Chinese words: {words_str}

Return ONLY valid JSON, no markdown, no backticks:
{{
  "explanation": "3-5 sentence English explanation of the semantic differences, nuances, register, and when to use each word. Be practical and specific.",
  "words": [
    {words_json_example}
  ]
}}

Rules:
- explanation: clear practical English, compare all words directly
- collocations: the 3 most typical real collocations for each word, 2-5 Chinese characters each, must contain the original word
- english: short translation of the collocation (1-4 words)
- Use SIMPLIFIED Chinese only
- ONLY valid JSON"""

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Chinese language expert. Return valid JSON only, no markdown."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        timeout=30.0
    )

    content_text = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    content_text = re.sub(r'^```(?:json)?\s*', '', content_text)
    content_text = re.sub(r'\s*```$', '', content_text)

    json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
    if json_match:
        content_text = json_match.group()

    return json.loads(content_text)


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
        '2. 发送「中文词 col」→ 搭配按钮 (保存到表格)\n'
        '3. 发送英文描述 → 直接生成图片\n'
        '4. 发送「词1 词2 dif」→ 词义对比 + 搭配按钮\n'
        '5. /anki → 生成 Anki 卡片 (TTS音频)\n\n'
        '例如:\n'
        '• 激发 → 定义和图片\n'
        '• 激发 col → 搭配列表\n'
        '• a tired donkey → 生成图片\n'
        '• 指引 导致 dif → 两词对比\n'
        '• 引导 指引 导致 dif → 三词对比\n'
        '• /anki → 导出最近N天的搭配为Anki卡片\n\n'
        '⚠️ 图片生成需要1-3分钟'
    )

async def handle_dif_request(update: Update, words: List[str]):
    """Process a 'dif' request: send English explanation + collocation buttons per word."""
    words_display = ' / '.join(words)
    status_msg = await update.message.reply_text(f"🔍 Analysing: {words_display}...")

    try:
        dif_data = await generate_dif_content(words)
    except Exception as e:
        logging.error(f"DIF generation error: {e}")
        await status_msg.edit_text("❌ Could not analyse differences. Please try again.")
        return

    explanation = dif_data.get('explanation', 'No explanation available.')
    word_data = dif_data.get('words', [])

    # Build response text
    response_text = f"📊 *{words_display}*\n\n{explanation}"

    # Store collocations in cache (same structure as existing col feature)
    chat_id = update.message.chat_id
    all_collocations = []
    keyboard = []

    for wd in word_data:
        word = wd.get('word', '')
        collocations = wd.get('collocations', [])[:3]

        # Header row (non-clickable label)
        keyboard.append([InlineKeyboardButton(f"— {word} —", callback_data="noop")])

        row = []
        for col in collocations:
            chinese = col.get('chinese', '')
            english = col.get('english', '')
            if not chinese:
                continue
            idx = len(all_collocations)
            all_collocations.append((chinese, english))
            btn_label = f"{chinese} {english}"
            if len(btn_label) > 60:
                btn_label = btn_label[:57] + "..."
            callback_data = f"save:{idx}"
            row.append(InlineKeyboardButton(btn_label, callback_data=callback_data))

        if row:
            keyboard.append(row)

    # Cache all collocations for this chat
    COLLOCATION_CACHE[chat_id] = all_collocations

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

    await status_msg.delete()
    await update.message.reply_text(
        response_text,
        parse_mode='Markdown',
        reply_markup=reply_markup
    )


# ------------------ /ANKI COMMAND ------------------

async def anki_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /anki command - ask how many days to go back"""
    await update.message.reply_text(
        "🗂️ *Anki Card Generator*\n\n"
        "How many days back should I look?\n"
        "_(e.g. send `5` to get all collocations from the last 5 days)_",
        parse_mode='Markdown'
    )
    return ANKI_WAITING_DAYS


async def anki_receive_days(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receive the number of days, fetch collocations, generate TTS, send files"""
    text = update.message.text.strip()

    if not text.isdigit() or int(text) < 1:
        await update.message.reply_text("❌ Please send a number like `5`.", parse_mode='Markdown')
        return ANKI_WAITING_DAYS

    days = int(text)
    cutoff = datetime.now() - timedelta(days=days)

    status_msg = await update.message.reply_text(
        f"⏳ Fetching collocations from the last {days} days..."
    )

    # --- Load rows from Google Sheet ---
    try:
        client = get_google_sheets_client()
        if not client:
            await status_msg.edit_text("❌ Could not connect to Google Sheets.")
            return ConversationHandler.END

        spreadsheet = client.open_by_url(SPREADSHEET_URL)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        all_rows = worksheet.get_all_values()
    except Exception as e:
        logging.error(f"Anki sheet load error: {e}")
        await status_msg.edit_text(f"❌ Sheet error: {e}")
        return ConversationHandler.END

    # --- Filter rows by date (col C = index 2) ---
    # Sheet columns: A=Chinese, B=English, C=timestamp
    selected = []
    seen = set()
    for row in all_rows:
        if len(row) < 3:
            continue
        chinese = row[0].strip()
        english = row[1].strip()
        date_str = row[2].strip()
        if not chinese or not english or not date_str:
            continue
        try:
            # Parse timestamp like "2026-02-13 07:38:08"
            row_date = datetime.strptime(date_str[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if row_date >= cutoff:
            key = (chinese, english)
            if key not in seen:
                seen.add(key)
                selected.append((chinese, english))

    if not selected:
        await status_msg.edit_text(
            f"😕 No collocations found in the last {days} days."
        )
        return ConversationHandler.END

    await status_msg.edit_text(
        f"✅ Found {len(selected)} collocations.\n🎙️ Generating TTS audio..."
    )

    # --- Generate TTS for each Chinese phrase ---
    tts_client_obj = None
    try:
        creds = Credentials.from_service_account_file(
            GOOGLE_CREDS_FILE,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        tts_client_obj = texttospeech.TextToSpeechClient(credentials=creds)
    except Exception as e:
        logging.error(f"TTS client init error: {e}")
        await status_msg.edit_text(f"❌ TTS init failed: {e}")
        return ConversationHandler.END

    audio_files = {}   # filename -> bytes
    anki_rows = []     # (english, chinese, sound_tag)

    for i, (chinese, english) in enumerate(selected):
        voice_name = CHINESE_CHIRP_VOICES[i % len(CHINESE_CHIRP_VOICES)]
        audio_filename = f"zh_tts_{hashlib.md5(chinese.encode()).hexdigest()}.mp3"

        if audio_filename not in audio_files:
            try:
                synthesis_input = texttospeech.SynthesisInput(text=chinese)
                voice = texttospeech.VoiceSelectionParams(
                    language_code="cmn-CN",
                    name=voice_name
                )
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.MP3,
                    speaking_rate=0.85
                )
                response = tts_client_obj.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                    timeout=30
                )
                audio_files[audio_filename] = response.audio_content
                logging.info(f"TTS OK: {chinese} → {audio_filename} ({voice_name})")
            except Exception as e:
                logging.error(f"TTS failed for {chinese}: {e}")
                audio_filename = ""

        sound_tag = f"[sound:{audio_filename}]" if audio_filename else ""
        anki_rows.append((english, chinese, sound_tag))

    # --- Build .txt file ---
    txt_buffer = BytesIO()
    for english, chinese, sound_tag in anki_rows:
        line = f"{english}\t{chinese}\t{sound_tag}\n"
        txt_buffer.write(line.encode("utf-8"))
    txt_buffer.seek(0)

    # --- Build .zip of audio files ---
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname, fdata in audio_files.items():
            zf.writestr(fname, fdata)
    zip_buffer.seek(0)

    date_label = datetime.now().strftime("%Y%m%d")
    txt_filename = f"anki_chinese_{date_label}_last{days}days.txt"
    zip_filename = f"anki_audio_{date_label}_last{days}days.zip"

    await status_msg.delete()

    await update.message.reply_document(
        document=txt_buffer,
        filename=txt_filename,
        caption=(
            f"📄 *{len(anki_rows)} Anki cards* (last {days} days)\n"
            f"Columns: English | Chinese | Audio\n"
            f"Import into Anki, then extract the zip to your `collection.media` folder."
        ),
        parse_mode='Markdown'
    )

    await update.message.reply_document(
        document=zip_buffer,
        filename=zip_filename,
        caption=f"🔊 {len(audio_files)} audio files"
    )

    return ConversationHandler.END


async def anki_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("❌ Cancelled.")
    return ConversationHandler.END


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages."""
    user_input = update.message.text.strip()
    
    if not user_input or user_input.startswith('/'):
        return

    # Check if this is a 'dif' request
    dif_words = extract_dif_request(user_input)
    if dif_words:
        await handle_dif_request(update, dif_words)
        return

    # Check if this is a collocation request
    chinese_word = extract_collocation_request(user_input)
    
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
    if is_chinese(user_input):
        # MODE 1: Chinese word only → Definition + Picture
        await update.message.reply_text(f'📖 正在查找 "{user_input}" 的定义...')
        
        # Generate definition
        definition = await generate_definition(user_input)
        await update.message.reply_text(f"📝 {definition}")
        
        # Generate image prompt
        await update.message.reply_text('🎨 正在生成图像...')
        image_prompt = await generate_image_prompt(user_input)
        logging.info(f"Image prompt for Chinese: {image_prompt}")
        
        # Generate image
        img_path = await generate_image_with_yandex(image_prompt, update)
        if not img_path:
            await update.message.reply_text("❌ 图像生成失败，请重试。")
            return
        
        try:
            with open(img_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=f"🖼️ {user_input}")
            os.remove(img_path)
            logging.info(f"Sent definition + image for: {user_input}")
        except Exception as e:
            logging.error(f"Send image exception: {e}")
            await update.message.reply_text(f"⚠️ 发送图片失败: {str(e)}")
            if os.path.exists(img_path):
                os.remove(img_path)
    else:
        # MODE 3: Non-Chinese description → Direct image generation
        await update.message.reply_text(f'🎨 Generating: "{user_input}"...')
        
        # Use text directly as prompt
        img_path = await generate_image_with_yandex(user_input, update)
        if not img_path:
            await update.message.reply_text("❌ Image generation failed. Please try again.")
            return
        
        try:
            with open(img_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo, caption=f"✅ Prompt: {user_input}")
            os.remove(img_path)
            logging.info(f"Sent image for English prompt: {user_input}")
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
    if query.data == "noop":
        return  # Word label buttons — do nothing

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

    # /anki conversation
    anki_conv = ConversationHandler(
        entry_points=[CommandHandler("anki", anki_start)],
        states={
            ANKI_WAITING_DAYS: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, anki_receive_days)
            ],
        },
        fallbacks=[CommandHandler("cancel", anki_cancel)],
    )
    app.add_handler(anki_conv)

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