# Setup Instructions for Collocation Bot

## 1. Install Dependencies on Yandex VM

```bash
pip install -r requirements.txt --break-system-packages
```

## 2. Set Up Google Sheets Credentials

### Option A: Create Service Account (Recommended)

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable Google Sheets API and Google Drive API
4. Go to "IAM & Admin" → "Service Accounts"
5. Create a new service account
6. Click on the service account → "Keys" → "Add Key" → "Create new key" → JSON
7. Download the JSON file

### Option B: Use Existing Service Account

If you already have a service account JSON file, just use that.

### Upload to Yandex VM

```bash
# Upload the credentials file to the same directory as bot.py
# Name it exactly: google-creds.json

# Example location: /home/ubuntu/bot/google-creds.json
# Make sure bot.py is in: /home/ubuntu/bot/bot.py
```

## 3. Share Google Sheet with Service Account

1. Open the JSON credentials file
2. Find the "client_email" field (looks like: `xxx@xxx.iam.gserviceaccount.com`)
3. Copy that email
4. Open your Google Sheet: https://docs.google.com/spreadsheets/d/dfbfdbh5Vcl3_sfbfs/edit
5. Click "Share" button
6. Paste the service account email
7. Give it "Editor" permissions
8. Click "Send"

## 4. Verify Sheet Structure

Make sure your sheet "Chinese" has:
- Column A: Chinese text
- Column B: English translation
- First row can be headers (optional)

## 5. Environment Variables

Make sure these are set (in your systemd service or .env):

```bash
TELEGRAM_BOT_TOKEN=your_bot_token
DASHSCOPE_API_KEY=your_dashscope_key
YANDEX_API_KEY=your_yandex_key
YANDEX_FOLDER_ID=fsdvsv h
```

## 6. Test the Bot

### Test 1: Image Generation (Chinese)
Send: `美好生活`
Expected: Bot generates a Chengdu scene image

### Test 2: Image Generation (English)
Send: `sunset over mountains`
Expected: Bot generates exactly what you described

### Test 3: Collocation Lookup
Send: `途径 col`
Expected: 
- Bot shows 5 collocation buttons like:
  - 有效途径 effective means
  - 法律途径 legal channel
  - etc.

### Test 4: Save Collocation
Click one of the collocation buttons
Expected:
- Bot confirms: "✅ 已保存: 中文: xxx, 英文: xxx"
- Check your Google Sheet - new row should appear

## 7. Deploy as Systemd Service (Optional)

Update your existing systemd service file:

```ini
[Unit]
Description=Telegram Collocation Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bot
Environment="TELEGRAM_BOT_TOKEN=xxx"
Environment="DASHSCOPE_API_KEY=xxx"
Environment="YANDEX_API_KEY=xxx"
Environment="YANDEX_FOLDER_IDsffsfs
ExecStart=/usr/bin/python3 /home/ubuntu/bot/bot.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Then reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart your-bot-service
sudo systemctl status your-bot-service
```

## Troubleshooting

### "Google credentials file not found"
- Make sure `google-creds.json` is in the same directory as `bot.py`
- Check the file name is exactly `google-creds.json` (case-sensitive)

### "Permission denied" when saving
- Make sure you shared the sheet with the service account email
- Check the service account has "Editor" permissions

### Collocation buttons not working
- Check bot logs: `journalctl -u your-bot-service -f`
- Verify Qwen API is responding
- Test with a simple word like `途径`

### Wrong sheet or no data appearing
- Verify SPREADSHEET_URL in code matches your sheet
- Verify SHEET_NAME is exactly "Chinese" (case-sensitive)
- Check if service account has access to the sheet
