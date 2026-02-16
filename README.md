# Chinese Learning Picture Bot

A Telegram bot that helps with Chinese language learning through three main features:
1. **Word definitions with AI-generated images** - Enter a Chinese word to get its definition and a contextual image
2. **Collocation management** - Request collocations and save them to Google Sheets
3. **Custom image generation** - Enter English descriptions to generate custom images

## Features

### Mode 1: Chinese Word → Definition + Picture
Enter a Chinese word (e.g., `激发`) and get:
- A definition in Chinese with example sentences
- An AI-generated image showing the word in context (set in Chengdu, China)

### Mode 2: Chinese Word + "col" → Collocations
Enter a Chinese word followed by `col` (e.g., `激发 col`) to:
- Generate 5 common collocations using the word
- Display them as clickable buttons
- Save selected collocations to Google Sheets with timestamps

### Mode 3: English Description → Picture
Enter an English description (e.g., `a tired donkey next to a milestone`) to:
- Generate an image directly from your description

## Setup

### Prerequisites
- Python 3.8+
- Telegram Bot Token
- Yandex Cloud API credentials (API Key + Folder ID)
- DeepSeek API Key
- Google Cloud Service Account (for Sheets integration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/johnaaiton-art/generate-picture-bot.git
cd generate-picture-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export TELEGRAM_BOT_TOKEN="your-telegram-token"
export YANDEX_API_KEY="your-yandex-api-key"
export YANDEX_FOLDER_ID="your-yandex-folder-id"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

4. Add Google credentials:
- Place your `google-creds.json` file in the project directory
- Update the `SPREADSHEET_URL` in `bot_fixed.py` to point to your Google Sheet

5. Run the bot:
```bash
python bot_fixed.py
```

## Usage

Start the bot with `/start` to see all available commands.

**Examples:**
- `激发` - Get definition and picture
- `激发 col` - Get collocations with save buttons
- `a tired donkey next to a milestone` - Generate custom image

## API Integration

### Yandex Cloud
Used for:
- Text generation (definitions and image prompts)
- Image generation via Yandex Art API

### DeepSeek
Used for:
- Generating Chinese collocations

### Google Sheets
Used for:
- Saving collocations with timestamps

## File Structure

```
generate-picture-bot/
├── bot_fixed.py          # Main bot code
├── requirements.txt      # Python dependencies
├── google-creds.json     # Google service account credentials (not in repo)
├── README.md            # This file
├── SETUP_INSTRUCTIONS   # Detailed setup guide
└── USER_GUIDE          # User-facing documentation
```

## Deployment

### Running as a systemd service (Linux)

1. Create a service file at `/etc/systemd/system/picture-bot.service`:
```ini
[Unit]
Description=Picture Creator Telegram Bot
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/generate-picture-bot
Environment="TELEGRAM_BOT_TOKEN=your-token"
Environment="YANDEX_API_KEY=your-key"
Environment="YANDEX_FOLDER_ID=your-folder"
Environment="DEEPSEEK_API_KEY=your-key"
ExecStart=/usr/bin/python3 bot_fixed.py
Restart=always

[Install]
WantedBy=multi-user.target
```

2. Enable and start:
```bash
sudo systemctl enable picture-bot.service
sudo systemctl start picture-bot.service
```

## Troubleshooting

### Image generation timeout
Images can take 1-3 minutes to generate. The bot will notify you if it's taking longer than 30 seconds.

### Google Sheets not saving
Check that:
- `google-creds.json` exists in the project directory
- The service account has edit permissions on your spreadsheet
- The `SPREADSHEET_URL` and `SHEET_NAME` are correct

### Bot not responding
Check logs:
```bash
sudo journalctl -u picture-bot.service -f
```

## License

MIT License - feel free to modify and use for your own learning purposes.

## Contributing

Pull requests welcome! Please ensure:
- Code follows existing style
- New features include documentation
- API keys are never committed

## Acknowledgments

- Yandex Cloud for image generation API
- DeepSeek for language model capabilities
- Google Sheets for data persistence
