# User Guide: Chinese Learning Picture Bot

## Quick Start

1. Find the bot on Telegram
2. Send `/start` to see the welcome message
3. Start learning!

## Three Ways to Use the Bot

### 1️⃣ Learn a Chinese Word
**What to type:** Just the Chinese word (e.g., `激发`)

**What you get:**
- A clear definition in Chinese
- 2-3 example sentences
- An AI-generated picture showing the concept in a real-world scene

**Example:**
```
You: 激发
Bot: 📝 定义：激发指激励、唤起或促使某种情绪、行为或状态产生...
     例句:
     1. 这个故事激发了我的创造力
     2. 老师的话激发了学生们的学习热情
     🖼️ [sends picture of a young person in Chengdu being inspired]
```

### 2️⃣ Get Collocations (and Save Them!)
**What to type:** Chinese word + space + `col` (e.g., `激发 col`)

**What you get:**
- 5 common collocations using that word
- Each shown as a clickable button with English translation
- Click any button to save it to your Google Sheet

**Example:**
```
You: 激发 col
Bot: ✅ 找到 5 个常用搭配：
     点击按钮保存到表格
     
     [激发兴趣 spark interest]
     [激发潜能 unleash potential]
     [激发创造力 stimulate creativity]
     [激发灵感 inspire ideas]
     [激发热情 ignite passion]
```

**After clicking a button:**
```
Bot: ✅ 已保存:
     中文: 激发兴趣
     英文: spark interest
     
     已添加到表格！
```

### 3️⃣ Generate Custom Pictures
**What to type:** Any English description (e.g., `a tired donkey next to a milestone`)

**What you get:**
- An AI-generated image of exactly what you described
- No Chinese context - just your creative vision!

**Example:**
```
You: a tired donkey next to a milestone
Bot: 🎨 Generating: "a tired donkey next to a milestone"...
     ⏳ 图像生成队列较长，请继续等待...
     🖼️ [sends picture of a tired donkey by a milestone]
```

## Tips & Tricks

### For Best Results:

**Chinese words:**
- Use 2-4 character words for best collocations
- Single characters work but may be very broad
- Longer phrases work for definitions but not collocations

**Collocation requests:**
- Format MUST be: `[word] col` or `[word] collocation`
- Space between word and "col" is required
- Case doesn't matter: `col`, `Col`, or `COL` all work

**English descriptions:**
- Be specific: "young woman reading in a coffee shop" beats "reading"
- Include details: weather, time of day, mood, colors
- Keep it under 100 words for best results

### Timing Expectations:

- **Definitions:** Usually 5-10 seconds
- **Collocations:** Usually 10-15 seconds
- **Pictures:** 1-3 minutes (sometimes longer during busy times)

If picture generation takes over 30 seconds, the bot will send you a "please wait" message.

## Common Questions

**Q: Can I request multiple collocations at once?**
A: No, but you can quickly send multiple requests one after another.

**Q: Where are my saved collocations stored?**
A: In your personal Google Sheet. Ask your teacher/administrator for the link.

**Q: What if the definition is wrong or unclear?**
A: The AI does its best, but for authoritative definitions, cross-reference with a dictionary.

**Q: Can I generate pictures in other languages?**
A: Chinese words create scenes in China (Chengdu). Other languages create generic scenes based on your description.

**Q: Why is my picture generation stuck?**
A: The image queue can be busy. Wait up to 3 minutes. If it fails, try again in a few minutes.

**Q: Can I delete a collocation I saved by mistake?**
A: Not through the bot. You'll need to edit the Google Sheet directly.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Bot not responding | Wait 30 seconds, then try `/start` |
| Collocation buttons not working | Request again with proper format: `word col` |
| Picture taking too long | Wait up to 3 minutes. Server may be busy. |
| Wrong definition | Try rephrasing or use a dictionary |
| "保存失败" (Save failed) | Contact administrator - Google Sheets may need setup |

## Examples Gallery

**Learning vocabulary:**
```
学习 → definition + picture of students in Chengdu library
坚持 → definition + picture of young person exercising in park
创新 → definition + picture of innovator in tech office
```

**Building collocation library:**
```
途径 col → 有效途径, 法律途径, 外交途径, etc.
方法 col → 科学方法, 教学方法, 研究方法, etc.
```

**Creative image generation:**
```
sunset over mountains with pink clouds
busy street market with colorful vegetables
quiet library with sunlight through windows
```

## Support

If you encounter issues:
1. Try `/start` to reset
2. Check your internet connection
3. Wait a few minutes and try again
4. Contact your administrator if problems persist

Happy learning! 加油！
