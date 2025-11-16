# Quick Start: Weekly Crawler Setup (5 minutes)

Get your weekly RAG documentation crawler running in 5 minutes!

## Step 1: Install Dependencies (1 min)

```bash
# Install Python requirements
pip install requests beautifulsoup4

# Install n8n (choose one)
npm install -g n8n
# OR use npx (no installation): npx n8n
```

## Step 2: Configure URLs (1 min)

Edit `automation/crawler/config/crawler_config.json`:

```json
{
  "urls": [
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://www.pinecone.io/learn/retrieval-augmented-generation/"
  ],
  "email_recipients": ["your-team@example.com"]
}
```

## Step 3: Test Crawler (1 min)

```bash
cd /home/user/Ai_Learning
python automation/crawler/web_crawler.py automation/crawler/config/crawler_config.json
```

You should see:
```
INFO - Starting crawl...
INFO - Fetching: https://python.langchain.com/docs/tutorials/rag/
INFO - New document: https://python.langchain.com/docs/tutorials/rag/
INFO - Crawl complete. New: 2, Updated: 0, Unchanged: 0, Failed: 0
```

## Step 4: Setup n8n (2 min)

```bash
# Start n8n
n8n start

# Open browser to http://localhost:5678
```

In n8n:
1. Click **"Add Workflow"** → **"Import from File"**
2. Select `automation/n8n/workflows/weekly_crawler_workflow.json`
3. Click on **"Send Email - With Updates"** node
4. Configure SMTP:
   - **Host:** `smtp.gmail.com`
   - **Port:** `587`
   - **User:** `your-email@gmail.com`
   - **Password:** Your Gmail app password ([create one here](https://myaccount.google.com/apppasswords))
   - **To:** `team@example.com`
5. Repeat for **"Send Email - No Updates"** node
6. Click **Save**
7. Toggle **Active** switch to ON

## Done! 🎉

Your crawler will now run **every Monday at 9 AM** and email your team with:
- ✅ New documents found
- ✅ Updated documents with changes
- ✅ Section listings and examples
- ✅ Weekly statistics

## What You'll Receive

### Email Example:

```
📚 Weekly RAG Document Update Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 STATISTICS
   2 New Documents
   1 Updated Documents
   4 Unchanged
   0 Failed

📄 NEW DOCUMENTS

   [NEW] Build a Retrieval Augmented Generation (RAG) App
   🔗 https://python.langchain.com/docs/tutorials/rag/

   Main Sections:
   • Overview
   • Architecture
   • Document Loading and Preparation
   • Retrieval and Generation
   • Choosing LLMs and Embeddings
   • Production Considerations
```

## Next Steps

- **Customize schedule:** See [README.md](README.md#change-crawl-schedule)
- **Add more URLs:** Edit `crawler_config.json`
- **Change email template:** Edit "Format Email Content" node in n8n
- **View logs:** Check n8n Executions tab

## Troubleshooting

**Problem:** Email not sending
- **Solution:** Verify SMTP credentials, use app password for Gmail

**Problem:** Crawler fails
- **Solution:** Check internet connection, verify URLs are accessible

**Problem:** n8n won't start
- **Solution:** Try `npx n8n` instead, or check if port 5678 is in use

---

**Full documentation:** [README.md](README.md)
