# RAG Document Crawler Automation

This automation system uses n8n to run a weekly web crawler that monitors RAG-related documentation for updates and sends email notifications to your team.

## 🎯 Features

- **Weekly Automated Crawling**: Runs every Monday at 9 AM
- **Change Detection**: Tracks document changes using content hashing
- **Section Extraction**: Identifies and lists main sections from each document
- **Email Notifications**: Sends formatted HTML emails with:
  - New documents discovered
  - Updated documents with changes
  - Section listings with examples
  - Crawl statistics and summaries
- **State Persistence**: Maintains crawl history to detect changes
- **Failure Handling**: Reports failed URLs and continues processing

## 📋 Prerequisites

### 1. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `requests` - HTTP client for web fetching
- `beautifulsoup4` - HTML parsing and content extraction
- Standard libraries: `json`, `hashlib`, `pathlib`

### 2. n8n Installation

Install n8n (choose one method):

**Using npm:**
```bash
npm install -g n8n
```

**Using Docker:**
```bash
docker pull n8nio/n8n
```

**Using npx (no installation):**
```bash
npx n8n
```

For detailed installation instructions, visit: https://docs.n8n.io/hosting/installation/

## 🚀 Setup Instructions

### Step 1: Configure Crawler URLs

Edit `automation/crawler/config/crawler_config.json` to add your target URLs:

```json
{
  "urls": [
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://docs.llamaindex.ai/en/stable/getting_started/starter_example.html",
    "https://www.pinecone.io/learn/retrieval-augmented-generation/",
    "https://huggingface.co/blog/rag",
    "https://blog.langchain.dev/deconstructing-rag/"
  ],
  "output_dir": "data/crawled_docs",
  "state_file": "automation/crawler/crawler_state.json",
  "email_recipients": [
    "team@example.com"
  ]
}
```

**Configuration Options:**
- `urls`: Array of URLs to crawl weekly
- `output_dir`: Directory to save crawled documents
- `state_file`: JSON file tracking crawl history
- `email_recipients`: List of email addresses for notifications

### Step 2: Configure Email Settings (SMTP)

You need to configure SMTP settings in the n8n workflow. Edit the email nodes in `automation/n8n/workflows/weekly_crawler_workflow.json`:

**Common SMTP Providers:**

**Gmail:**
```
Host: smtp.gmail.com
Port: 587
User: your-email@gmail.com
Password: your-app-password (not your regular password)
```
Note: Enable 2FA and create an App Password in your Google Account settings.

**Outlook/Office 365:**
```
Host: smtp.office365.com
Port: 587
User: your-email@outlook.com
Password: your-password
```

**SendGrid:**
```
Host: smtp.sendgrid.net
Port: 587
User: apikey
Password: your-sendgrid-api-key
```

**Custom SMTP:**
```
Host: your-smtp-server.com
Port: 587 (or 465 for SSL)
User: your-username
Password: your-password
```

### Step 3: Import n8n Workflow

1. Start n8n:
   ```bash
   n8n start
   ```

2. Open n8n in your browser (default: http://localhost:5678)

3. Import the workflow:
   - Click "Add Workflow" → "Import from File"
   - Select `automation/n8n/workflows/weekly_crawler_workflow.json`
   - Or paste the JSON content directly

4. Configure the email nodes:
   - Click on "Send Email - With Updates" node
   - Add your SMTP credentials
   - Set the recipient email(s)
   - Repeat for "Send Email - No Updates" node

5. Activate the workflow:
   - Click the toggle switch to activate
   - The workflow will run every Monday at 9 AM

### Step 4: Test the Crawler

Run the crawler manually to test:

```bash
cd /home/user/Ai_Learning
python automation/crawler/web_crawler.py automation/crawler/config/crawler_config.json
```

Expected output:
```
2025-11-16 09:00:00 - INFO - Starting crawl...
2025-11-16 09:00:01 - INFO - Fetching: https://python.langchain.com/docs/tutorials/rag/
2025-11-16 09:00:03 - INFO - New document: https://python.langchain.com/docs/tutorials/rag/
...
2025-11-16 09:00:15 - INFO - Crawl complete. New: 5, Updated: 0, Unchanged: 0, Failed: 0

================================================================================
WEEKLY DOCUMENT CRAWLER REPORT
================================================================================
...
```

The results are saved to:
- `automation/crawler/last_crawl_results.json` - Full JSON results
- `automation/crawler/crawler_state.json` - State tracking
- `data/crawled_docs/*.txt` - Downloaded documents

## 📊 Workflow Architecture

```
┌─────────────────────────────┐
│  Schedule Trigger           │
│  (Every Monday 9 AM)        │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Run Crawler Script         │
│  (Python execution)         │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Read Crawl Results         │
│  (Load JSON file)           │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Format Email Content       │
│  (Generate HTML)            │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Check If Has Updates       │
│  (Conditional branch)       │
└──────┬────────────┬─────────┘
       │            │
   YES │            │ NO
       │            │
       ▼            ▼
┌──────────┐  ┌──────────┐
│  Email   │  │  Email   │
│  With    │  │  Summary │
│  Updates │  │  Only    │
└──────────┘  └──────────┘
```

## 📧 Email Template

The email includes:

### Statistics Section
- Count of new documents
- Count of updated documents
- Count of unchanged documents
- Count of failed URLs

### New Documents Section
For each new document:
- Title
- URL
- List of main sections with headings
- Visual badge: "NEW"

### Updated Documents Section
For each updated document:
- Title
- URL
- Current section structure
- Visual badge: "UPDATED"

### Example Sections
Each document shows its main sections (up to 10), for example:
- Introduction to RAG
- Architecture Overview
- Implementation Steps
- Best Practices
- Common Pitfalls

## 🔧 Customization

### Change Crawl Schedule

Edit the schedule trigger in the n8n workflow:

```json
{
  "parameters": {
    "rule": {
      "interval": [
        {
          "field": "weeks",
          "weeksInterval": 1,
          "triggerAtHour": 9,
          "triggerAtMinute": 0,
          "triggerAtDayOfWeek": 1
        }
      ]
    }
  }
}
```

Options:
- `weeksInterval`: 1 (weekly), 2 (bi-weekly), etc.
- `triggerAtHour`: 0-23 (24-hour format)
- `triggerAtDayOfWeek`: 1 (Monday) to 7 (Sunday)

### Add More URLs

Add URLs to `crawler_config.json`:

```json
{
  "urls": [
    "https://your-new-url.com/documentation",
    "https://another-url.com/blog/rag"
  ]
}
```

### Customize Email Template

Edit the JavaScript code in the "Format Email Content" node in the n8n workflow. The template uses inline CSS for maximum email client compatibility.

### Change Detection Logic

The crawler uses SHA-256 hashing to detect content changes. To modify detection sensitivity, edit `web_crawler.py`:

```python
def _calculate_hash(self, content: str) -> str:
    """Customize what content is hashed"""
    # Current: Hash entire content
    # Option: Hash only main sections
    # Option: Hash with normalized whitespace
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

## 📁 Directory Structure

```
automation/
├── README.md                          # This file
├── crawler/
│   ├── web_crawler.py                 # Main crawler script
│   ├── config/
│   │   └── crawler_config.json        # Configuration
│   ├── crawler_state.json             # State tracking (auto-generated)
│   └── last_crawl_results.json        # Latest results (auto-generated)
└── n8n/
    └── workflows/
        └── weekly_crawler_workflow.json   # n8n workflow definition

data/
└── crawled_docs/                      # Downloaded documents (auto-generated)
    ├── abc123.txt
    ├── def456.txt
    └── ...
```

## 🐛 Troubleshooting

### Crawler fails with "Connection timeout"

**Solution:** Increase timeout in `crawler_config.json`:
```json
{
  "crawl_settings": {
    "timeout": 60,
    "retry_count": 3
  }
}
```

### Email not sending

**Common issues:**
1. **Wrong SMTP credentials**: Verify username/password
2. **App password required**: Gmail requires app-specific passwords
3. **Firewall blocking**: Check port 587 or 465 is open
4. **Wrong host**: Verify SMTP server address

**Debug steps:**
- Test SMTP settings in n8n workflow manually
- Check n8n execution logs for detailed errors
- Try sending a test email from the email node

### No changes detected

**Reasons:**
1. **First run**: All documents are "new" on first crawl
2. **No actual changes**: Pages haven't been updated
3. **Dynamic content**: Some sites have timestamps/ads that change

**Solution:** Check `crawler_state.json` to see stored hashes.

### Some URLs failing

**Check:**
1. **URL accessibility**: Can you access the URL in a browser?
2. **Rate limiting**: Site may block automated requests
3. **Authentication required**: Some sites need login
4. **JavaScript-heavy**: Crawler doesn't execute JavaScript

**Solution:**
- For JavaScript sites, consider using Selenium (requires code modification)
- Add delays between requests in config

## 🔐 Security Best Practices

1. **Never commit SMTP credentials** to version control
2. **Use environment variables** for sensitive data
3. **Use app-specific passwords** instead of main passwords
4. **Restrict n8n access** if exposing to network
5. **Validate URLs** before adding to config
6. **Monitor crawler logs** for suspicious activity

## 📈 Monitoring & Logs

### View Crawler Logs

The crawler outputs logs to stdout:
```bash
python automation/crawler/web_crawler.py | tee crawler.log
```

### View n8n Execution History

In n8n UI:
1. Click on "Executions" tab
2. View status: Success/Error/Running
3. Click on execution to see detailed logs

### Check Crawl Results

```bash
# View last results
cat automation/crawler/last_crawl_results.json | python -m json.tool

# View state
cat automation/crawler/crawler_state.json | python -m json.tool

# List downloaded documents
ls -lh data/crawled_docs/
```

## 🚦 Next Steps

1. ✅ Configure your URLs in `crawler_config.json`
2. ✅ Set up SMTP credentials in n8n workflow
3. ✅ Import workflow to n8n
4. ✅ Test crawler manually
5. ✅ Activate workflow
6. ✅ Monitor first automated run

## 🤝 Contributing

To add new features:
1. Fork the repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## 📚 Additional Resources

- [n8n Documentation](https://docs.n8n.io/)
- [RAG Learning Course](../README.md)
- [LangChain Documentation](https://python.langchain.com/)
- [Web Scraping Best Practices](https://www.scrapehero.com/web-scraping-best-practices/)

## ⚖️ Legal & Ethics

**Important:**
- Respect `robots.txt` files
- Don't overload servers with requests
- Review terms of service for each site
- Use appropriate delays between requests
- Only crawl publicly accessible content

---

**Questions?** Open an issue or check the main [README](../README.md)
