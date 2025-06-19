# Streamlit Cloud Deployment Guide

## Quick Start

1. **Fork/Upload to GitHub**
   - Upload this project to a GitHub repository
   - Make sure `app.py` and `requirements.txt` are in the root directory

2. **Deploy to Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file path to `app.py`

3. **Configure Secrets**
   - In Streamlit Cloud dashboard, go to your app settings
   - Add secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "sk-your-openai-key-here"
   PINECONE_API_KEY = "your-pinecone-key-here"
   ```

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Add your API keys to .streamlit/secrets.toml
# Run the app
streamlit run app.py
```

## Key Features for Production

- ✅ **Session Isolation**: Each user gets unique Pinecone index
- ✅ **Secrets Management**: API keys via Streamlit secrets
- ✅ **Error Handling**: Comprehensive error messages
- ✅ **Resource Cleanup**: Automatic cleanup on session reset
- ✅ **Chat History**: Last 5 Q&A pairs stored
- ✅ **Responsive UI**: Clean, professional interface

## Configuration Options

### Environment Variables (Alternative to Secrets)
```bash
export OPENAI_API_KEY="your-key"
export PINECONE_API_KEY="your-key"
```

### Customization
- Modify `chunk_size` and `chunk_overlap` in `process_pdf()` for different document types
- Adjust `temperature` and `max_tokens` in `ask_question()` for different AI behavior
- Change Pinecone region in `setup_pinecone()` for better performance

## Troubleshooting

**Common Issues:**
1. **Pinecone quota exceeded**: Each session creates a new index
2. **OpenAI rate limits**: Implement request throttling if needed
3. **Large PDFs**: May hit processing timeouts on Streamlit Cloud

**Solutions:**
- Use session reset to clean up unused indexes
- Implement index reuse for returning users
- Split large documents before upload