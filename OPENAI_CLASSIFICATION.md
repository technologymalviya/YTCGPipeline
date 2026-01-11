# OpenAI Genre Classification Integration

## Overview

The genre classification system now supports OpenAI API for more accurate classification, with automatic fallback to keyword-based classification on any error or rate limit.

## Features

### ✅ Robust Error Handling
- **Rate Limit Protection**: Automatically tracks and enforces rate limits (500 RPM, 50K TPM)
- **Exponential Backoff**: Retries with exponential backoff on rate limit errors
- **Graceful Fallback**: Falls back to keyword-based classification on any error
- **No Script Failures**: Script never fails due to OpenAI issues

### ✅ Cost-Effective
- Uses `gpt-4o-mini` model (cheapest GPT-4 model)
- Minimal token usage (max 10 tokens per request)
- Low temperature (0.1) for consistent results

### ✅ Rate Limiting
- **Requests Per Minute (RPM)**: 500 requests/minute
- **Tokens Per Minute (TPM)**: 50,000 tokens/minute
- Automatic tracking and enforcement
- Falls back to keyword-based if limits reached

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install `openai>=1.0.0` along with other dependencies.

### 2. Configure API Key

Set the OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Or in GitHub Actions:

```yaml
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### 3. Usage

The system works automatically! No code changes needed.

- If `OPENAI_API_KEY` is set and OpenAI is available → Uses OpenAI
- If OpenAI fails or is unavailable → Falls back to keyword-based classification
- Script never fails due to OpenAI issues

## How It Works

1. **Primary**: Tries OpenAI classification first
2. **Fallback**: If OpenAI fails (rate limit, error, unavailable), uses keyword-based classification
3. **Transparent**: All errors are logged but don't stop the script

## Configuration

You can adjust these constants in `generate_json.py`:

```python
OPENAI_MODEL = "gpt-4o-mini"  # Model to use
OPENAI_MAX_TOKENS = 10  # Max tokens in response
OPENAI_TEMPERATURE = 0.1  # Temperature (lower = more consistent)
OPENAI_RATE_LIMIT_RPM = 500  # Requests per minute
OPENAI_RATE_LIMIT_TPM = 50000  # Tokens per minute
OPENAI_MAX_RETRIES = 3  # Max retries on rate limit
OPENAI_RETRY_DELAY_BASE = 1  # Base delay for exponential backoff
OPENAI_REQUEST_TIMEOUT = 30  # Request timeout in seconds
```

## Error Handling

The system handles these errors gracefully:

- ✅ **RateLimitError**: Retries with exponential backoff, falls back if max retries exceeded
- ✅ **APIConnectionError**: Falls back immediately
- ✅ **APIError**: Falls back immediately
- ✅ **Missing API Key**: Falls back immediately
- ✅ **OpenAI Not Installed**: Falls back immediately
- ✅ **Token Limit Exceeded**: Falls back immediately
- ✅ **Request Timeout**: Falls back immediately

## Cost Estimation

With `gpt-4o-mini`:
- **Input**: ~200-300 tokens per request
- **Output**: ~5-10 tokens per request
- **Cost**: ~$0.00015 per classification
- **1000 classifications**: ~$0.15

## Monitoring

The system logs all OpenAI operations:

```
[OpenAI] Rate limit hit, waiting 1s before retry 1/3
[OpenAI] Rate limit exceeded after 3 retries, falling back to keyword-based classification
[OpenAI] Connection error: ..., falling back to keyword-based classification
[OpenAI] Unexpected genre response: ..., falling back to keyword-based classification
```

## Testing

Test without OpenAI (keyword-based only):
```bash
# Don't set OPENAI_API_KEY
python generate_json.py
```

Test with OpenAI:
```bash
export OPENAI_API_KEY="sk-your-key"
python generate_json.py
```

## Best Practices

1. **Start with keyword-based**: Test without OpenAI first
2. **Monitor costs**: Check OpenAI usage dashboard
3. **Adjust rate limits**: If you have higher tier, increase `OPENAI_RATE_LIMIT_RPM` and `OPENAI_RATE_LIMIT_TPM`
4. **Use fallback**: The keyword-based system is reliable, OpenAI is enhancement

## Troubleshooting

### OpenAI not being used
- Check if `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`
- Check if `openai` package is installed: `pip list | grep openai`
- Check logs for error messages

### Rate limit errors
- Increase `OPENAI_RATE_LIMIT_RPM` if you have higher tier
- Reduce number of videos processed per run
- The system will automatically fall back to keyword-based

### Unexpected classifications
- Check OpenAI response in logs
- Adjust prompt in `classify_genre_with_openai()` function
- The system validates responses and falls back if invalid

## Notes

- The system is **completely optional** - works fine without OpenAI
- **No breaking changes** - existing functionality preserved
- **Zero downtime** - always falls back gracefully
- **Cost control** - rate limiting prevents unexpected costs
