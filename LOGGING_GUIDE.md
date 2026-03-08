# OpenAI Classification Logging Guide

## Overview

Comprehensive logging has been added to track OpenAI classification usage in the pipeline. All logs are prefixed with `[OpenAI]`, `[Keyword]`, or `[Classification]` for easy filtering.

## Log Categories

### 1. [OpenAI] - OpenAI API Operations

**Initialization:**
- `[OpenAI] OpenAI library not available, using keyword-based classification`
- `[OpenAI] API key not configured, using keyword-based classification`

**Classification Attempts:**
- `[OpenAI] Attempting classification for: <title>...`
- `[OpenAI] API call attempt 1/3 (model: gpt-4o-mini)`
- `[OpenAI] API call successful (attempt 1)`

**Success:**
- `[OpenAI] Classification successful: "<response>" → <genre>`

**Rate Limiting:**
- `[OpenAI] Rate limit check failed, using keyword-based classification`
- `[OpenAI] Token limit check failed (estimated: X tokens), using keyword-based classification`
- `[OpenAI] Rate limit hit, waiting Xs before retry Y/3`
- `[OpenAI] Rate limit exceeded after 3 retries, falling back to keyword-based classification`
- `[OpenAI] Rate limit reached (500 RPM), falling back to keyword-based classification`
- `[OpenAI] Token limit reached (X/50000 TPM), falling back to keyword-based classification`

**Errors:**
- `[OpenAI] Connection error: <error>, falling back to keyword-based classification`
- `[OpenAI] API error: <error>, falling back to keyword-based classification`
- `[OpenAI] Unexpected error: <error>, falling back to keyword-based classification`
- `[OpenAI] Unexpected genre response: '<response>', falling back to keyword-based classification`
- `[OpenAI] Error in classification: <error>, falling back to keyword-based classification`

### 2. [Keyword] - Keyword-based Classification

- `[Keyword] Classification result: <genre>`

### 3. [Classification] - Batch Processing Summary

**Start:**
- `[Classification] Starting genre classification for N videos...`

**Progress (for batches > 50 videos):**
- `[Classification] Progress: X/N videos classified (OpenAI: A, Keyword: B, Special: C)`

**Completion:**
- `[Classification] Complete: X OpenAI, Y keyword-based, Z special types (Live/Scheduled)`
- `[Classification] Summary: X/Y used OpenAI (Z%)`

## Example Pipeline Output

```
[Classification] Starting genre classification for 100 videos...
[OpenAI] Attempting classification for: सड़क दुर्घटना में तीन लोग घायल...
[OpenAI] API call attempt 1/3 (model: gpt-4o-mini)
[OpenAI] API call successful (attempt 1)
[OpenAI] Classification successful: "Traffic" → Traffic
[OpenAI] Attempting classification for: नौकरी सूचना: 100 पदों पर भर्ती...
[OpenAI] API call attempt 1/3 (model: gpt-4o-mini)
[OpenAI] API call successful (attempt 1)
[OpenAI] Classification successful: "Jobs" → Jobs
[OpenAI] Attempting classification for: राजनीतिक रैली में हजारों लोग...
[OpenAI] API call attempt 1/3 (model: gpt-4o-mini)
[OpenAI] API error: Error code: 401 - Invalid API key, falling back to keyword-based classification
[Keyword] Classification result: Politics
[Classification] Progress: 50/100 videos classified (OpenAI: 48, Keyword: 2, Special: 0)
[Classification] Progress: 100/100 videos classified (OpenAI: 95, Keyword: 5, Special: 0)
[Classification] Complete: 95 OpenAI, 5 keyword-based, 0 special types (Live/Scheduled)
[Classification] Summary: 95/100 used OpenAI (95.0%)
```

## Filtering Logs in GitHub Actions

### View only OpenAI logs:
```bash
grep "\[OpenAI\]" output.log
```

### View only classification summary:
```bash
grep "\[Classification\]" output.log
```

### View errors only:
```bash
grep "\[OpenAI\].*error\|\[OpenAI\].*falling back" output.log
```

### View success rate:
```bash
grep "\[Classification\] Summary" output.log
```

## Monitoring in Pipeline

The logs will appear in:
1. **GitHub Actions console output** - Real-time during workflow execution
2. **Workflow logs** - Available in Actions tab for each run
3. **Console output** - When running locally with `python generate_json.py`

## Key Metrics to Monitor

1. **OpenAI Success Rate**: Check `[Classification] Summary` line
2. **Rate Limit Issues**: Look for `Rate limit` messages
3. **API Errors**: Look for `API error` or `Connection error` messages
4. **Fallback Usage**: Compare OpenAI vs keyword-based counts

## Troubleshooting

### High fallback rate:
- Check if `OPENAI_API_KEY` is set correctly
- Verify API key is valid and has credits
- Check rate limit messages

### All using keyword-based:
- `[OpenAI] API key not configured` → Add secret to GitHub
- `[OpenAI] OpenAI library not available` → Check requirements.txt installation

### Rate limit errors:
- Increase `OPENAI_RATE_LIMIT_RPM` and `OPENAI_RATE_LIMIT_TPM` in code
- Or reduce number of videos processed per run
