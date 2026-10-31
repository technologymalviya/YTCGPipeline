#!/usr/bin/env python3
"""
YT Aggregator for Bhilai News - GitHub Actions Pipeline Version
Fetches latest videos from Bhilai news channels and generates JSON feed.
"""

import json
import math
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import requests

# OpenAI integration (optional)
try:
    import openai
    # Verify the module has the required OpenAI class
    if hasattr(openai, 'OpenAI'):
        OPENAI_AVAILABLE = True
    else:
        OPENAI_AVAILABLE = False
        openai = None
        print("[OpenAI] OpenAI module imported but missing OpenAI class")
except ImportError as e:
    OPENAI_AVAILABLE = False
    openai = None
    print(f"[OpenAI] OpenAI library import failed: {e}")
except Exception as e:
    OPENAI_AVAILABLE = False
    openai = None
    print(f"[OpenAI] Unexpected error importing OpenAI: {e}")


# Configuration Constants
MAX_RESULTS = 20
LAST_HOURS_FILTER = 24
REQUEST_TIMEOUT = 20

# File Constants
OUTPUT_FILE_NAME = "output.json"

# YouTube API Configuration
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# Environment Variable Names
ENV_YOUTUBE_API_KEY = "YOUTUBE_API_KEY"
ENV_BHILAI_CHANNELS = "BHILAI_CHANNELS"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_REQUEST_IP = "REQUEST_IP"
ENV_IP_LOOKUP_URL = "IP_LOOKUP_URL"

# OpenAI Configuration
OPENAI_MODEL = "gpt-4o-mini"  # Cost-effective model
OPENAI_MAX_TOKENS = 10  # Just need genre name
OPENAI_TEMPERATURE = 0.1  # Low temperature for consistent classification
OPENAI_RATE_LIMIT_RPM = 500  # Requests per minute limit
OPENAI_RATE_LIMIT_TPM = 50000  # Tokens per minute limit
OPENAI_MAX_RETRIES = 1  # Reduced to 1 retry for faster execution
OPENAI_RETRY_DELAY_BASE = 0.5  # Reduced delay for faster execution
OPENAI_REQUEST_TIMEOUT = 5  # Reduced timeout to 5 seconds per request
OPENAI_SAFETY_MARGIN_RPM = 0.8  # Use only 80% of rate limit to avoid hitting threshold
OPENAI_SAFETY_MARGIN_TPM = 0.8  # Use only 80% of token limit to avoid hitting threshold
OPENAI_MAX_EXECUTION_TIME = 50  # Maximum time in seconds for OpenAI classification (leave 10s buffer for other operations)

# IP/City lookup configuration (city section)
IP_LOOKUP_URL_DEFAULT = "http://ip-api.com/json/{ip}"
IP_LOOKUP_TIMEOUT = 3
IP_LOOKUP_CACHE_TTL_SECONDS = 300
IP_LOOKUP_CACHE_MAX_SIZE = 5000

# Rate limiting tracking
_openai_rate_limit_tracker = {
    "requests": [],
    "tokens": [],
    "last_reset": time.time()
}

# Lightweight in-memory cache for IP->city resolution.
# Helps under heavy request volume and protects upstream lookup service.
_ip_city_cache: Dict[str, Dict[str, object]] = {}


def _log_city(message: str):
    """Structured logger for city/IP-section flow."""
    print(f"[CitySection] {message}")


def _prune_ip_city_cache():
    """Prune expired and excess cache entries."""
    now = time.time()
    expired = [k for k, v in _ip_city_cache.items() if now - float(v.get("ts", 0)) > IP_LOOKUP_CACHE_TTL_SECONDS]
    for k in expired:
        _ip_city_cache.pop(k, None)
    if expired:
        _log_city(f"Pruned {len(expired)} expired IP cache entries")

    if len(_ip_city_cache) > IP_LOOKUP_CACHE_MAX_SIZE:
        # Drop oldest entries first.
        sorted_items = sorted(_ip_city_cache.items(), key=lambda kv: float(kv[1].get("ts", 0)))
        overflow = len(_ip_city_cache) - IP_LOOKUP_CACHE_MAX_SIZE
        for k, _ in sorted_items[:overflow]:
            _ip_city_cache.pop(k, None)
        _log_city(f"Pruned {overflow} overflow IP cache entries (max={IP_LOOKUP_CACHE_MAX_SIZE})")

# Error Messages
MSG_API_KEY_NOT_SET = "WARNING: YOUTUBE_API_KEY not set. Not updating output file."
MSG_CHANNELS_NOT_SET = "WARNING: BHILAI_CHANNELS not set. Not updating output file."
MSG_QUOTA_EXCEEDED = "[QUOTA EXCEEDED] API key #{} quota exceeded"
MSG_ALL_KEYS_EXHAUSTED = "[ERROR] All API keys have exceeded quota"
MSG_SWITCHING_KEY = "[INFO] Switching to API key #{}"
MSG_INVALID_API_KEY = "[ERROR] HTTP 400: Invalid or wrong API key (key #{})"
MSG_CHECK_API_KEY = "Please check that the YouTube API key is correct and has YouTube Data API v3 enabled."

# Log Messages
MSG_NO_VIDEOS = "WARNING: No videos fetched. This may indicate all API keys are exhausted."
MSG_PRESERVE_DATA = "Not updating output.json to preserve existing data."
MSG_KEEP_EXISTING = "Keeping existing output.json file (if present)"
MSG_SET_CHANNELS = "Please set the BHILAI_CHANNELS environment variable with comma-separated channel IDs."

# Error Types
ERR_QUOTA_REASON = "quotaExceeded"
ERR_QUOTA_KEYWORD = "quota"
ERR_API_KEY_NOT_CONFIGURED = "YouTube API key not configured"
ERR_CHANNELS_NOT_CONFIGURED = "BHILAI_CHANNELS not configured"

# Video Types
VIDEO_TYPE_LIVE = "LIVE"
VIDEO_TYPE_VOD = "VOD"
VIDEO_TYPE_SCHEDULED = "SCHEDULED"
VIDEO_TYPE_UNKNOWN = "UNKNOWN"

# Genre Constants
GENRE_LIVE = "Live"
GENRE_SCHEDULED = "Scheduled"
GENRE_CRIME = "Crime"
GENRE_TRAFFIC = "Traffic"
GENRE_POLITICS = "Politics"
GENRE_JOBS = "Jobs"
GENRE_EVENTS = "Events"
GENRE_CIVIC = "Civic"
GENRE_PODCAST = "Podcast"
GENRE_GENERAL = "General"

# Every upload from these channels goes in the Podcast row, not only titles that say "podcast".
PODCAST_ALLOWED_CHANNEL_TITLES = [
    "Chhattisgarh Podcast",
    "Z-Series CG Podcast",
    "The Lok Ras",
    "The PS Show",
]

# Section Names
SECTION_LIVE = "Live"
SECTION_SCHEDULED = "Scheduled"
SECTION_CRIME = "Crime"
SECTION_TRAFFIC = "Traffic"
SECTION_NEGATIVE = "Negative"
SECTION_POLITICS = "Politics"
SECTION_JOBS = "Jobs"
SECTION_EVENTS = "Events"
SECTION_CIVIC = "Civic"
SECTION_PODCAST = "Podcast"
SECTION_GENERAL = "General"

# Section -> Index map (per requested mapping)
# Mapping: "General" 1, "Podcast" 2, "Jobs" 3, "Politics" 4, "Events" 5, "Civic" 6, "Negative" 7
# Negative is the single tab/rail for Crime and Traffic.
SECTION_INDEX = {
    "General": 1,
    "Podcast": 2,
    "Jobs": 3,
    "Politics": 4,
    "Events": 5,
    "Civic": 6,
    "Negative": 7,
}

# Load multiple API keys for fallback support
def load_api_keys():
    """Load all available YouTube API keys from environment variables."""
    keys = []
    # Check for primary key
    primary_key = os.environ.get(ENV_YOUTUBE_API_KEY, "")
    if primary_key:
        keys.append(primary_key)
    
    # Check for additional keys (YOUTUBE_API_KEY_2, YOUTUBE_API_KEY_3, etc.)
    index = 2
    while True:
        key = os.environ.get(f"{ENV_YOUTUBE_API_KEY}_{index}", "")
        if key:
            keys.append(key)
            index += 1
        else:
            break
    
    return keys

API_KEYS = load_api_keys()

# Current API key index for fallback mechanism
current_api_key_index = 0

def get_current_api_key():
    """Get the current active API key."""
    global current_api_key_index
    if not API_KEYS:
        return ""
    if current_api_key_index >= len(API_KEYS):
        return ""
    return API_KEYS[current_api_key_index]

def switch_to_next_api_key():
    """Switch to the next available API key."""
    global current_api_key_index
    current_api_key_index += 1
    if current_api_key_index < len(API_KEYS):
        print(MSG_SWITCHING_KEY.format(current_api_key_index + 1))
        return True
    return False

def is_quota_exceeded_error(response):
    """Check if the error is a quota exceeded error."""
    if response.status_code == 403:
        try:
            error_data = response.json()
            error_obj = error_data.get("error", {})
            
            # Check error reason in the errors array
            errors = error_obj.get("errors", [])
            for error in errors:
                if error.get("reason") == ERR_QUOTA_REASON:
                    return True
            
            # Also check message as fallback
            message = error_obj.get("message", "")
            if ERR_QUOTA_KEYWORD in message.lower():
                return True
                
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Failed to parse JSON or extract error message
            pass
    return False

def is_invalid_api_key_error(response):
    """Check if the error is an invalid API key error (HTTP 400)."""
    if response.status_code == 400:
        try:
            error_data = response.json()
            error_obj = error_data.get("error", {})
            
            # Check error reason in the errors array
            errors = error_obj.get("errors", [])
            for error in errors:
                reason = error.get("reason", "")
                # Common reasons for invalid API key in HTTP 400
                if reason in ("keyInvalid", "badRequest"):
                    return True
            
            # Also check message for API key related errors
            message = error_obj.get("message", "").lower()
            if "api key" in message or ("invalid" in message and "key" in message):
                return True
                
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Failed to parse JSON or extract error message
            pass
    return False

def load_bhilai_channels():
    """Load channel IDs from environment variable."""
    channels_env = os.environ.get(ENV_BHILAI_CHANNELS, "")
    if channels_env:
        # Parse comma-separated channel IDs from environment variable
        channels = [ch.strip() for ch in channels_env.split(",") if ch.strip()]
        if channels:
            return channels
    # Return empty list if no channels configured
    return []

BHILAI_CHANNELS = load_bhilai_channels()


def _safe_int(x, default=0) -> int:
    """Safely convert to int."""
    try:
        return int(x)
    except Exception:
        return default


def parse_iso_duration(duration: str) -> int:
    """Parse ISO 8601 duration to seconds."""
    if not duration or not duration.startswith("PT"):
        return 0
    
    pattern = re.compile(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?')
    match = pattern.match(duration)
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds


def format_duration(seconds: int) -> str:
    """Format seconds to HH:MM:SS or MM:SS."""
    if seconds is None:
        return "0"
    
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def normalize(text: str) -> str:
    """Normalize text for genre classification."""
    if not text:
        return ""

    text = text.lower()
    # Keep spaces, word characters, and Devanagari characters
    text = re.sub(r"[^\w\s\u0900-\u097F]", "", text)
    # Normalize multiple spaces to single space
    text = re.sub(r"\s+", " ", text).strip()
    
    replacements = {
        "h!dsa": "हादसा",
        "ha*dsa": "हादसा",
        "h@dsa": "हादसा",
        "mou.t": "मौत",
        "mou t": "मौत",
        "la sh": "लाश",
        "la!sh": "लाश",
        "murd r": "murder",
        "murdर": "murder",
        "रफ्त!र": "रफ्तार",
        "d!rt": "दुर्घटना",
        "durghatna": "दुर्घटना",
        "hadsa": "हादसा",
        "mout": "मौत",
        " hatya": " हत्या",
    }
    
    # Use word boundaries for replacements to avoid breaking words like "clash" -> "cलाश"
    for k, v in replacements.items():
        # Use word boundary regex for English words, simple replace for Hindi/Devanagari
        if re.search(r'[a-zA-Z]', k):
            # English word - use word boundary
            pattern = r'\b' + re.escape(k) + r'\b'
            text = re.sub(pattern, v, text)
        else:
            # Hindi/Devanagari or special chars - simple replace
            text = text.replace(k, v)
    
    return text


def _is_allowed_podcast_channel(channel_title: str) -> bool:
    """True for dedicated podcast channels. All of their uploads belong in the Podcast row."""
    if not channel_title:
        return False
    raw = channel_title.strip()
    folded = raw.lower()
    allowed = {name.lower() for name in PODCAST_ALLOWED_CHANNEL_TITLES}
    if folded in allowed:
        return True
    return "podcast" in folded or "पॉडकास्ट" in raw or "पोडकास्ट" in raw


def _looks_like_podcast_content(title: str, description: str = "") -> bool:
    """True when the title or description itself is a podcast episode."""
    title_n = normalize(title or "")
    desc_n = normalize(description or "")
    text = f"{title_n} {desc_n}"
    needles = (
        "podcast",
        "podcasts",
        "पॉडकास्ट",
        "पोडकास्ट",
        "36garhpodcast",
        "36 garh podcast",
        "cgpodcast",
        "cg podcast",
        "z-series",
        "z series",
        "zseries",
    )
    return any(n in text for n in needles)


def _is_podcast_video(title: str, description: str = "", channel_title: str = "") -> bool:
    if _is_allowed_podcast_channel(channel_title):
        return True
    return _looks_like_podcast_content(title, description)


# Create word boundary patterns for better matching
def create_patterns(keywords):
    """Create regex patterns with word boundaries for better matching."""
    patterns = []
    for kw in keywords:
        # Use word boundaries for English, space boundaries for Hindi
        if re.search(r'[\u0900-\u097F]', kw):
            # Hindi/Devanagari - use space or start/end of string, also allow punctuation (।, ,, .)
            pattern = r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$|।|,|\.|!|\?)'
        else:
            # English - use word boundaries
            pattern = r'\b' + re.escape(kw) + r'\b'
        patterns.append(re.compile(pattern, re.IGNORECASE))
    return patterns


def classify_genre_with_openai(title: str, description: str = "") -> Optional[str]:
    """
    Classify video genre using OpenAI API with rate limiting and error handling.
    Returns None if OpenAI is unavailable or encounters errors (fallback to keyword-based).
    """
    if not OPENAI_AVAILABLE:
        # Only log once per session to avoid spam
        if not hasattr(classify_genre_with_openai, '_logged_unavailable'):
            print("[OpenAI] OpenAI library not available, using keyword-based classification")
            print("[OpenAI] To fix: pip install openai>=1.0.0 (or check import errors above)")
            classify_genre_with_openai._logged_unavailable = True
        return None
    
    # Check if API key is configured
    api_key = os.environ.get(ENV_OPENAI_API_KEY)
    if not api_key:
        # Only log once per session to avoid spam
        if not hasattr(classify_genre_with_openai, '_logged_no_key'):
            print("[OpenAI] API key not configured, using keyword-based classification")
            print("[OpenAI] To fix: Set OPENAI_API_KEY environment variable or GitHub secret")
            classify_genre_with_openai._logged_no_key = True
        return None
    
    print(f"[OpenAI] Attempting classification for: {title[:60]}...")
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key, timeout=OPENAI_REQUEST_TIMEOUT)
        
        # Check rate limits
        if not _check_openai_rate_limits():
            print("[OpenAI] Rate limit check failed, using keyword-based classification")
            return None
        
        # Prepare prompt
        prompt = f"""Classify this video into ONE of these genres based on title and description:
- Crime: Criminal cases and any negative news: deaths, disasters, crop loss, blasts, raids, arrests, fraud, missing people, violence, allegations of harm. If the story is bad news, choose Crime even when a politician is mentioned.
- Traffic: Road accidents, traffic jams, vehicle collisions, highway incidents only
- Jobs: Only hiring/recruitment/vacancy/admit card/merit lists/government job exams — NOT school board exam human-interest, NOT investigative stories about doctors' private practice or staff shortages
- Events: Festivals, ceremonies, celebrations, inaugurations, cultural or sports events (not crime controversies, not IPL betting)
- Civic: Municipal services, civic issues, government services, certificates, utilities
- Politics: Elections, parties, legislation, schemes, CM/PM/minister policy — "MP" in Indian regional titles usually means Madhya Pradesh state, not Member of Parliament
- General: Everything else that doesn't fit above categories

Title: {title}
Description: {description[:500]}

Respond with ONLY the genre name (Crime, Traffic, Jobs, Events, Civic, Politics, or General)."""
        
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        estimated_tokens = len(prompt) // 4 + OPENAI_MAX_TOKENS
        
        # Check token rate limit
        if not _check_openai_token_limit(estimated_tokens):
            print(f"[OpenAI] Token limit check failed (estimated: {estimated_tokens} tokens), using keyword-based classification")
            return None
        
        # Make API call with minimal retry logic (fast fail)
        response = None
        for attempt in range(OPENAI_MAX_RETRIES + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    print(f"[OpenAI] API call retry {attempt}/{OPENAI_MAX_RETRIES} (model: {OPENAI_MODEL})")
                else:
                    print(f"[OpenAI] API call (model: {OPENAI_MODEL})")
                
                # Use shorter timeout for faster execution
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a news genre classifier. Respond with only the genre name."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=OPENAI_MAX_TOKENS,
                    temperature=OPENAI_TEMPERATURE,
                    timeout=OPENAI_REQUEST_TIMEOUT,
                )
                if attempt > 0:
                    print(f"[OpenAI] API call successful (retry {attempt})")
                break
            except openai.RateLimitError as e:
                # Rate limit hit - fail fast, don't retry
                print(f"[OpenAI] Rate limit hit, falling back to keyword-based classification immediately")
                return None
            except openai.AuthenticationError as e:
                # Invalid API key (401 error) - don't expose the key in logs
                if not hasattr(classify_genre_with_openai, '_logged_invalid_key'):
                    # Try to extract error details without exposing API key
                    error_code = 'invalid_api_key'
                    error_type = 'authentication_error'
                    
                    # Check if error has response with JSON
                    if hasattr(e, 'response') and hasattr(e.response, 'json'):
                        try:
                            error_data = e.response.json()
                            error_obj = error_data.get('error', {})
                            error_code = error_obj.get('code', 'invalid_api_key')
                            error_type = error_obj.get('type', 'authentication_error')
                        except:
                            pass
                    
                    # Sanitize error message - remove any API key patterns
                    error_msg = str(e)
                    # Remove API key patterns (sk-proj-, sk-, etc.)
                    error_msg = re.sub(r'sk-[a-zA-Z0-9_-]+', 'sk-***', error_msg)
                    error_msg = re.sub(r'[a-zA-Z0-9]{20,}', '***', error_msg)  # Remove long alphanumeric strings
                    
                    print(f"[OpenAI] Invalid API key (error: {error_code}), falling back to keyword-based classification")
                    print("[OpenAI] To fix: Check OPENAI_API_KEY environment variable or GitHub secret")
                    print(f"[OpenAI] Error details: {error_type}")
                    classify_genre_with_openai._logged_invalid_key = True
                return None
            except openai.APIConnectionError as e:
                if attempt < OPENAI_MAX_RETRIES:
                    wait_time = OPENAI_RETRY_DELAY_BASE
                    print(f"[OpenAI] Connection error, waiting {wait_time}s before retry {attempt + 1}/{OPENAI_MAX_RETRIES}")
                    time.sleep(wait_time)
                else:
                    print(f"[OpenAI] Connection error after {OPENAI_MAX_RETRIES} retries: {e}, falling back to keyword-based classification")
                    return None
            except openai.APIError as e:
                # Check if it's a 401 error (invalid API key) even if not AuthenticationError
                error_msg = str(e)
                # Sanitize error message - remove any API key patterns
                sanitized_msg = re.sub(r'sk-[a-zA-Z0-9_-]+', 'sk-***', error_msg)
                sanitized_msg = re.sub(r'[a-zA-Z0-9]{20,}', '***', sanitized_msg)
                
                if '401' in error_msg or 'invalid_api_key' in error_msg.lower() or 'incorrect api key' in error_msg.lower():
                    if not hasattr(classify_genre_with_openai, '_logged_invalid_key'):
                        print("[OpenAI] Invalid API key (401 error), falling back to keyword-based classification")
                        print("[OpenAI] To fix: Check OPENAI_API_KEY environment variable or GitHub secret")
                        classify_genre_with_openai._logged_invalid_key = True
                    return None
                else:
                    print(f"[OpenAI] API error: {sanitized_msg}, falling back to keyword-based classification")
                    return None
            except Exception as e:
                error_msg = str(e)
                # Sanitize error message - remove any API key patterns
                sanitized_msg = re.sub(r'sk-[a-zA-Z0-9_-]+', 'sk-***', error_msg)
                sanitized_msg = re.sub(r'[a-zA-Z0-9]{20,}', '***', sanitized_msg)
                
                # Check for API key in error message and sanitize
                if 'api key' in error_msg.lower() or '401' in error_msg or 'invalid_api_key' in error_msg.lower():
                    if not hasattr(classify_genre_with_openai, '_logged_invalid_key'):
                        print("[OpenAI] Invalid API key detected, falling back to keyword-based classification")
                        print("[OpenAI] To fix: Check OPENAI_API_KEY environment variable or GitHub secret")
                        classify_genre_with_openai._logged_invalid_key = True
                    return None
                else:
                    print(f"[OpenAI] Unexpected error: {sanitized_msg}, falling back to keyword-based classification")
                    return None
        
        if not response:
            return None
        
        # Extract genre from response
        genre = response.choices[0].message.content.strip()
        
        # Track usage
        _track_openai_usage(estimated_tokens)
        
        # Validate and map genre
        genre_map = {
            "crime": GENRE_CRIME,
            "traffic": GENRE_TRAFFIC,
            "jobs": GENRE_JOBS,
            "events": GENRE_EVENTS,
            "civic": GENRE_CIVIC,
            "politics": GENRE_POLITICS,
            "general": GENRE_GENERAL,
        }
        
        genre_lower = genre.lower()
        if genre_lower in genre_map:
            mapped_genre = genre_map[genre_lower]
            print(f"[OpenAI] Classification successful: '{genre}' → {mapped_genre}")
            return mapped_genre
        
        # If response doesn't match expected genres, fallback
        print(f"[OpenAI] Unexpected genre response: '{genre}', falling back to keyword-based classification")
        return None
        
    except Exception as e:
        print(f"[OpenAI] Error in classification: {e}, falling back to keyword-based classification")
        return None


def _check_openai_rate_limits() -> bool:
    """Check if we're within rate limits for requests per minute (with safety margin)."""
    current_time = time.time()
    
    # Reset tracking if a minute has passed
    if current_time - _openai_rate_limit_tracker["last_reset"] >= 60:
        _openai_rate_limit_tracker["requests"] = []
        _openai_rate_limit_tracker["tokens"] = []
        _openai_rate_limit_tracker["last_reset"] = current_time
    
    # Check requests per minute with safety margin
    recent_requests = [
        req_time for req_time in _openai_rate_limit_tracker["requests"]
        if current_time - req_time < 60
    ]
    
    safe_limit = int(OPENAI_RATE_LIMIT_RPM * OPENAI_SAFETY_MARGIN_RPM)
    if len(recent_requests) >= safe_limit:
        print(f"[OpenAI] Rate limit safety threshold reached ({len(recent_requests)}/{safe_limit} RPM), falling back to keyword-based classification")
        return False
    
    return True


def _check_openai_token_limit(estimated_tokens: int) -> bool:
    """Check if we're within token rate limits (with safety margin)."""
    current_time = time.time()
    
    # Reset tracking if a minute has passed
    if current_time - _openai_rate_limit_tracker["last_reset"] >= 60:
        _openai_rate_limit_tracker["requests"] = []
        _openai_rate_limit_tracker["tokens"] = []
        _openai_rate_limit_tracker["last_reset"] = current_time
    
    # Check tokens per minute with safety margin
    recent_tokens = [
        token_count for token_count in _openai_rate_limit_tracker["tokens"]
        if current_time - token_count["time"] < 60
    ]
    
    total_tokens = sum(token_count["tokens"] for token_count in recent_tokens)
    safe_limit = int(OPENAI_RATE_LIMIT_TPM * OPENAI_SAFETY_MARGIN_TPM)
    
    if total_tokens + estimated_tokens > safe_limit:
        print(f"[OpenAI] Token limit safety threshold reached ({total_tokens + estimated_tokens}/{safe_limit} TPM), falling back to keyword-based classification")
        return False
    
    return True


def _track_openai_usage(tokens: int):
    """Track OpenAI API usage for rate limiting."""
    current_time = time.time()
    _openai_rate_limit_tracker["requests"].append(current_time)
    _openai_rate_limit_tracker["tokens"].append({"time": current_time, "tokens": tokens})


def is_recent_video(published_at: str, hours: int = 1) -> bool:
    """
    Check if a video was published within the last N hours.
    
    Args:
        published_at: ISO8601 formatted datetime string (e.g., "2024-01-01T12:00:00Z")
        hours: Number of hours to check (default: 1)
    
    Returns:
        True if video was published within the last N hours, False otherwise
    """
    if not published_at:
        return False
    
    try:
        # Parse ISO8601 format (handles both Z and +00:00 formats)
        published_at_clean = published_at.replace('Z', '+00:00')
        published_time = datetime.fromisoformat(published_at_clean)
        if published_time.tzinfo is None:
            published_time = published_time.replace(tzinfo=timezone.utc)
        
        # Get current time in UTC
        now = datetime.now(timezone.utc)
        
        # Check if published within last N hours (include videos up to N hours old)
        time_diff = now - published_time
        # Use <= with small buffer to account for precision and include videos exactly N hours old
        return time_diff <= timedelta(hours=hours, seconds=1)
    except (ValueError, AttributeError) as e:
        # If parsing fails, assume not recent (fallback to keyword-based)
        print(f"[Classification] Error parsing publishedAt '{published_at}': {e}, using keyword-based classification")
        return False


def classify_genre(title: str, description: str = "", published_at: str = None, use_openai: bool = True) -> str:
    """
    Classify video genre based on title and description.
    
    Args:
        title: Video title
        description: Video description
        published_at: ISO8601 formatted datetime string (optional)
        use_openai: Whether to try OpenAI (default: True). If False, uses keyword-based directly.
    
    Returns:
        Genre classification string
    """
    # Only use OpenAI if explicitly enabled and video is recent (within last 1 hour)
    if use_openai and published_at and is_recent_video(published_at, hours=1):
        openai_result = classify_genre_with_openai(title, description)
        if openai_result:
            scores = semantic_context_scores(title, description)
            return apply_negative_crime_context(openai_result, scores)
    
    # Use keyword-based classification for older videos or if OpenAI fails
    keyword_result = classify_genre_keyword_based(title, description)
    if not use_openai or not published_at or not is_recent_video(published_at, hours=1):
        # Only print for non-recent videos to avoid spam
        print(f"[Keyword] Classification result: {keyword_result}")
    return keyword_result


def _classify_genre_by_keywords(title: str, description: str = "") -> str:
    """Keyword rules only. Podcast and negative-news context are applied by the public wrapper."""
    # Normalize title and description separately - title gets more weight
    title_text = normalize(title or "")
    desc_text = normalize(description or "")
    
    # Combine with title weighted 3x more than description (titles are more indicative)
    text = f"{title_text} {title_text} {title_text} {desc_text}"
    
    # Crime keywords - specific and strong indicators
    crime_keywords = [
        # English - specific crime terms
        "murder", "killed", "killing", "homicide", "assassination", "manslaughter",
        "crime",  # e.g. "MP Crime" segments (avoid false Politics on standalone "MP")
        "robbery", "loot", "theft", "steal", "stolen", "burglary", "break-in", "heist", "mugging",
        "assault", "rape", "sexual assault", "sexual violence",
        "arrest", "arrested", "detained", "custody", "suspect", "accused", "criminal",
        "gang", "cartel", "mafia", "mob", "violence", "rioting", "clash", "clashes",  # Clashes/violence
        "shoot", "shot", "firing", "gunfire", "bullets",
        "stab", "stabbing", "knife attack", "weapon", "knife fight",
        "jail", "prison", "court case", "trial", "investigation",
        "kidnapping", "abduction", "abducted", "hostage", "hijack", "hijacking", "extortion", "ransom",
        "naxal attack", "naxal violence", "naxalite attack", "naxal surrender",  # Naxal-related crimes (only when crime-related)
        # Negative/death headlines should not remain General
        "passes away", "passed away", "no more", "death news", "death update",
        "last rites", "funeral", "obituary",
        "सट्टा", "satta", "betting", "match fixing", "ipl betting", "आईपीएल सट्टा", "illegal betting",
        "gas cylinder", "cylinder blast", "explode", "explosion",  # Explosions
        "ied blast", "ied", "bomb blast", "bomb explosion",  # IED/Bomb explosions
        "fire incident", "arson",  # Fire-related crimes
        "chain snatching", "snatching",  # Chain snatching
        "fraud case", "scam case", "cheat", "swindle", "dupe", "forgery", "money laundering",  # Fraud/scam (only when clearly crime, not political discussions)
        "corruption scam", "financial scam", "bank scam",  # Specific scams
        "cyber fraud", "cyber theft", "cyber crime", "online scam", "telegram scam",  # Cyber crimes
        "dog attack", "dog bite", "stray dog",  # Dog attacks
        "monkey attack", "leopard attack", "animal attack", "wild animal attack",  # Animal attacks
        "food poisoning", "death from food", "contaminated food",  # Food-related deaths
        "mid-day meal", "midday meal", "school meal",  # Mid-day meal food poisoning
        "harassment", "eve teasing", "molestation", "sexual harassment", "abuse", "domestic violence",  # Harassment
        "stunt", "illegal stunt", "dangerous stunt", "stunt video",  # Illegal stunts
        "it raid", "income tax raid", "raid", "search",  # IT raids and searches
        "acb raid", "anti corruption bureau", "acb",  # ACB raids
        "food department raid", "food dept raid",  # Food department raids
        # Hindi/Devanagari - specific crime terms
        "हत्या","कत्ल", "लाश", "शव", "अपराध", "अपराधी", "हत्याकांड",
        # "मौत" only when combined with crime terms (not medical deaths)
        "हत्या में मौत", "कत्ल में मौत", "अपराध में मौत", "गोली से मौत", "चाकू से मौत",
        # Medical negligence death (crime)
        "इलाज के दौरान मरीज की मौत", "इलाज के दौरान मौत", "डॉक्टर लापरवाही", "मेडिकल नेग्लिजेंस",
        "कातिल", "जेल", "कारागार", "अदालत", "जज", "मुकदमा", "क्राइम", "मौत",
        "निधन", "निधन की खबर", "अंतिम संस्कार", "अंतिम विदाई", "श्रद्धांजलि",
        "crime branch", "क्राइम ब्रांच",
        "missile", "मिसाइल", "drone strike", "air strike",
        "पुलिस गिरफ्तार", "पुलिस ने गिरफ्तार", "पुलिस जांच",  # Police action (crime-related)
        "गिरोह", "संगठित गिरोह", "गिरोह का पर्दाफाश",  # Gang/exposure
        "पर्दाफाश", "किया पर्दाफाश",  # Exposure/revelation (crime-related)
        "खाद्य विभाग का छापा", "एसीबी की दबिश", "एसीबी रेड",  # ACB/Food dept raids
        "दबिश",  # Raid (crime-related)
        "हिंसा", "दंगा", "चाकू", "चाकूबाजी", "बंदूक", "गोली", "फायरिंग",  # Violence/knife attack
        "हंगामा", "हंगामा किया", "उपद्रव",  # Riot/commotion
        "अपहरण",  # Kidnapping
        "लापता", "missing", "फरार", "भाग गई", "भाग गया", "विवाद", "controversy",
        "नक्सली हमला", "नक्सल हिंसा", "नक्सल सरेंडर",  # Naxal-related crimes (only when crime-related)
        "सिलेंडर ब्लास्ट", "सिलेंडर विस्फोट", "विस्फोट", "आईईडी ब्लास्ट", "बम ब्लास्ट",  # Explosions
        # Fire: avoid bare "आग" (matches rice धान burning, generic fire news) — use stronger phrases
        "अग्निकांड", "भीषण आग", "आग लगी", "लगी आग", "आग से",
        "चेन छीन", "चेन स्नैचिंग", "स्नैचिंग",  # Chain snatching
        "ठग", "ठगी", "धोखाधड़ी", "फ्रॉड", "जालसाजी", "उगाही", "रंगदारी",  # Fraud/scam
        "स्कैम केस", "धोखाधड़ी केस", "भ्रष्टाचार स्कैम",  # Specific scams (not political discussions)
        "सायबर चोरी", "सायबर फ्रॉड", "ऑनलाइन स्कैम", "टेलीग्राम स्कैम",  # Cyber crimes
        "कुत्ते का आतंक", "कुत्ते ने काटा", "कुत्ता काटा", "कुत्ता",  # Dog attack
        "बंदर", "तेंदुआ", "तेंदुए", "तेंदुए ने", "जंगली जानवर", "जानवर का हमला", "जानवर ने काटा", "शिकार",  # Animal attacks
        "मिठाई से मौत", "खाने से मौत", "जहर",  # Food-related deaths
        "मध्यान भोजन", "मिड डे मील", "स्कूल भोजन",  # Mid-day meal
        "छेड़छाड़", "परेशानी", "यौन उत्पीड़न", "मनचले",  # Harassment
        "स्टंट", "अवैध स्टंट", "खतरनाक स्टंट", "स्टंट का अड्डा",  # Illegal stunts
        "दुष्कर्म", "बलात्कार", "मारपीट", "हमला",  # Rape/sexual assault
        "आईटी रेड", "आयकर रेड", "रेड", "छापा",  # IT raids (दबिश listed above)
        # Context-specific: जांच only when combined with crime terms
        "अपराध जांच", "पुलिस जांच", "हत्या जांच", "मुकदमा जांच"
    ]

    # Traffic keywords - specific traffic/accident terms
    traffic_keywords = [
        # English - multi-word phrases for stronger matching (check first)
        "traffic accident", "road accident", "car accident", "vehicle accident", "motorcycle accident", "bike accident",
        "road crash", "car crash", "highway crash", "train crash", "bus crash", "truck crash",
        "traffic jam", "road jam", "gridlock",
        "head-on collision", "rear-end collision", "pile-up", "multi-vehicle collision",
        "fatal accident", "deadly accident", "hit-and-run", "run-over",
        "vehicle overturned", "truck overturned", "bus overturned", "tractor overturned",  # Vehicle overturning
        "thar accident", "speed havoc", "road rage", "drunk driving", "dui",  # Specific vehicle/accident types
        # Hindi/Devanagari - multi-word phrases (check first)
        "सड़क दुर्घटना", "कार हादसा", "वाहन हादसा", "बाइक हादसा", "मोटरसाइकिल हादसा",
        "ट्रैक्टर पलटी", "ट्रॉली पलटी", "गाड़ी पलटी", "वाहन पलटी",  # Vehicle overturned
        "रफ्तार का कहर", "थार का कहर",  # Speed/Thar vehicle accident
        "एनएच", "NH-", "हाइवे पर",  # Highway accidents
        "ट्रैफिक जाम", "सड़क जाम", "लंबा जाम", "भीषण जाम",
        # English - strong single-word indicators (context-specific)
        # Note: standalone "crash" omitted — matches gold/stock "price crash" (false Traffic)
        "accident", "collision", "traffic", "jam", "congestion",
        "ambulance", "rescue", "highway", "expressway",
        "bus accident", "train accident", "rail accident", "derailed", "derailment",
        "emergency landing", "runway", "aircraft",
        # Hindi/Devanagari - strong single-word indicators (context-specific)
        "दुर्घटना", "हादसा", "टक्कर", "जाम", "एम्बुलेंस", "पलटी", "यातायात",  # पलटी = overturned
        "बस हादसा", "ट्रेन हादसा", "रेल हादसा", "रेलवे", "ट्रेन से गिरा", "चलती ट्रेन",
        "train से गिरा", "चलती train", "from train",
        "इमरजेंसी लैंडिंग", "आपात लैंडिंग", "विमान की लैंडिंग",
        "हिट एंड रन", "हिट-एंड-रन", "ओवरस्पीड", "ओवर स्पीड", "नशे में ड्राइविंग"
        # Note: "घायल" (injured) removed - too generic, appears in non-traffic contexts
    ]
    
    # Jobs keywords - specific employment/job terms
    jobs_keywords = [
        # English - strong single-word indicators
        "job", "jobs", "recruitment", "vacancy", "hiring", "employment",
        # "bharti"/"bharte" omitted — false positives (e.g. "Bharti Media"); use "police bharti" etc.
        "career", "opportunity", "application", "opening",
        "job post", "govt post", "government post", "sarkari post",
        "admit card", "merit", "salary", "wage",
        # Omit bare "exam"/"परीक्षा" — false Jobs on school board exam human-interest clips
        "police recruitment", "police bharti",  # Police recruitment (specific)
        "appointment", "appointments", "नियुक्ति",  # Appointments
        # English - multi-word phrases for stronger matching
        "job notification", "job vacancy", "job opening", "job opportunity",
        "government job", "sarkari naukri", "govt job",
        "job application", "apply for job",
        "exam result", "merit list", "job result",  # Job/exam results (not election results)
        "walk-in interview", "job interview", "employment interview",
        "pay scale", "vacancy notification",
        # Hindi/Devanagari - strong single-word indicators
        "नौकरी", "रोजगार", "भर्ती", "आवेदन", "एडमिट कार्ड",
        "मेरिट", "वेतन",
        # Vacancy spelled in Devanagari (English "vacancy" does not match "वैकेंसी")
        "वैकेंसी", "वेकेंसी", "बंपर वैकेंसी",
        "रिक्त पद", "रिक्तियां", "रिक्तियाँ", "पदों पर निकली",
        "पुलिस भर्ती",  # Police recruitment (specific)
        "नियुक्ति",  # Appointments
        # Hindi/Devanagari - multi-word phrases (context-specific)
        "सरकारी नौकरी", "नौकरी सूचना",
        "नौकरी इंटरव्यू", "भर्ती इंटरव्यू", "रोजगार इंटरव्यू",
        "निगम मंडल", "मंडल में नियुक्ति",  # Corporation board appointments
        "परीक्षा रिजल्ट", "भर्ती रिजल्ट", "नौकरी रिजल्ट", "मेरिट रिजल्ट"  # Job/exam results (not election)
    ]

    # Events keywords - specific event/festival terms
    events_keywords = [
        # English - strong single-word indicators
        "event", "events", "festival", "celebration", "ceremony", "fair", "mela", "exhibition",
        "inauguration", "wedding", "anniversary", "birthday",
        "kumbh", "jatra", "yatra", "puja", "aarti", "pooja",
        "republic day", "independence day", "26 january", "15 august",
        # Sports/Cricket events
        "cricket match", "t20", "odi", "test match", "ipl match", "ipl final", "ipl 20", "cricket",
        "wpl", "women premier league", "women's premier league",  # Women's Premier League
        "match", "sports event", "tournament", "championship",
        "conference", "summit", "ai conference", "tech conference",  # Conferences
        "launch", "launched", "launching",  # Launch events
        "police commissioner system", "commissioner system", "police commissioner",  # Police commissioner system launch
        "competition", "contest", "जसगीत", "झांकी प्रतियोगिता",  # Competitions
        "शुरुआत", "उद्घाटन", "लॉन्च", "प्रतियोगिता",  # Launch/inauguration/competition (when not political)
        "पुलिस कमिश्नर सिस्टम", "कमिश्नर प्रणाली", "पुलिस कमिश्नर प्रणाली", "पुलिस कमिश्नर",  # Police commissioner system
        # English - multi-word phrases for stronger matching
        "cultural event", "religious event",
        "launch ceremony", "marriage ceremony",
        "birthday celebration",
        # Hindi/Devanagari - strong single-word indicators
        "त्योहार", "उत्सव", "समारोह", "मेला", "प्रदर्शनी",
        "उद्घाटन", "शुभारंभ", "शादी", "विवाह", "जन्मदिन",
        "यात्रा", "पूजा", "आरती", "जुलूस",
        "गणतंत्र दिवस", "26 जनवरी", "स्वतंत्रता दिवस", "15 अगस्त",
        "मैच", "क्रिकेट", "टी20", "वनडे", "टेस्ट मैच", "डब्ल्यूपीएल"  # Cricket/sports
    ]

    # Civic keywords - specific civic/municipal service terms
    civic_keywords = [
        # English - specific civic terms
        "municipal corporation", "municipality", "municipal", "nagar nigam",
        "water supply", "electricity supply", "power supply",
        "contaminated water", "water crisis", "water problem", "water contamination",  # Water issues
        "water supply improvement", "drinking water", "पेयजल",  # Water supply improvements
        "mid-day meal food poisoning", "school meal poisoning", "midday meal poisoning",  # School meal food poisoning
        "garbage collection", "waste management", "cleanliness drive",
        "pothole repair", "road repair", "street repair",
        "road construction", "road not built", "infrastructure",
        "road quality", "road collapse", "underbridge", "अंडरब्रिज",  # Road quality issues
        "corruption", "भ्रष्टाचार",  # Municipal corruption (omit bare negligence/लापरवाही — matches intl. politics)
        "नगर निगम लापरवाही", "पानी लापरवाही", "सड़क लापरवाही", "अस्पताल लापरवाही",
        "public health", "sanitation", "swachh bharat",
        "air pollution", "aqi", "air quality index", "pm2.5", "pm10",  # Air quality issues
        "हवा खतरनाक", "प्रदूषित हवा", "वायु प्रदूषण",  # Dangerous air/air pollution
        "property tax", "house tax", "municipal tax",
        "birth certificate", "death certificate", "marriage certificate",
        "aadhaar card", "pan card", "voter id card",
        "driving license", "passport",
        "csc", "common service center",
        "mayor", "councilor",
        # Note: "commissioner" removed - too generic, appears in police commissioner (general news) and municipal commissioner (civic)
        "nrda",  # NRDA - Raipur Development Authority
        # Avoid bare "protest"/"demonstration" — matches international politics (e.g. Nepal)
        "student protest",
        # Hindi/Devanagari - specific civic terms
        "नगर निगम", "नगर पालिका",
        "पानी की समस्या", "बिजली की समस्या",
        "प्रदूषित पानी", "सीवरेज का पानी", "गंदा पानी",  # Contaminated water
        "नाली का पानी", "नाले का पानी",  # Drain water
        "मध्यान भोजन से बीमारी", "स्कूल भोजन से बीमारी", "मिड डे मील से बीमारी",  # Mid-day meal food poisoning
        "पानी पीने को मजबूर", "पानी नहीं मिल रहा",  # Water shortage
        "पानी संकट", "जल संकट",  # Water crisis
        "पेयजल व्यवस्था", "पेयजल",  # Drinking water supply
        "कचरा", "सफाई", "स्वच्छता", "गड्ढा",
        "मरम्मत", "शिकायत", "संपत्ति कर",
        "अतिक्रमण", "अतिक्रमण हटाने",  # Encroachment removal
        "सड़क नहीं बनी", "सड़क निर्माण",  # Road construction
        "सड़क धंसी", "सड़क खराब", "सड़क की गुणवत्ता",  # Road quality
        "जन्म प्रमाणपत्र", "मृत्यु प्रमाणपत्र", "विवाह प्रमाणपत्र",
        "आधार", "पैन कार्ड", "मतदाता पहचान",
        "ड्राइविंग लाइसेंस", "पासपोर्ट",
        "मेयर", "पार्षद",
        # Note: "कमिश्नर" removed - too generic, appears in police commissioner (general) and municipal commissioner (civic)
        "छात्र प्रदर्शन", "उग्र प्रदर्शन", "प्रदर्शन",  # Student protest
        # Safety inspections and administrative checks
        "सुरक्षा जांच", "स्कूल बस जांच", "वाहन जांच", "सार्वजनिक जांच"
    ]
    
    # Politics keywords - specific political terms
    politics_keywords = [
        # English - specific political terms
        "election", "voting", "vote", "election campaign",
        "minister", "chief minister", "cm", "mla",
        # Omit bare "mp" — in regional news it almost always means Madhya Pradesh, not MP (MP)
        "member of parliament", "lok sabha", "rajya sabha",
        "prime minister", "pm modi", "narendra modi", "modi",
        "government", "govt", "political party",
        "political rally", "political speech", "political news",
        "budget", "assembly session", "parliament",
        "political leader", "politician",
        "councilor", "councillor", "alderman",
        "government scheme", "yojana", "scheme",  # Government schemes
        "paddy procurement", "dhan kharidi", "rice procurement",  # Paddy procurement
        "election result", "election results", "vote counting", "poll results",  # Election results
        "सियासत", "सियासी", "कांड पर सियासत",  # Politics/controversy + politics
        # Major political parties (common in Indian news)
        "congress", "bjp", "aap", "sp", "bsp", "dmk", "aiadmk", "tmc", "cpi", "cpm",
        # Political leaders and figures
        "owaisi", "asaduddin owaisi",
        # Political issues and terms
        "land jihad", "jihad",
        # Hindi/Devanagari - specific political terms
        "चुनाव", "मतदान", "वोट",
        "मंत्री", "मुख्यमंत्री", "विधायक", "सांसद",
        "प्रधानमंत्री", "पीएम मोदी", "मोदी",
        "सरकार", "पार्टी", "राजनीति", "राजनीतिक", "सियासी",
        "रैली", "भाषण", "अभियान",
        "बजट", "विधानसभा", "संसद", "नेता",
        "ओवैसी",  # Owaisi - political leader
        "लैंड जिहाद", "जिहाद",  # Land jihad - political issue
        "पार्षद",  # Councilor - political position
        "योजना", "सरकारी योजना", "लाडली बहना योजना",  # Government schemes
        # Omit bare "धान" — false Politics on "धान के पैरा" (rice stalk) human-interest etc.
        "धान खरीदी", "धान खरीद", "msp धान", "धान की खरीद",
        "चुनाव का रिजल्ट", "चुनाव रिजल्ट", "मतगणना", "वोट गिनती",  # Election results
        # Neighbouring / international politics often covered by MP–CG channels
        "nepal", "नेपाल", "kathmandu", "oli", "ओली", "kp oli", "sharma oli", "केपी शर्मा ओली",
    ]

    # Local health / disease outbreak — before Politics so "सांसद ने निरीक्षण" jaundice stories stay General
    health_outbreak_keywords = [
        "jaundice", "पीलिया", "डेंगू", "dengue", "malaria", "मलेरिया",
        "chikungunya", "चिकनगुनिया", "typhoid", "टाइफॉयड",
        "food poisoning", "फूड पॉइजनिंग", "food poison",
        "संक्रमण फैला", "महामारी", "epidemic", "outbreak",
        "पीलिया का प्रकोप", "डेंगू का प्रकोप", "बुखार का प्रकोप",
    ]

    # Weather keywords - check BEFORE Politics to avoid "MP Weather" matching "MP" (Member of Parliament)
    weather_keywords = [
        "weather", "weather alert", "weather update", "weather forecast",
        "cold wave", "heat wave", "rain", "rainfall", "monsoon",
        "temperature", "humidity", "fog", "foggy", "winter", "summer",
        "snow", "snowfall", "storm", "cyclone", "typhoon",
        "मौसम", "मौसम की जानकारी", "मौसम अपडेट", "मौसम चेतावनी",
        "ठंड", "सर्दी", "गर्मी", "बारिश", "वर्षा", "बारिश",
        "तापमान", "पारा", "कोहरा", "सर्द हवाएं", "बर्फीली हवाएं",
        "शीतलहर", "गर्मी की लहर", "बारिश के आसार", "मावठा"
    ]
    
    # Create patterns and check in order of specificity
    crime_patterns = create_patterns(crime_keywords)
    traffic_patterns = create_patterns(traffic_keywords)
    jobs_patterns = create_patterns(jobs_keywords)
    events_patterns = create_patterns(events_keywords)
    civic_patterns = create_patterns(civic_keywords)
    politics_patterns = create_patterns(politics_keywords)
    weather_patterns = create_patterns(weather_keywords)
    health_outbreak_patterns = create_patterns(health_outbreak_keywords)

    # Check each category - order matters (most specific first)
    # Weather - check BEFORE Politics to avoid "MP Weather" matching "MP" (Member of Parliament)
    for pattern in weather_patterns:
        if pattern.search(text):
            return GENRE_GENERAL  # Weather news is General

    for pattern in health_outbreak_patterns:
        if pattern.search(text):
            return GENRE_GENERAL

    # Traffic - check BEFORE Crime (road accidents can have deaths but should be Traffic)
    for pattern in traffic_patterns:
        if pattern.search(text):
            return GENRE_TRAFFIC

    # Jobs - employment opportunities
    _jobs_skip_substrings = [
        "धर्मांतरण", "जबरन धर्मांतरण", "religious conversion", "conversion law",
        "anti conversion", "anti-conversion", "धर्म स्वातंत्र्य",
        "नौकरी के नाम पर", "नौकरी के नाम से",
        "puchti hai public", "पूछती है पब्लिक", "पुचति है पब्लिक",
        "medical shortage", "डॉक्टर्स की कमी", "डॉक्टरों की कमी",
    ]
    _jobs_skip_school_exam_markers = [
        "बोर्ड परीक्षा", "board exam", "board examination",
        "छात्रों ने लिखा", "छात्र ने लिखा", "अजब गजब उत्तर", "टीचर हैरान",
        "students wrote", "exam answers",
    ]
    _jobs_skip_if_both = [
        ("नौकरी", "ठगी"),
        ("नौकरी", "फर्जीवाड़ा"),
        ("नौकरी", "धोखा"),
        ("job", "fraud"),
        ("job", "scam"),
        ("भर्ती", "ठगी"),
    ]

    def _should_skip_jobs_for_crime_or_religion(t: str) -> bool:
        if any(s in t for s in _jobs_skip_substrings):
            return True
        if any(s in t for s in _jobs_skip_school_exam_markers):
            return True
        return any(a in t and b in t for a, b in _jobs_skip_if_both)

    for pattern in jobs_patterns:
        if pattern.search(text):
            if _should_skip_jobs_for_crime_or_religion(text):
                continue
            return GENRE_JOBS

    # Crime - check BEFORE Events to catch crimes at events (e.g., "मेला में दुष्कर्म")
    for pattern in crime_patterns:
        if pattern.search(text):
            return GENRE_CRIME
    
    # Events - check BEFORE Politics to catch conferences/events/launches (not political discussions)
    _events_crime_story_markers = [
        "murder", "killed", "killing", "homicide", "rape", "kidnap", "abduct",
        "हत्या", "कत्ल", "मर्ड", "दुष्कर्म", "अपहरण", "चाकू", "गोली", "फायरिंग",
        "लूट", "चोरी", "robbery", "theft", "snatching", "हत्याकांड",
        "fraud case", "scam case", "ठगी", "फर्जीवाड़ा केस",
        "controversy", "विवाद", "लापता", "missing", "भाग गई", "फरार",
        "खुलासा", "सट्टा", "betting", "viral girl", "वायरल गर्ल", "monalisa",
        "case update", "केस में",
    ]
    for pattern in events_patterns:
        if pattern.search(text):
            if any(m in text for m in _events_crime_story_markers):
                continue
            # But exclude if it's clearly a political discussion (e.g., "CM meets PM", "political rally")
            # Policy launches and inaugurations are Events, not Politics
            if not any(pol_term in text for pol_term in ["meets", "मुलाकात", "रैली", "भाषण", "चुनाव"]):
                return GENRE_EVENTS
    
    # Politics - check AFTER Events and Crime to catch political discussions/scams (not criminal cases)
    for pattern in politics_patterns:
        if pattern.search(text):
            # But exclude if it's clearly a crime (e.g., "fraud case", "scam case")
            # And exclude if it's a launch/inauguration event (those are Events)
            if not any(crime_term in text for crime_term in ["fraud case", "scam case", "स्कैम केस", "धोखाधड़ी केस"]):
                if not any(event_term in text for event_term in ["launch", "लॉन्च", "शुरुआत", "उद्घाटन", "conference"]):
                    return GENRE_POLITICS

    # Civic - municipal services
    for pattern in civic_patterns:
        if pattern.search(text):
            return GENRE_CIVIC
    
    return GENRE_GENERAL


# Context prototypes for cosine similarity. Negative / harm stories sit on Crime
# so a partial overlap (crop loss, raid, death, blast) still beats a passing
# politician or city name.
_CONTEXT_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "is", "are",
    "with", "from", "at", "by", "this", "that", "news", "update", "live",
    "में", "को", "की", "का", "के", "से", "ने", "है", "हैं", "पर", "और", "या",
    "एक", "यह", "इस", "कि", "भी", "हो", "गया", "गई", "तो", "ही", "तक", "कर",
    "किया", "लिए", "वाला", "वाले", "वाली", "नहीं", "रहा", "रहे", "रही", "सिर्फ",
}

# Common words that must not count on their own. They still count inside a phrase.
_CONTEXT_UNIGRAM_SKIP = {
    "नहीं", "रहा", "रहे", "रही", "शराब", "helicopter", "हेलीकॉप्टर",
    "किसान", "किसानों", "फसल", "फसलें", "धान", "बारिश", "चक्रवात", "तूफान",
}

_CONTEXT_PHRASES = (
    "नहीं रहे",
    "फसल बर्बाद",
    "फसल को नुकसान",
    "फसलें तबाह",
    "फसलें बर्बाद",
    "helicopter crash",
    "हेलीकॉप्टर क्रैश",
    "शराब जब्त",
    "गंभीर आरोप",
    "आर्थिक संकट",
)

_GENRE_CONTEXT_EXAMPLES = {
    GENRE_CRIME: [
        "murder arrested raid fraud scam robbery theft kidnapping shooting blast explosion missing torture accused",
        "हत्या गिरफ्तार गिरफ्तारी छापेमारी ठगी घोटाला लूट चोरी अपहरण लापता फरार विस्फोट आरोपी कांड",
        "फसल बर्बाद फसल को नुकसान फसलें तबाह किसानों की फसल चौपट चक्रवात से तबाही कहर संकट",
        "निधन मौत नहीं रहे अंतिम संस्कार helicopter crash हेलीकॉप्टर क्रैश विमान हादसा",
        "गैस ब्लास्ट सिलेंडर धमाका बाढ़ तबाही",
        "शराब जब्त हेरोइन पिस्टल मादक पदार्थ अवैध ब्याज खुलासा",
        "ईडी रेड एसीबी छापा वसूली धोखाधड़ी गंभीर आरोप",
        "मारपीट हमला हिंसा फायरिंग गोली चाकू बवाल",
        "मंगलसूत्र छीन snatching chain snatching",
    ],
    GENRE_TRAFFIC: [
        "road accident car crash traffic jam collision highway overturned hit and run",
        "सड़क दुर्घटना कार हादसा टक्कर जाम ट्रैफिक पलटी एक्सीडेंट",
    ],
    GENRE_POLITICS: [
        "election rally speech minister scheme modi congress bjp vote budget parliament",
        "चुनाव रैली भाषण मंत्री मुख्यमंत्री योजना सम्मान निधि किस्त विधानसभा कांग्रेस भाजपा",
        "लोकतंत्र बचाओ पदयात्रा अभियान सियासत",
    ],
    GENRE_EVENTS: [
        "festival celebration ceremony puja cricket match wedding inauguration karwa chauth",
        "त्योहार उत्सव समारोह पूजा मेला क्रिकेट मैच शादी उद्घाटन करवा चौथ",
    ],
    GENRE_JOBS: [
        "job vacancy recruitment admit card government job result hiring",
        "नौकरी भर्ती वैकेंसी एडमिट कार्ड सरकारी नौकरी रिजल्ट",
    ],
    GENRE_CIVIC: [
        "municipal water supply garbage road repair cleanliness certificate",
        "नगर निगम पानी सफाई कचरा सड़क निर्माण पेयजल शिकायत",
    ],
    GENRE_GENERAL: [
        "weather forecast rain alert temperature horoscope air show",
        "मौसम अपडेट बारिश का अलर्ट तापमान राशिफल",
    ],
}

# Crime wins when its context cosine is real and close to the next-best genre.
SEMANTIC_CRIME_MIN = 0.12
SEMANTIC_CRIME_RATIO = 0.72

_context_idf: Optional[Dict[str, float]] = None
_genre_example_vectors: Optional[Dict[str, List[Dict[str, float]]]] = None


def _context_tokens(text: str) -> Counter:
    """Word, phrase, and adjacent-word tokens used as the context vector."""
    normalized = normalize(text)
    words = [w for w in normalized.split() if len(w) >= 2 and w not in _CONTEXT_STOPWORDS]
    counts: Counter = Counter()
    for word in words:
        if word in _CONTEXT_UNIGRAM_SKIP:
            continue
        counts["w:" + word] += 2
    for left, right in zip(words, words[1:]):
        if left in _CONTEXT_UNIGRAM_SKIP and right in _CONTEXT_UNIGRAM_SKIP:
            continue
        counts["b:" + left + "_" + right] += 3
    for phrase in _CONTEXT_PHRASES:
        if phrase in normalized:
            counts["p:" + phrase.replace(" ", "_")] += 4
    return counts


def _tfidf_vector(counts: Counter, idf: Dict[str, float]) -> Dict[str, float]:
    vector: Dict[str, float] = {}
    for token, count in counts.items():
        weight = idf.get(token)
        if not weight or count <= 0:
            continue
        tf = 1.0 + math.log(count)
        vector[token] = tf * weight
    return vector


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    dot = 0.0
    for token, weight in left.items():
        other = right.get(token)
        if other:
            dot += weight * other
    left_norm = math.sqrt(sum(weight * weight for weight in left.values()))
    right_norm = math.sqrt(sum(weight * weight for weight in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _ensure_context_vectors() -> None:
    global _context_idf, _genre_example_vectors
    if _genre_example_vectors is not None and _context_idf is not None:
        return
    documents = []
    for examples in _GENRE_CONTEXT_EXAMPLES.values():
        for example in examples:
            documents.append(_context_tokens(example))
    doc_count = len(documents) or 1
    doc_freq: Counter = Counter()
    for document in documents:
        doc_freq.update(document.keys())
    _context_idf = {
        token: math.log((1 + doc_count) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }
    _genre_example_vectors = {
        genre: [_tfidf_vector(_context_tokens(example), _context_idf) for example in examples]
        for genre, examples in _GENRE_CONTEXT_EXAMPLES.items()
    }


def semantic_context_scores(title: str, description: str = "") -> Dict[str, float]:
    """Cosine similarity of a video against each genre's context examples.

    Title is repeated so the headline dominates the description.
    The score for a genre is the closest example, not a diluted average.
    """
    _ensure_context_vectors()
    assert _context_idf is not None and _genre_example_vectors is not None
    query_text = f"{title or ''} {title or ''} {title or ''} {(description or '')[:400]}"
    query = _tfidf_vector(_context_tokens(query_text), _context_idf)
    scores: Dict[str, float] = {}
    for genre, vectors in _genre_example_vectors.items():
        scores[genre] = max((_cosine_similarity(query, vector) for vector in vectors), default=0.0)
    return scores


def apply_negative_crime_context(genre: str, scores: Dict[str, float]) -> str:
    """Move negative stories into Crime when that context is the closest match.

    Road-traffic stories stay Traffic. Podcast, Live, and Scheduled labels are left alone.
    """
    if genre in (GENRE_PODCAST, GENRE_LIVE, GENRE_SCHEDULED, GENRE_CRIME):
        return genre
    if genre == GENRE_TRAFFIC:
        return GENRE_TRAFFIC
    crime = scores.get(GENRE_CRIME, 0.0)
    best_other = max((value for name, value in scores.items() if name != GENRE_CRIME), default=0.0)
    if crime >= SEMANTIC_CRIME_MIN and crime >= best_other * SEMANTIC_CRIME_RATIO:
        return GENRE_CRIME
    return genre


def classify_genre_keyword_based(title: str, description: str = "", channel_title: str = "") -> str:
    """Classify by keywords, then pull negative context into Crime.

    Every video from a podcast channel is Podcast, as is any title that is itself a podcast.
    """
    if _is_podcast_video(title, description, channel_title):
        return GENRE_PODCAST
    genre = _classify_genre_by_keywords(title, description)
    return apply_negative_crime_context(genre, semantic_context_scores(title, description))


def fetch_video_details(video_ids: List[str]) -> Dict[str, Dict]:
    """Fetch detailed information for video IDs."""
    if not video_ids:
        return {}
    
    # Retry logic with API key fallback
    while True:
        api_key = get_current_api_key()
        if not api_key:
            return {}
            
        params = {
            "part": "snippet,contentDetails,statistics,liveStreamingDetails",
            "id": ",".join(video_ids),
            "key": api_key,
        }
        
        try:
            response = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                data = response.json()
                break
            elif is_quota_exceeded_error(response):
                print(MSG_QUOTA_EXCEEDED.format(current_api_key_index + 1))
                if not switch_to_next_api_key():
                    print(MSG_ALL_KEYS_EXHAUSTED)
                    return {}
                continue
            elif is_invalid_api_key_error(response):
                print(MSG_INVALID_API_KEY.format(current_api_key_index + 1))
                print(MSG_CHECK_API_KEY)
                if not switch_to_next_api_key():
                    print(MSG_ALL_KEYS_EXHAUSTED)
                    return {}
                continue
            else:
                print(f"Error fetching video details: HTTP {response.status_code}")
                return {}
        except Exception as e:
            print(f"Error fetching video details: {e}")
            return {}
    
    details: Dict[str, Dict] = {}
    
    for item in data.get("items", []):
        video_id = item.get("id")
        snippet = item.get("snippet", {}) or {}
        content = item.get("contentDetails", {}) or {}
        stats = item.get("statistics", {}) or {}
        live = item.get("liveStreamingDetails", {}) or {}
        
        iso = (content.get("duration") or "").upper()
        live_bc = (snippet.get("liveBroadcastContent") or "").lower()
        
        secs = 0
        duration_formatted = "00:00"
        video_type = VIDEO_TYPE_VOD
        
        if live_bc == "live" or (live.get("actualStartTime") and not live.get("actualEndTime")):
            video_type = VIDEO_TYPE_LIVE
            duration_formatted = VIDEO_TYPE_LIVE
        elif iso:
            secs = parse_iso_duration(iso)
            duration_formatted = format_duration(secs)
            if secs == 0 and iso in ("", "P0D", "PT0S", "P0DT0S"):
                video_type = VIDEO_TYPE_SCHEDULED
                duration_formatted = VIDEO_TYPE_SCHEDULED
            else:
                video_type = VIDEO_TYPE_VOD
        
        details[video_id] = {
            "durationSeconds": secs,
            "durationFormatted": duration_formatted,
            "videoType": video_type,
            "views": _safe_int(stats.get("viewCount", 0)),
            "likes": _safe_int(stats.get("likeCount", 0)),
        }
    
    return details


def fetch_latest_videos_for_channel(channel_id: str) -> List[Dict]:
    """Fetch latest videos from a channel."""
    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=LAST_HOURS_FILTER)
    ).isoformat().replace("+00:00", "Z")
    
    # Retry logic with API key fallback
    while True:
        api_key = get_current_api_key()
        if not api_key:
            return []
            
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "order": "date",
            "maxResults": MAX_RESULTS,
            "type": "video",
            "key": api_key,
            "safeSearch": "strict",
            "publishedAfter": published_after,
        }
        
        try:
            response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                break
            elif is_quota_exceeded_error(response):
                print(MSG_QUOTA_EXCEEDED.format(current_api_key_index + 1) + f" for channel {channel_id}")
                if not switch_to_next_api_key():
                    print(MSG_ALL_KEYS_EXHAUSTED)
                    return []
                continue
            elif is_invalid_api_key_error(response):
                print(MSG_INVALID_API_KEY.format(current_api_key_index + 1) + f" for channel {channel_id}")
                print(MSG_CHECK_API_KEY)
                if not switch_to_next_api_key():
                    print(MSG_ALL_KEYS_EXHAUSTED)
                    return []
                continue
            else:
                print(f"[ERROR] HTTP {response.status_code}: {response.text}")
                return []
        except Exception as e:
            print(f"[NETWORK ERROR] channel {channel_id}: {e}")
            return []
    
    items = response.json().get("items", [])
    video_ids = [item["id"]["videoId"] for item in items if item.get("id", {}).get("videoId")]
    video_details = fetch_video_details(video_ids)
    
    videos: List[Dict] = []
    for item in items:
        vid_dict = item.get("id", {})
        video_id = vid_dict.get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet", {})
        video_info = video_details.get(video_id, {})
        
        videos.append({
            "videoId": video_id,
            "title": snippet.get("title"),
            "description": snippet.get("description"),
            "channelTitle": snippet.get("channelTitle"),
            "publishedAt": snippet.get("publishedAt"),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url"),
            "durationSeconds": video_info.get("durationSeconds"),
            "durationFormatted": video_info.get("durationFormatted"),
            "views": video_info.get("views", 0),
            "likes": video_info.get("likes", 0),
            "videoType": video_info.get("videoType", VIDEO_TYPE_UNKNOWN),
        })
    
    return videos


def aggregate_bhilai_videos(channels: List[str]) -> List[Dict]:
    """Fetch videos for each channel and merge them."""
    all_videos: List[Dict] = []
    
    for ch in channels:
        vids = fetch_latest_videos_for_channel(ch)
        all_videos.extend(vids)
    
    all_videos.sort(key=lambda v: v.get("publishedAt", ""), reverse=True)
    return all_videos


def add_genres_to_feed(feed: List[Dict]) -> List[Dict]:
    """
    Add genre classification to each video.
    Only videos published in the last 1 hour use OpenAI; others use keyword-based classification.
    Circuit breaker: Once OpenAI fails, stops trying OpenAI for rest of batch.
    Optimized for fast execution (< 1 minute) and rate limit safety.
    """
    start_time = time.time()
    
    total_videos = len(feed)
    openai_count = 0
    keyword_count = 0
    special_count = 0
    recent_count = 0
    openai_failed = False  # Circuit breaker flag
    execution_time_exceeded = False  # Time limit flag

    print(f"[Classification] Starting genre classification for {total_videos} videos...")
    print(f"[Classification] Strategy: OpenAI for videos published in last 1 hour, keyword-based for others")
    print(f"[Classification] Circuit breaker: Will stop using OpenAI after first failure")
    print(f"[Classification] Time limit: {OPENAI_MAX_EXECUTION_TIME}s for OpenAI classification")

    for i, v in enumerate(feed, 1):
        # Check execution time - if exceeded, use keyword-based for all remaining
        elapsed_time = time.time() - start_time
        if elapsed_time > OPENAI_MAX_EXECUTION_TIME:
            if not execution_time_exceeded:
                print(f"[Classification] Execution time limit reached ({elapsed_time:.1f}s), using keyword-based for remaining {total_videos - i + 1} videos")
                execution_time_exceeded = True
            openai_failed = True  # Treat as failure to use keyword-based
        video_type = (v.get("videoType") or "").strip().upper()
        
        if video_type == VIDEO_TYPE_LIVE:
            v["genre"] = GENRE_LIVE
            special_count += 1
        elif VIDEO_TYPE_SCHEDULED in video_type or "UPCOMING" in video_type:
            v["genre"] = GENRE_SCHEDULED
            special_count += 1
        else:
            title = v.get("title", "")
            description = v.get("description", "")
            published_at = v.get("publishedAt", "")
            channel_title = v.get("channelTitle", "")  # Get channelTitle for Podcast classification

            # Check if video is recent (within last 1 hour)
            is_recent = is_recent_video(published_at, hours=1)
            if is_recent:
                recent_count += 1
            
            # Only use OpenAI for recent videos (published in last 1 hour) and if circuit breaker hasn't tripped
            if is_recent and not openai_failed and not execution_time_exceeded:
                # Quick rate limit check before attempting
                if not _check_openai_rate_limits():
                    if not openai_failed:
                        print(f"[Classification] Rate limit safety threshold reached, circuit breaker activated - using keyword-based for remaining {total_videos - i} videos")
                        openai_failed = True
                    v["genre"] = classify_genre_keyword_based(title, description, channel_title)
                    keyword_count += 1
                else:
                    openai_result = classify_genre_with_openai(title, description)
                    if openai_result:
                        if _is_podcast_video(title, description, channel_title):
                            v["genre"] = GENRE_PODCAST
                        else:
                            v["genre"] = apply_negative_crime_context(
                                openai_result, semantic_context_scores(title, description)
                            )
                        openai_count += 1
                    else:
                        # OpenAI failed - set circuit breaker and use keyword-based
                        if not openai_failed:
                            print(f"[Classification] OpenAI failed, circuit breaker activated - using keyword-based for remaining {total_videos - i} videos")
                            openai_failed = True
                        v["genre"] = classify_genre_keyword_based(title, description, channel_title)
                        keyword_count += 1
            else:
                # Use keyword-based directly for older videos or if circuit breaker is active
                v["genre"] = classify_genre_keyword_based(title, description, channel_title)
                keyword_count += 1
        
        # Progress indicator for large batches
        if total_videos > 50 and i % 50 == 0:
            status = " (Circuit breaker active)" if openai_failed else ""
            print(f"[Classification] Progress: {i}/{total_videos} videos classified (Recent: {recent_count}, OpenAI: {openai_count}, Keyword: {keyword_count}, Special: {special_count}){status}")
    
    elapsed_time = time.time() - start_time
    print(f"[Classification] Complete: {recent_count} recent videos, {openai_count} OpenAI, {keyword_count} keyword-based, {special_count} special types (Live/Scheduled)")
    print(f"[Classification] Execution time: {elapsed_time:.2f}s")
    if openai_failed:
        if execution_time_exceeded:
            print(f"[Classification] Time limit exceeded - OpenAI stopped after {OPENAI_MAX_EXECUTION_TIME}s")
        else:
            print(f"[Classification] Circuit breaker was activated - OpenAI stopped after first failure")
    if (total_videos - special_count) > 0:
        recent_percent = (recent_count / (total_videos - special_count)) * 100 if (total_videos - special_count) > 0 else 0
        openai_percent = (openai_count / recent_count) * 100 if recent_count > 0 else 0
        print(f"[Classification] Summary: {recent_count}/{total_videos - special_count} recent ({recent_percent:.1f}%), {openai_count}/{recent_count} used OpenAI ({openai_percent:.1f}% of recent)")
    else:
        print("[Classification] Summary: All videos were special types")
    
    return feed


def filter_by_genre(feed: List[Dict], genres: List[str]) -> List[Dict]:
    """Filter videos by genre."""
    return [v for v in feed if v.get("genre") in genres]


def _is_valid_ip(ip: str) -> bool:
    """Basic IPv4 validation."""
    if not ip:
        return False
    parts = ip.strip().split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit():
            return False
        n = int(p)
        if n < 0 or n > 255:
            return False
    return True


def _resolve_ip_lookup_url(ip: str, template: Optional[str] = None) -> str:
    """Build lookup URL from a configurable template/url.

    Supported formats:
    - Template with {ip}: http://ip-api.com/json/{ip}
    - Raw endpoint base:   http://ip-api.com/json/
    - Full URL with query: https://example.com/lookup?ip={ip}
    """
    template = (template or os.environ.get(ENV_IP_LOOKUP_URL) or IP_LOOKUP_URL_DEFAULT).strip()
    ip_q = quote_plus(ip.strip())
    _log_city(f"Resolving lookup URL using template='{template}' and ip='{ip}'")
    if "{ip}" in template:
        return template.replace("{ip}", ip_q)
    if template.endswith("/"):
        return f"{template}{ip_q}"
    return f"{template}?ip={ip_q}"


def resolve_city_from_ip(ip: str, lookup_template: Optional[str] = None) -> str:
    """Resolve city for an IP with cache and safe fallbacks.

    Returns empty string if city cannot be resolved; never raises.
    """
    ip = (ip or "").strip()
    if not _is_valid_ip(ip):
        _log_city(f"Skipping city lookup: invalid IP '{ip}'")
        return ""

    _prune_ip_city_cache()
    cached = _ip_city_cache.get(ip)
    now = time.time()
    if cached and now - float(cached.get("ts", 0)) <= IP_LOOKUP_CACHE_TTL_SECONDS:
        _log_city(f"Cache hit for IP '{ip}' -> city '{cached.get('city', '')}'")
        return str(cached.get("city", "") or "")
    _log_city(f"Cache miss for IP '{ip}', querying city service")

    url = _resolve_ip_lookup_url(ip, lookup_template)
    attempts = 2
    for attempt in range(attempts):
        try:
            _log_city(f"IP lookup attempt {attempt + 1}/{attempts}: {url}")
            response = requests.get(url, timeout=IP_LOOKUP_TIMEOUT)
            if response.status_code != 200:
                _log_city(f"Lookup non-200 status={response.status_code} for IP '{ip}'")
                continue
            data = response.json() if hasattr(response, "json") else {}
            city = str((data or {}).get("city", "")).strip()
            if city:
                _ip_city_cache[ip] = {"city": city, "ts": now}
                _log_city(f"Resolved IP '{ip}' -> city '{city}' (cached)")
                return city
            _log_city(f"Lookup succeeded but city missing in response for IP '{ip}'")
        except Exception:
            # Retry once, then fallback to no-city section.
            if attempt < attempts - 1:
                _log_city(f"Lookup attempt {attempt + 1} failed for IP '{ip}', retrying")
                continue
    _log_city(f"Failed to resolve city for IP '{ip}', city section will be skipped")
    return ""


# Places and names that mean the story is about Chhattisgarh or Madhya Pradesh.
_CG_MP_REGION_TERMS = (
    "chhattisgarh", "chattisgarh", "chhatisgarh", "chhattishgarh", "chhattishgard",
    "chatisgarh", "chhattisgarhi", "madhya pradesh", "madhyapradesh", "madya pradesh",
    "madyapradesh", "mpcg", "mp news", "cg news",
    "छत्तीसगढ़", "छत्तीसगढ", "छतीसगढ़", "छत्तीसगढ़ी", "मध्य प्रदेश", "मध्यप्रदेश",
    "raipur", "रायपुर", "bhilai", "भिलाई", "bilaspur", "बिलासपुर", "raigarh", "रायगढ़",
    "korba", "कोरबा", "jagdalpur", "जगदलपुर", "bastar", "बस्तर", "rajnandgaon",
    "राजनांदगांव", "राजनांदगाँव", "ambikapur", "अंबिकापुर", "dhamtari", "धमतरी",
    "kanker", "कांकेर", "dantewada", "दंतेवाड़ा", "दंतेवाडा", "mahasamund", "महासमुंद",
    "janjgir", "जांजगीर", "kawardha", "कवर्धा", "bemetara", "बेमेतरा", "gariaband",
    "gariyaband", "गरियाबंद", "surguja", "सरगुजा", "kondagaon", "कोंडागांव",
    "sukma", "सुकमा", "mungeli", "मुंगेली", "bhatapara", "भाटापारा", "dongargarh",
    "डोंगरगढ़", "rajim", "राजिम", "khairagarh", "खैरागढ़", "mainpat", "मैनपाट",
    "ratanpur", "रतनपुर", "abhanpur", "अभनपुर", "arang", "आरंग", "charoda", "चरोदा",
    "durg", "दुर्ग",
    "bhopal", "भोपाल", "indore", "इंदौर", "इन्दौर", "jabalpur", "जबलपुर", "gwalior",
    "ग्वालियर", "ujjain", "उज्जैन", "rewa", "रीवा", "satna", "सतना", "ratlam", "रतलाम",
    "dewas", "देवास", "khandwa", "खंडवा", "chhindwara", "छिंदवाड़ा", "burhanpur",
    "बुरहानपुर", "singrauli", "सिंगरौली", "shahdol", "शहडोल", "mandsaur", "मंदसौर",
    "neemuch", "नीमच", "vidisha", "विदिशा", "sehore", "सिहोर", "narmadapuram",
    "नर्मदापुरम", "katni", "कटनी", "shivpuri", "शिवपुरी", "betul", "बैतूल", "balaghat",
    "बालाघाट", "chhatarpur", "छतरपुर", "tikamgarh", "टीकमगढ़", "itarsi", "इटारसी",
    "khargone", "खरगोन", "khandwa", "omkareshwar", "ओंकारेश्वर", "pachmarhi", "पचमढ़ी",
    "amarkantak", "अमरकंटक", "maheshwar", "महेश्वर", "jhabua", "झाबुआ", "barwani",
    "बड़वानी", "morena", "मुरैना", "datia", "दतिया", "malwa", "मालवा",
)

# Other Indian states. A headline that names one of these, and does not also name
# Chhattisgarh or Madhya Pradesh, is left out of the feed.
_OTHER_STATE_REGION_TERMS = (
    "uttar pradesh", "uttarpradesh", "उत्तर प्रदेश", "उत्तरप्रदेश", "यूपी", "up", "uppsc", "uppcl",
    "lucknow", "लखनऊ", "kanpur", "कानपुर", "varanasi", "वाराणसी", "banaras", "बनारस",
    "noida", "नोएडा", "ghaziabad", "गाजियाबाद", "agra", "आगरा", "prayagraj", "प्रयागराज",
    "meerut", "मेरठ", "gorakhpur", "गोरखपुर", "ayodhya", "अयोध्या", "mathura", "मथुरा",
    "vrindavan", "वृंदावन", "aligarh", "अलीगढ़", "bareilly", "बरेली", "moradabad", "मुरादाबाद",
    "jhansi", "झांसी", "saharanpur", "सहारनपुर", "muzaffarnagar", "मुजफ्फरनगर", "amethi", "अमेठी",
    "raebareli", "rae bareli", "रायबरेली", "azamgarh", "आजमगढ़", "yogi adityanath",
    "योगी आदित्यनाथ", "cm yogi", "सीएम योगी",
    "bihar", "बिहार", "patna", "पटना", "muzaffarpur", "मुजफ्फरपुर", "bhagalpur", "भागलपुर",
    "rajasthan", "राजस्थान", "jaipur", "जयपुर", "jodhpur", "जोधपुर", "udaipur", "उदयपुर",
    "ajmer", "अजमेर", "bikaner", "बीकानेर", "jaisalmer", "जैसलमेर", "bharatpur", "भरतपुर",
    "maharashtra", "महाराष्ट्र", "mumbai", "मुंबई", "bombay", "pune", "पुणे", "nagpur", "नागपुर",
    "nashik", "नासिक", "thane", "ठाणे",
    "gujarat", "गुजरात", "ahmedabad", "अहमदाबाद", "gandhinagar", "गांधीनगर", "vadodara", "वडोदरा",
    "rajkot", "राजकोट",
    "punjab", "पंजाब", "amritsar", "अमृतसर", "ludhiana", "लुधियाना", "jalandhar", "जालंधर",
    "chandigarh", "चंडीगढ़",
    "haryana", "हरियाणा", "gurugram", "गुरुग्राम", "gurgaon", "गुड़गांव", "faridabad", "फरीदाबाद",
    "panipat", "पानीपत", "rohtak", "रोहतक", "sonipat", "सोनीपत",
    "jharkhand", "झारखंड", "ranchi", "रांची", "jamshedpur", "जमशेदपुर", "dhanbad", "धनबाद",
    "bokaro", "बोकारो",
    "odisha", "odisa", "orissa", "ओडिशा", "उड़ीसा", "bhubaneswar", "भुवनेश्वर", "cuttack", "कटक",
    "west bengal", "पश्चिम बंगाल", "kolkata", "calcutta", "कोलकाता", "कलकत्ता", "howrah", "हावड़ा",
    "siliguri", "सिलीगुड़ी", "darjeeling", "दार्जिलिंग", "bengal", "बंगाल",
    "karnataka", "कर्नाटक", "bengaluru", "bangalore", "बेंगलुरु", "बेंगलोर", "mysuru", "mysore", "मैसूर",
    "telangana", "तेलंगाना", "hyderabad", "हैदराबाद",
    "andhra", "आंध्र", "आन्ध्र", "visakhapatnam", "vijayawada", "विजयवाड़ा",
    "tamil nadu", "tamilnadu", "तमिलनाडु", "chennai", "चेन्नई", "madurai", "मदुरै", "coimbatore",
    "kerala", "केरल", "thiruvananthapuram", "तिरुवनंतपुरम", "kochi", "कोच्चि",
    "assam", "असम", "guwahati", "गुवाहाटी",
    "goa", "गोवा",
    "himachal", "हिमाचल", "shimla", "शिमला", "dharamshala", "धर्मशाला",
    "uttarakhand", "उत्तराखंड", "उत्तराखण्ड", "dehradun", "देहरादून", "haridwar", "हरिद्वार",
    "rishikesh", "ऋषिकेश", "nainital", "नैनीताल",
    "jammu", "जम्मू", "kashmir", "कश्मीर", "srinagar", "श्रीनगर", "ladakh", "लद्दाख",
    "manipur", "मणिपुर", "meghalaya", "मेघालय", "mizoram", "मिजोरम", "nagaland", "नागालैंड",
    "tripura", "त्रिपुरा", "sikkim", "सिक्किम", "arunachal", "अरुणाचल",
    "delhi", "दिल्ली", "new delhi", "नई दिल्ली",
    "puducherry", "पुडुचेरी",
)

_REGION_TITLE_EXCEPTIONS = (
    (r"बंगाल की खाड़ी", " "),
    (r"bay of bengal", " "),
    (r"mini kashmir", " "),
    (r"मिनी कश्मीर", " "),
)


def _region_term_pattern(terms) -> re.Pattern:
    """Match a place name as a whole word, in Latin or Devanagari."""
    latin = []
    devanagari = []
    for term in terms:
        term = term.strip().lower()
        if not term:
            continue
        if re.search(r"[a-z]", term):
            latin.append(re.escape(term))
        else:
            devanagari.append(re.escape(term))
    parts = []
    if latin:
        body = "|".join(sorted(set(latin), key=len, reverse=True))
        parts.append(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")
    if devanagari:
        body = "|".join(sorted(set(devanagari), key=len, reverse=True))
        parts.append(rf"(?<![\u0900-\u097F])(?:{body})(?![\u0900-\u097F])")
    return re.compile("|".join(parts))


_CG_MP_REGION_PATTERN = _region_term_pattern(_CG_MP_REGION_TERMS)
_OTHER_STATE_REGION_PATTERN = _region_term_pattern(_OTHER_STATE_REGION_TERMS)


def _region_title_text(title: str) -> str:
    """Normalize a headline for the state filter.

    Punctuation becomes a space so "Delhi-Varanasi" still matches both cities.
    Channel descriptions often stamp "chhattisgarh" on every upload, so the
    decision uses the title only.
    """
    text = (title or "").lower()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for phrase, replacement in _REGION_TITLE_EXCEPTIONS:
        text = re.sub(phrase, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def video_is_cg_or_mp(title: str, description: str = "") -> bool:
    """True when the headline is not about a state other than CG or MP.

    Stories that name Uttar Pradesh or any other Indian state are removed.
    A headline that also names Chhattisgarh or Madhya Pradesh stays, as does
    a local or national headline that names no other state.
    """
    del description  # Title is the source of truth; descriptions carry channel tags.
    text = _region_title_text(title)
    if not text:
        return True
    if not _OTHER_STATE_REGION_PATTERN.search(text):
        return True
    return _CG_MP_REGION_PATTERN.search(text) is not None


def filter_to_cg_mp_news(feed: List[Dict]) -> List[Dict]:
    """Drop videos whose headline is about a state other than CG or MP."""
    kept: List[Dict] = []
    dropped = 0
    for video in feed:
        if video_is_cg_or_mp(video.get("title") or "", video.get("description") or ""):
            kept.append(video)
        else:
            dropped += 1
    print(f"[Region] Kept {len(kept)} videos, dropped {dropped} from other states")
    return kept


def filter_by_city(feed: List[Dict], city: str) -> List[Dict]:
    """Return videos whose title mentions the resolved city (not channel name or description alone)."""
    city_n = normalize(city or "")
    if not city_n:
        _log_city("City filter skipped: empty city after normalization")
        return []
    _log_city(f"Filtering feed by city='{city}' normalized='{city_n}', feed_size={len(feed)}")
    out: List[Dict] = []
    for v in feed:
        text = normalize(v.get('title', '') or '')
        if city_n and city_n in text:
            out.append(v)
    _log_city(f"City filter matched {len(out)} videos for city='{city}'")
    return out


def generate_ott_json(feed: List[Dict], request_ip: str = "", ip_lookup_url: str = "") -> Dict:
    """Generate OTT-style JSON feed."""
    _log_city(
        f"Starting OTT JSON generation with feed_size={len(feed)}, request_ip='{request_ip or '<none>'}'"
    )
    feed = filter_to_cg_mp_news(feed)
    city_section = None
    city = resolve_city_from_ip(request_ip, ip_lookup_url)
    if city:
        city_items = filter_by_city(feed, city)
        if city_items:
            city_section = {"section": city, "count": len(city_items), "items": city_items, "sectionIndex": 0}
            _log_city(f"Prepared city section '{city}' with {len(city_items)} items (index=0)")
        else:
            _log_city(f"Resolved city '{city}' but no matching videos found")
    else:
        _log_city("No city resolved, proceeding without city section")

    # Podcast section = genre Podcast only (assigned only for allow-listed channels + podcast signals)
    podcast_news = filter_by_genre(feed, [GENRE_PODCAST])
    # Scheduled streams have no row. They stay out of every other section too.
    other_feed = [v for v in feed if v.get("genre") not in (GENRE_PODCAST, GENRE_SCHEDULED)]
    
    live_news = filter_by_genre(other_feed, [GENRE_LIVE])
    general_news = filter_by_genre(other_feed, [GENRE_GENERAL])
    politics_news = filter_by_genre(other_feed, [GENRE_POLITICS])
    # One tab and one rail: crime and traffic together, newest first.
    negative_news = filter_by_genre(other_feed, [GENRE_CRIME, GENRE_TRAFFIC])
    negative_news.sort(key=lambda v: v.get("publishedAt") or "", reverse=True)
    jobs_news = filter_by_genre(other_feed, [GENRE_JOBS])
    events_news = filter_by_genre(other_feed, [GENRE_EVENTS])
    civic_news = filter_by_genre(other_feed, [GENRE_CIVIC])
    
    sections = [
        {"section": SECTION_LIVE, "count": len(live_news), "items": live_news},
        {"section": SECTION_GENERAL, "count": len(general_news), "items": general_news},
        {"section": SECTION_PODCAST, "count": len(podcast_news), "items": podcast_news},
        {"section": SECTION_JOBS, "count": len(jobs_news), "items": jobs_news},
        {"section": SECTION_POLITICS, "count": len(politics_news), "items": politics_news},
        {"section": SECTION_EVENTS, "count": len(events_news), "items": events_news},
        {"section": SECTION_CIVIC, "count": len(civic_news), "items": civic_news},
        {"section": SECTION_NEGATIVE, "count": len(negative_news), "items": negative_news},
    ]

    if city_section:
        sections = [city_section] + sections
        _log_city("City section prepended to sections list")

    # Attach sectionIndex to each section using SECTION_INDEX map; default to 0 if unknown
    for sec in sections:
        if sec.get("sectionIndex") == 0:
            continue
        sec_name = sec.get("section", "")
        sec["sectionIndex"] = SECTION_INDEX.get(sec_name, 0)
    _log_city(f"Completed section indexing; total_sections={len(sections)}")
    
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "sections": sections
    }


def main():
    """Main function to generate and save JSON."""
    should_update_file = True  # Flag to determine if we should update the output file
    
    if not API_KEYS:
        print(MSG_API_KEY_NOT_SET)
        print(MSG_KEEP_EXISTING)
        should_update_file = False
        json_data = {
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "error": ERR_API_KEY_NOT_CONFIGURED,
            "sections": []
        }
    elif not BHILAI_CHANNELS:
        print(MSG_CHANNELS_NOT_SET)
        print(MSG_SET_CHANNELS)
        should_update_file = False
        json_data = {
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "error": ERR_CHANNELS_NOT_CONFIGURED,
            "sections": []
        }
    else:
        print(f"Loaded {len(API_KEYS)} API key(s) for fallback support")
        print(f"Fetching videos from {len(BHILAI_CHANNELS)} Bhilai news channels...")
        feed = aggregate_bhilai_videos(BHILAI_CHANNELS)
        feed = filter_to_cg_mp_news(feed)
        feed = add_genres_to_feed(feed)
        request_ip = os.environ.get(ENV_REQUEST_IP, "").strip()
        ip_lookup_url = os.environ.get(ENV_IP_LOOKUP_URL, "").strip()
        _log_city(
            f"Runtime config: REQUEST_IP='{request_ip or '<none>'}', "
            f"IP_LOOKUP_URL='{ip_lookup_url or IP_LOOKUP_URL_DEFAULT}'"
        )
        json_data = generate_ott_json(feed, request_ip=request_ip, ip_lookup_url=ip_lookup_url)
        print(f"Fetched {len(feed)} total videos")
        
        # Check if we have any data
        if len(feed) == 0:
            print(MSG_NO_VIDEOS)
            print(MSG_PRESERVE_DATA)
            should_update_file = False
        
        if current_api_key_index > 0:
            print(f"Note: Used {current_api_key_index + 1} API key(s) due to quota limits")
    
    # Save to file only if we have valid data
    if should_update_file:
        with open(OUTPUT_FILE_NAME, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON data generated successfully and saved to {OUTPUT_FILE_NAME}")
    else:
        print(f"Skipped updating {OUTPUT_FILE_NAME} to preserve existing data")
    
    print(f"Generated at: {json_data['generatedAt']}")
    print(f"\nSections summary:")
    for section in json_data.get("sections", []):
        idx = section.get("sectionIndex", 0)
        print(f"  - {section['section']}: {section['count']} videos (index={idx})")


if __name__ == "__main__":
    main()
