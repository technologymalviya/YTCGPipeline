#!/usr/bin/env python3
"""
YT Aggregator for Bhilai News - GitHub Actions Pipeline Version
Fetches latest videos from Bhilai news channels and generates JSON feed.
"""

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
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

# OpenAI Configuration
OPENAI_MODEL = "gpt-4.1-mini"  # Cost-effective model
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

# Rate limiting tracking
_openai_rate_limit_tracker = {
    "requests": [],
    "tokens": [],
    "last_reset": time.time()
}

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
GENRE_GENERAL = "General"

# Section Names
SECTION_LIVE = "Live"
SECTION_SCHEDULED = "Scheduled"
SECTION_CRIME = "Crime"
SECTION_TRAFFIC = "Traffic"
SECTION_POLITICS = "Politics"
SECTION_JOBS = "Jobs"
SECTION_EVENTS = "Events"
SECTION_CIVIC = "Civic"
SECTION_GENERAL = "General"

# Section -> Index map (per requested mapping)
# Mapping provided: "General" 1 , "Jobs" 2, "Politics" 3, "Events" 4, "Civic" 5, "Traffic" 6, "Crime" 7
SECTION_INDEX = {
    "General": 1,
    "Jobs": 2,
    "Politics": 3,
    "Events": 4,
    "Civic": 5,
    "Traffic": 6,
    "Crime": 7,
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
        "lash": "लाश",
        "hatya": "हत्या",
    }
    
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    return text


# Create word boundary patterns for better matching
def create_patterns(keywords):
    """Create regex patterns with word boundaries for better matching."""
    patterns = []
    for kw in keywords:
        # Use word boundaries for English, space boundaries for Hindi
        if re.search(r'[\u0900-\u097F]', kw):
            # Hindi/Devanagari - use space or start/end of string
            pattern = r'(?:^|\s)' + re.escape(kw) + r'(?:\s|$)'
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
- Crime: Criminal activities, arrests, violence, murders, thefts, police cases
- Traffic: Road accidents, traffic jams, vehicle collisions, highway incidents
- Jobs: Employment opportunities, job notifications, recruitment, interviews, exams
- Events: Festivals, ceremonies, celebrations, inaugurations, cultural events
- Civic: Municipal services, civic issues, government services, certificates, utilities
- Politics: Political news, elections, government announcements, political rallies, CM/PM speeches
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
            return openai_result
    
    # Use keyword-based classification for older videos or if OpenAI fails
    keyword_result = classify_genre_keyword_based(title, description)
    if not use_openai or not published_at or not is_recent_video(published_at, hours=1):
        # Only print for non-recent videos to avoid spam
        print(f"[Keyword] Classification result: {keyword_result}")
    return keyword_result


def classify_genre_keyword_based(title: str, description: str = "") -> str:
    """Classify video genre based on title and description with improved accuracy using keywords."""
    # Normalize title and description separately - title gets more weight
    title_text = normalize(title or "")
    desc_text = normalize(description or "")
    
    # Combine with title weighted 3x more than description (titles are more indicative)
    text = f"{title_text} {title_text} {title_text} {desc_text}"
    
    # Crime keywords - specific and strong indicators
    crime_keywords = [
        # English - specific crime terms
        "murder", "killed", "killing", "homicide", "assassination",
        "robbery", "loot", "theft", "steal", "stolen",
        "assault", "rape", "sexual assault",
        "arrest", "arrested", "suspect", "accused", "criminal",
        "gang", "mob", "violence", "rioting",
        "shoot", "shot", "firing", "gunfire", "bullets",
        "stab", "knife attack", "weapon",
        "jail", "prison", "court case", "trial", "investigation",
        "kidnapping", "abduction", "abducted",
        # Hindi/Devanagari - specific crime terms
        "हत्या", "कत्ल", "लाश", "शव", "मौत", "अपराध", "अपराधी",
        "गिरफ्तार", "गिरफ्तारी", "संदिग्ध", "आरोपी",
        "पुलिस", "कातिल", "जेल", "कारागार", "अदालत", "जज", "मुकदमा",
        "हिंसा", "दंगा", "चाकू", "बंदूक", "गोली", "फायरिंग",
        "अपहरण",  # Kidnapping
        # Context-specific: जांच only when combined with crime terms
        "अपराध जांच", "पुलिस जांच", "हत्या जांच", "मुकदमा जांच"
    ]
    
    # Traffic keywords - specific traffic/accident terms
    traffic_keywords = [
        # English - multi-word phrases for stronger matching (check first)
        "traffic accident", "road accident", "car accident", "vehicle accident",
        "traffic jam", "road jam",
        "head-on collision", "rear-end collision",
        "fatal accident", "deadly accident",
        # Hindi/Devanagari - multi-word phrases (check first)
        "सड़क दुर्घटना", "कार हादसा", "वाहन हादसा",
        "ट्रैफिक जाम", "सड़क जाम",
        # English - strong single-word indicators (context-specific)
        "accident", "collision", "crash", "traffic", "jam",
        "ambulance", "rescue", "highway", "expressway",
        # Hindi/Devanagari - strong single-word indicators (context-specific)
        "दुर्घटना", "हादसा", "टक्कर", "जाम", "एम्बुलेंस"
        # Note: "घायल" (injured) removed - too generic, appears in non-traffic contexts
    ]
    
    # Jobs keywords - specific employment/job terms
    jobs_keywords = [
        # English - strong single-word indicators
        "job", "jobs", "recruitment", "vacancy", "hiring", "employment", "bharti", "bharte",
        "career", "opportunity", "post", "application", "opening",
        "admit card", "merit", "salary", "wage", "exam",
        # English - multi-word phrases for stronger matching
        "job notification", "job vacancy", "job opening", "job opportunity",
        "government job", "sarkari naukri", "govt job",
        "job application", "apply for job",
        "exam result", "merit list",
        "walk-in interview", "job interview", "employment interview",
        "pay scale", "vacancy notification",
        # Hindi/Devanagari - strong single-word indicators
        "नौकरी", "रोजगार", "भर्ती", "आवेदन", "एडमिट कार्ड",
        "रिजल्ट", "मेरिट", "परीक्षा", "वेतन",
        # Hindi/Devanagari - multi-word phrases (context-specific)
        "सरकारी नौकरी", "नौकरी सूचना",
        "नौकरी इंटरव्यू", "भर्ती इंटरव्यू", "रोजगार इंटरव्यू"
    ]

    # Events keywords - specific event/festival terms
    events_keywords = [
        # English - strong single-word indicators
        "event", "events", "festival", "celebration", "ceremony", "fair", "mela", "exhibition",
        "inauguration", "wedding", "anniversary", "birthday",
        "kumbh", "jatra", "yatra", "puja", "aarti", "pooja",
        "republic day", "independence day", "26 january", "15 august",
        # English - multi-word phrases for stronger matching
        "cultural event", "religious event",
        "launch ceremony", "marriage ceremony",
        "birthday celebration",
        # Hindi/Devanagari - strong single-word indicators
        "त्योहार", "उत्सव", "समारोह", "मेला", "प्रदर्शनी",
        "उद्घाटन", "शुभारंभ", "शादी", "विवाह", "जन्मदिन",
        "यात्रा", "पूजा", "आरती", "जुलूस",
        "गणतंत्र दिवस", "26 जनवरी", "स्वतंत्रता दिवस", "15 अगस्त"
    ]

    # Civic keywords - specific civic/municipal service terms
    civic_keywords = [
        # English - specific civic terms
        "municipal corporation", "municipality", "municipal",
        "water supply", "electricity supply", "power supply",
        "garbage collection", "waste management", "cleanliness drive",
        "pothole repair", "road repair", "street repair",
        "road construction", "road not built", "infrastructure",
        "public health", "sanitation", "swachh bharat",
        "property tax", "house tax", "municipal tax",
        "birth certificate", "death certificate", "marriage certificate",
        "aadhaar card", "pan card", "voter id card",
        "driving license", "passport",
        "csc", "common service center",
        "mayor", "commissioner", "councilor",
        "nrda",  # NRDA - Raipur Development Authority
        # Hindi/Devanagari - specific civic terms
        "नगर निगम", "नगर पालिका",
        "पानी की समस्या", "बिजली की समस्या",
        "कचरा", "सफाई", "स्वच्छता", "गड्ढा",
        "मरम्मत", "शिकायत", "संपत्ति कर",
        "अतिक्रमण", "अतिक्रमण हटाने",  # Encroachment removal
        "सड़क नहीं बनी", "सड़क निर्माण",  # Road construction
        "जन्म प्रमाणपत्र", "मृत्यु प्रमाणपत्र", "विवाह प्रमाणपत्र",
        "आधार", "पैन कार्ड", "मतदाता पहचान",
        "ड्राइविंग लाइसेंस", "पासपोर्ट",
        "मेयर", "कमिश्नर", "पार्षद",
        # Safety inspections and administrative checks
        "सुरक्षा जांच", "स्कूल बस जांच", "वाहन जांच", "सार्वजनिक जांच"
    ]
    
    # Politics keywords - specific political terms
    politics_keywords = [
        # English - specific political terms
        "election", "voting", "vote", "election campaign",
        "minister", "chief minister", "cm", "mla", "mp",
        "prime minister", "pm modi", "narendra modi", "modi",
        "government", "govt", "political party",
        "political rally", "political speech", "political news",
        "budget", "assembly session", "parliament",
        "political leader", "politician",
        "councilor", "councillor", "alderman",
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
        "पार्षद"  # Councilor - political position
    ]

    # Create patterns and check in order of specificity
    crime_patterns = create_patterns(crime_keywords)
    traffic_patterns = create_patterns(traffic_keywords)
    jobs_patterns = create_patterns(jobs_keywords)
    events_patterns = create_patterns(events_keywords)
    civic_patterns = create_patterns(civic_keywords)
    politics_patterns = create_patterns(politics_keywords)

    # Check each category - order matters (most specific first)
    # Traffic - check BEFORE Crime (road accidents can have deaths but should be Traffic)
    for pattern in traffic_patterns:
        if pattern.search(text):
            return GENRE_TRAFFIC

    # Crime - specific crime terms (check after Traffic to avoid false positives)
    for pattern in crime_patterns:
        if pattern.search(text):
            return GENRE_CRIME

    # Jobs - employment opportunities
    for pattern in jobs_patterns:
        if pattern.search(text):
            return GENRE_JOBS

    # Events - festivals/ceremonies
    for pattern in events_patterns:
        if pattern.search(text):
            return GENRE_EVENTS

    # Civic - municipal services
    for pattern in civic_patterns:
        if pattern.search(text):
            return GENRE_CIVIC

    # Politics - political activities (check last to avoid overlap)
    for pattern in politics_patterns:
        if pattern.search(text):
            return GENRE_POLITICS
    
    return GENRE_GENERAL


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
                    v["genre"] = classify_genre_keyword_based(title, description)
                    keyword_count += 1
                else:
                    openai_result = classify_genre_with_openai(title, description)
                    if openai_result:
                        v["genre"] = openai_result
                        openai_count += 1
                    else:
                        # OpenAI failed - set circuit breaker and use keyword-based
                        if not openai_failed:
                            print(f"[Classification] OpenAI failed, circuit breaker activated - using keyword-based for remaining {total_videos - i} videos")
                            openai_failed = True
                        v["genre"] = classify_genre_keyword_based(title, description)
                        keyword_count += 1
            else:
                # Use keyword-based directly for older videos or if circuit breaker is active
                v["genre"] = classify_genre_keyword_based(title, description)
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


def generate_ott_json(feed: List[Dict]) -> Dict:
    """Generate OTT-style JSON feed."""
    live_news = filter_by_genre(feed, [GENRE_LIVE])
    scheduled_news = filter_by_genre(feed, [GENRE_SCHEDULED])
    general_news = filter_by_genre(feed, [GENRE_GENERAL])
    politics_news = filter_by_genre(feed, [GENRE_POLITICS])
    traffic_news = filter_by_genre(feed, [GENRE_TRAFFIC])
    crime_news = filter_by_genre(feed, [GENRE_CRIME])
    jobs_news = filter_by_genre(feed, [GENRE_JOBS])
    events_news = filter_by_genre(feed, [GENRE_EVENTS])
    civic_news = filter_by_genre(feed, [GENRE_CIVIC])
    
    sections = [
        {"section": SECTION_LIVE, "count": len(live_news), "items": live_news},
        {"section": SECTION_SCHEDULED, "count": len(scheduled_news), "items": scheduled_news},
        {"section": SECTION_GENERAL, "count": len(general_news), "items": general_news},
        {"section": SECTION_JOBS, "count": len(jobs_news), "items": jobs_news},
        {"section": SECTION_POLITICS, "count": len(politics_news), "items": politics_news},
        {"section": SECTION_EVENTS, "count": len(events_news), "items": events_news},
        {"section": SECTION_CIVIC, "count": len(civic_news), "items": civic_news},
        {"section": SECTION_TRAFFIC, "count": len(traffic_news), "items": traffic_news},
        {"section": SECTION_CRIME, "count": len(crime_news), "items": crime_news},
    ]

    # Attach sectionIndex to each section using SECTION_INDEX map; default to 0 if unknown
    for sec in sections:
        sec_name = sec.get("section", "")
        sec["sectionIndex"] = SECTION_INDEX.get(sec_name, 0)
    
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
        feed = add_genres_to_feed(feed)
        json_data = generate_ott_json(feed)
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
