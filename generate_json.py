#!/usr/bin/env python3
"""
YT Aggregator for Bhilai News - GitHub Actions Pipeline Version
Fetches latest videos from Bhilai news channels and generates JSON feed.
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import requests


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
ENV_CHANNEL_PROVIDERS = "CHANNEL_PROVIDERS"

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
MSG_SET_PROVIDERS = "Please set the CHANNEL_PROVIDERS environment variable with JSON configuration for provider-based channel control."

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

def load_channel_providers():
    """
    Load channel providers configuration from environment variable.
    Supports JSON format: {"provider1": {"channels": ["id1", "id2"], "maxChannels": 5}, ...}
    Returns dict with provider names as keys and their channel lists as values.
    """
    providers_env = os.environ.get(ENV_CHANNEL_PROVIDERS, "")
    if providers_env:
        try:
            providers_config = json.loads(providers_env)
            if isinstance(providers_config, dict):
                result = {}
                for provider_name, provider_data in providers_config.items():
                    if isinstance(provider_data, dict):
                        channels = provider_data.get("channels", [])
                        max_channels = provider_data.get("maxChannels", None)
                        
                        # Filter channels if maxChannels is specified
                        if max_channels is not None and isinstance(max_channels, int) and max_channels > 0:
                            channels = channels[:max_channels]
                        
                        if channels:
                            result[provider_name] = channels
                    elif isinstance(provider_data, list):
                        # Support simple list format: {"provider1": ["id1", "id2"]}
                        if provider_data:
                            result[provider_name] = provider_data
                
                if result:
                    return result
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[WARNING] Failed to parse CHANNEL_PROVIDERS JSON: {e}")
            print("[INFO] Falling back to BHILAI_CHANNELS format")
    
    return None

def load_bhilai_channels():
    """
    Load channel IDs from environment variable.
    First tries CHANNEL_PROVIDERS (provider-based), then falls back to BHILAI_CHANNELS (flat list).
    Returns list of channel IDs (flat list for backward compatibility).
    """
    # Try provider-based configuration first
    providers = load_channel_providers()
    if providers:
        # Flatten all provider channels into a single list
        all_channels = []
        for provider_name, channels in providers.items():
            print(f"[INFO] Provider '{provider_name}': {len(channels)} channel(s)")
            all_channels.extend(channels)
        return all_channels
    
    # Fallback to old comma-separated format
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
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
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


def classify_genre(title: str, description: str = "") -> str:
    """Classify video genre based on title and description with improved matching."""
    # Normalize title and description separately for better weighting
    title_text = normalize(title or "")
    desc_text = normalize(description or "")
    
    # Combine with title having more weight (appears 3 times)
    text = f"{title_text} {title_text} {title_text} {desc_text}"
    
    # Crime keywords (expanded list)
    crime_keywords = [
        # English
        "murder", "killed", "killing", "death", "dead", "died", "crime", "criminal",
        "loot", "robbery", "rob", "assault", "rape", "fraud", "arrest", "arrested",
        "police", "gang", "shoot", "shot", "stab", "knife", "weapon", "violence",
        "suspect", "accused", "jail", "prison", "court", "judge", "case", "investigation",
        # Hindi/Devanagari
        "हत्या", "मौत", "लाश", "शव", "अपराध", "अपराधी", "गिरफ्तार", "गिरफ्तारी",
        "पुलिस", "हत्या", "कातिल", "जेल", "कारागार", "अदालत", "जज", "मुकदमा",
        "जांच", "संदिग्ध", "आरोपी", "हिंसा", "चाकू", "बंदूक", "गोली"
    ]
    
    # Traffic keywords (expanded list)
    traffic_keywords = [
        # English
        "accident", "traffic", "jam", "collision", "crash", "vehicle", "car",
        "bus", "truck", "road", "highway", "injured", "injury", "ambulance",
        "blocked", "congestion", "lane", "signal", "intersection", "bridge",
        # Hindi/Devanagari
        "दुर्घटना", "हादसा", "जाम", "टक्कर", "वाहन", "कार", "बस", "ट्रक",
        "सड़क", "हाइवे", "घायल", "चोट", "एम्बुलेंस", "रुका", "भीड़", "लेन",
        "सिग्नल", "चौराहा", "पुल", "रफ्तार"
    ]
    
    # Politics keywords (expanded list)
    politics_keywords = [
        # English
        "election", "vote", "voting", "minister", "ministry", "mla", "mp", "cm",
        "chief minister", "government", "govt", "party", "political", "politician",
        "leader", "speech", "rally", "campaign", "candidate", "constituency",
        "parliament", "assembly", "budget", "policy", "scheme", "program",
        # Hindi/Devanagari
        "चुनाव", "वोट", "मतदान", "मंत्री", "मंत्रालय", "विधायक", "सांसद",
        "मुख्यमंत्री", "सरकार", "पार्टी", "राजनीति", "राजनीतिक", "नेता", "भाषण",
        "रैली", "अभियान", "उम्मीदवार", "निर्वाचन क्षेत्र", "संसद", "विधानसभा",
        "बजट", "नीति", "योजना", "कार्यक्रम"
    ]
    
    # Jobs keywords
    jobs_keywords = [
        # English
        "job", "jobs", "employment", "recruitment", "vacancy", "vacancies", "hiring",
        "interview", "career", "opportunity", "opportunities", "post", "posts",
        "notification", "admit card", "result", "exam", "examination", "syllabus",
        "apply", "application", "form", "salary", "wage", "wages", "position",
        "opening", "openings", "govt job", "government job", "sarkari naukri",
        "walk in", "walk-in", "bharti", "bharte", "roster", "merit", "cutoff",
        # Hindi/Devanagari
        "नौकरी", "रोजगार", "भर्ती", "रिक्ति", "रिक्तियां", "भरती", "भर्ती",
        "इंटरव्यू", "करियर", "अवसर", "पद", "पदों", "सूचना", "एडमिट कार्ड",
        "रिजल्ट", "परिणाम", "परीक्षा", "सिलेबस", "आवेदन", "फॉर्म", "वेतन",
        "पदस्थापना", "सरकारी नौकरी", "सरकारी रोजगार", "भर्ती", "मेरिट", "कटऑफ"
    ]
    
    # Events keywords
    events_keywords = [
        # English
        "event", "events", "festival", "celebration", "ceremony", "function",
        "program", "programme", "conference", "meeting", "gathering", "fair",
        "exhibition", "show", "concert", "performance", "inauguration", "launch",
        "anniversary", "birthday", "wedding", "marriage", "fest", "mela", "utsav",
        "kumbh", "mela", "jatra", "yatra", "puja", "aarti", "pooja", "prayer",
        "cultural", "religious", "traditional", "commemoration", "memorial",
        # Hindi/Devanagari
        "कार्यक्रम", "उत्सव", "त्योहार", "समारोह", "सम्मेलन", "बैठक", "मेला",
        "प्रदर्शनी", "शो", "संगीत", "प्रदर्शन", "उद्घाटन", "शुभारंभ", "जन्मदिन",
        "शादी", "विवाह", "मेला", "यात्रा", "पूजा", "आरती", "प्रार्थना", "सांस्कृतिक",
        "धार्मिक", "परंपरागत", "स्मरण", "महाकुंभ", "कुंभ", "जलसा"
    ]
    
    # Civic keywords (civic services, public services, community issues)
    civic_keywords = [
        # English
        "civic", "municipal", "municipality", "corporation", "municipal corporation",
        "water supply", "electricity", "power", "street light", "road", "street",
        "drainage", "sewage", "garbage", "waste", "cleanliness", "swachh", "swachhta",
        "public health", "sanitation", "pothole", "repair", "maintenance", "development",
        "infrastructure", "public works", "pwd", "public welfare", "community", "ward",
        "councilor", "mayor", "commissioner", "complaint", "grievance", "petition",
        "tax", "property tax", "house tax", "license", "permit", "certificate",
        "birth certificate", "death certificate", "marriage certificate", "ration",
        "aadhaar", "pan card", "voter id", "driving license", "passport", "service",
        "public service", "citizen service", "csc", "common service center",
        # Hindi/Devanagari
        "नगर निगम", "नगर पालिका", "नगर पंचायत", "पानी", "जल", "बिजली", "सड़क",
        "गली", "नाली", "नालियां", "कचरा", "सफाई", "स्वच्छ", "स्वच्छता", "सार्वजनिक स्वास्थ्य",
        "स्वच्छता", "गड्ढा", "मरम्मत", "रखरखाव", "विकास", "बुनियादी ढांचा", "निर्माण",
        "सार्वजनिक कार्य", "समुदाय", "वार्ड", "पार्षद", "मेयर", "कमिश्नर", "शिकायत",
        "कर", "संपत्ति कर", "घर कर", "लाइसेंस", "परमिट", "प्रमाणपत्र", "जन्म प्रमाणपत्र",
        "मृत्यु प्रमाणपत्र", "विवाह प्रमाणपत्र", "राशन", "आधार", "पैन कार्ड", "मतदाता पहचान पत्र",
        "ड्राइविंग लाइसेंस", "पासपोर्ट", "सेवा", "सार्वजनिक सेवा", "नागरिक सेवा"
    ]
    
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
    
    crime_patterns = create_patterns(crime_keywords)
    traffic_patterns = create_patterns(traffic_keywords)
    politics_patterns = create_patterns(politics_keywords)
    jobs_patterns = create_patterns(jobs_keywords)
    events_patterns = create_patterns(events_keywords)
    civic_patterns = create_patterns(civic_keywords)
    
    # Check each category with pattern matching (order matters - more specific first)
    # Check Crime first (most specific)
    for pattern in crime_patterns:
        if pattern.search(text):
            return GENRE_CRIME
    
    # Check Traffic
    for pattern in traffic_patterns:
        if pattern.search(text):
            return GENRE_TRAFFIC
    
    # Check Jobs
    for pattern in jobs_patterns:
        if pattern.search(text):
            return GENRE_JOBS
    
    # Check Events
    for pattern in events_patterns:
        if pattern.search(text):
            return GENRE_EVENTS
    
    # Check Civic
    for pattern in civic_patterns:
        if pattern.search(text):
            return GENRE_CIVIC
    
    # Check Politics (after civic to avoid overlap with civic services)
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
    video_ids = [item["id"]["videoId"] for item in items]
    video_details = fetch_video_details(video_ids)
    
    videos: List[Dict] = []
    for item in items:
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]
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
    """Add genre classification to each video."""
    for v in feed:
        video_type = (v.get("videoType") or "").strip().upper()
        
        if video_type == VIDEO_TYPE_LIVE:
            v["genre"] = GENRE_LIVE
        elif VIDEO_TYPE_SCHEDULED in video_type or "UPCOMING" in video_type:
            v["genre"] = GENRE_SCHEDULED
        else:
            v["genre"] = classify_genre(v.get("title", ""), v.get("description", ""))
    
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
        providers = load_channel_providers()
        if providers:
            print(f"[INFO] CHANNEL_PROVIDERS format detected with {len(providers)} provider(s)")
            print(MSG_SET_CHANNELS)
        else:
            print(MSG_SET_CHANNELS)
        should_update_file = False
        json_data = {
            "generatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "error": ERR_CHANNELS_NOT_CONFIGURED,
            "sections": []
        }
    else:
        print(f"Loaded {len(API_KEYS)} API key(s) for fallback support")
        providers = load_channel_providers()
        if providers:
            print(f"[INFO] Using provider-based channel configuration: {len(providers)} provider(s)")
            for provider_name, channels in providers.items():
                print(f"  - {provider_name}: {len(channels)} channel(s)")
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
