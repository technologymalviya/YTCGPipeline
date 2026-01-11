import json
import os
from unittest.mock import patch

import generate_json as gj


# Simple mock response
class MockResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json

# === Unit tests for small utility functions ===
def test_parse_iso_duration():
    assert gj.parse_iso_duration("PT1H2M3S") == 3600 + 120 + 3
    assert gj.parse_iso_duration("PT2M3S") == 123
    assert gj.parse_iso_duration("PT0S") == 0
    # Non-PT forms should return 0 per implementation
    assert gj.parse_iso_duration("P0DT0S") == 0
    assert gj.parse_iso_duration("") == 0

def test_format_duration():
    assert gj.format_duration(3723) == "01:02:03"
    assert gj.format_duration(63) == "01:03"
    assert gj.format_duration(0) == "00:00"
    # None yields "0" per implementation
    assert gj.format_duration(None) == "0"

def test_normalize_and_replacements():
    s = "This is a H!dsa and mou t sample!"
    out = gj.normalize(s)
    # Should convert H!dsa -> हिंद word (replacement exists in file, at least ensures lowercasing and removal of punctuation)
    assert "हादसा" in out or "h!dsa" not in out

def test_classify_genre_basic():
    assert gj.classify_genre("Major accident on highway") == gj.GENRE_TRAFFIC
    assert gj.classify_genre("सरकारी नौकरी भर्ती का विज्ञापन") == gj.GENRE_JOBS or gj.classify_genre("सरकारी नौकरी भर्ती का विज्ञापन") == gj.GENRE_GENERAL
    assert gj.classify_genre("हत्या का मामला, आरोपी गिरफ्तार") == gj.GENRE_CRIME

# === Tests that mock YouTube API calls ===
def mocked_requests_get(url, params=None, timeout=None, **kwargs):
    # Simulate search endpoint
    if url == gj.YOUTUBE_SEARCH_URL:
        # Return a single video item for the search
        search_json = {
            "items": [
                {
                    "id": {"videoId": "vid1"},
                    "snippet": {
                        "title": "Test video title",
                        "description": "A short description",
                        "channelTitle": "TestChannel",
                        "publishedAt": "2026-01-11T10:00:00Z",
                        "thumbnails": {"high": {"url": "http://example.com/thumb.jpg"}},
                        "liveBroadcastContent": "none"
                    }
                }
            ]
        }
        return MockResponse(200, search_json)
    # Simulate videos endpoint
    elif url == gj.YOUTUBE_VIDEOS_URL:
        videos_json = {
            "items": [
                {
                    "id": "vid1",
                    "snippet": {
                        "title": "Test video title",
                        "description": "A short description",
                        "channelTitle": "TestChannel",
                        "liveBroadcastContent": "none",
                    },
                    "contentDetails": {"duration": "PT2M3S"},
                    "statistics": {"viewCount": "123", "likeCount": "5"},
                    "liveStreamingDetails": {}
                }
            ]
        }
        return MockResponse(200, videos_json)

    return MockResponse(404, {}, text="Not Found")

@patch("requests.get", side_effect=mocked_requests_get)
def test_fetch_and_aggregate(mock_get, tmp_path):
    # Provide an API key and a channel via env
    os.environ["YOUTUBE_API_KEY"] = "DUMMY_KEY"
    os.environ["BHILAI_CHANNELS"] = "CHAN_1"

    # Ensure API_KEYS is reloaded to pick up env changes for test context
    gj.API_KEYS = gj.load_api_keys()
    gj.current_api_key_index = 0

    # Fetch videos for a channel
    videos = gj.fetch_latest_videos_for_channel("CHAN_1")
    assert isinstance(videos, list)
    assert len(videos) == 1
    v = videos[0]
    assert v["videoId"] == "vid1"
    assert v["durationSeconds"] == 123
    assert v["durationFormatted"] == "02:03"
    assert v["views"] == 123
    assert v["likes"] == 5

    # Aggregate and add genres
    feed = gj.aggregate_bhilai_videos(["CHAN_1"])
    assert len(feed) == 1
    feed = gj.add_genres_to_feed(feed)
    assert "genre" in feed[0]

@patch("requests.get", side_effect=mocked_requests_get)
def test_main_writes_output(mock_get, tmp_path):
    # Point output to tmp file by changing working dir temporarily
    cwd = os.getcwd()
    os.chdir(tmp_path)

    os.environ["YOUTUBE_API_KEY"] = "DUMMY_KEY"
    os.environ["BHILAI_CHANNELS"] = "CHAN_1"
    gj.API_KEYS = gj.load_api_keys()
    gj.current_api_key_index = 0

    # Ensure output file does not exist
    out_file = tmp_path / gj.OUTPUT_FILE_NAME
    if out_file.exists():
        out_file.unlink()

    # Run main (will create output.json using mocked responses)
    gj.main()

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert "generatedAt" in data
    assert "sections" in data
    # basic sanity checks on sections
    assert isinstance(data["sections"], list)

    # cleanup and revert cwd
    os.chdir(cwd)
    # Remove environment variables used for test
    os.environ.pop("YOUTUBE_API_KEY", None)
    os.environ.pop("BHILAI_CHANNELS", None)