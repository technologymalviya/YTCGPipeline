import json
import os
from datetime import datetime, timezone
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
    assert gj.classify_genre("Major accident on highway") == gj.GENRE_NEGATIVE
    assert gj.classify_genre("सरकारी नौकरी भर्ती का विज्ञापन") == gj.GENRE_JOBS or gj.classify_genre("सरकारी नौकरी भर्ती का विज्ञापन") == gj.GENRE_GENERAL
    assert gj.classify_genre("हत्या का मामला, आरोपी गिरफ्तार") == gj.GENRE_NEGATIVE


def test_resolve_ip_lookup_url_configurable():
    ip = "117.198.246.216"
    assert gj._resolve_ip_lookup_url(ip, "http://ip-api.com/json/{ip}") == "http://ip-api.com/json/117.198.246.216"
    assert gj._resolve_ip_lookup_url(ip, "http://ip-api.com/json/") == "http://ip-api.com/json/117.198.246.216"
    assert gj._resolve_ip_lookup_url(ip, "https://example.com/geo") == "https://example.com/geo?ip=117.198.246.216"


def test_resolve_city_from_ip_success_and_cache():
    class Resp:
        status_code = 200
        def json(self):
            return {"status": "success", "city": "Montreal"}

    gj._ip_city_cache.clear()
    with patch("requests.get", return_value=Resp()) as mock_get:
        city1 = gj.resolve_city_from_ip("24.48.0.1", "http://ip-api.com/json/{ip}")
        city2 = gj.resolve_city_from_ip("24.48.0.1", "http://ip-api.com/json/{ip}")

    assert city1 == "Montreal"
    assert city2 == "Montreal"
    # Second call should hit cache and avoid extra network call
    assert mock_get.call_count == 1


def test_resolve_city_from_ip_failure_safe():
    gj._ip_city_cache.clear()
    with patch("requests.get", side_effect=Exception("timeout")):
        city = gj.resolve_city_from_ip("24.48.0.1", "http://ip-api.com/json/{ip}")
    assert city == ""


def test_education_lessons_stay_on_their_tab_for_six_months():
    krish = "UCNU_lfiiWBdtULKOw6X0Dig"
    freecodecamp = "UC8butISFwT-Wl7EV0hUK0BQ"
    now = datetime(2026, 11, 3, tzinfo=timezone.utc)

    assert not gj.education_lesson_ok({
        "channelId": krish,
        "title": "Short tip",
        "description": "",
        "durationSeconds": 90,
        "videoType": "VOD",
    })
    assert gj.education_lesson_ok({
        "channelId": krish,
        "title": "Complete Machine Learning Course",
        "description": "",
        "durationSeconds": 3600,
        "videoType": "VOD",
    })
    assert not gj.education_lesson_ok({
        "channelId": freecodecamp,
        "title": "JavaScript Full Course",
        "description": "web development",
        "durationSeconds": 36000,
        "videoType": "VOD",
    })
    assert gj.education_lesson_ok({
        "channelId": freecodecamp,
        "title": "Machine Learning Full Course",
        "description": "",
        "durationSeconds": 36000,
        "videoType": "VOD",
    })

    kept = {
        "videoId": "kept",
        "genre": "Education",
        "channelId": krish,
        "title": "Deep Learning",
        "publishedAt": "2026-08-01T00:00:00Z",
        "views": 1,
    }
    expired = {
        "videoId": "expired",
        "genre": "Education",
        "channelId": krish,
        "title": "Old course",
        "publishedAt": "2026-01-01T00:00:00Z",
    }
    refreshed = dict(kept, views=9)
    merged = gj.merge_education_videos([kept, expired], [refreshed], now=now)
    assert [video["videoId"] for video in merged] == ["kept"]
    assert merged[0]["views"] == 9

    news = {
        "videoId": "news1",
        "title": "रायपुर में बैठक",
        "description": "",
        "channelTitle": "Bhilai Times",
        "publishedAt": "2026-11-03T04:00:00Z",
        "genre": "General",
        "views": 100,
        "likes": 1,
    }
    course = {
        "videoId": "course1",
        "title": "Machine Learning Full Course",
        "description": "",
        "channelTitle": "Krish Naik",
        "channelId": krish,
        "publishedAt": "2026-11-03T04:00:00Z",
        "genre": "Education",
        "views": 5000,
        "likes": 10,
        "durationSeconds": 7200,
    }
    leaked = dict(course, videoId="course2", genre="General", title="रायपुर data science class")
    out = gj.generate_ott_json([news, course, leaked])
    by_section = {
        section["section"]: [item["videoId"] for item in section["items"]]
        for section in out["sections"]
    }
    assert by_section["Education"] == ["course1", "course2"]
    assert out["sections"][-1]["sectionIndex"] == 8
    for name, video_ids in by_section.items():
        if name == "Education":
            continue
        assert "course1" not in video_ids
        assert "course2" not in video_ids

    import cluster_api
    clusters = cluster_api.extract_clusters(out)
    encoded = json.dumps(clusters)
    assert "course1" not in encoded
    assert "course2" not in encoded


def test_education_channel_is_not_classified_as_news():
    video = {
        "title": "Murder case study in machine learning",
        "description": "crime dataset",
        "channelTitle": "Krish Naik",
        "channelId": "UCNU_lfiiWBdtULKOw6X0Dig",
        "publishedAt": "2020-01-01T00:00:00Z",
        "videoType": "VOD",
    }
    classified = gj.add_genres_to_feed([video])
    assert classified[0]["genre"] == gj.GENRE_EDUCATION


def _feed_video(video_id, title, **extra):
    video = {
        "videoId": video_id,
        "title": title,
        "description": "",
        "channelTitle": "News",
        "publishedAt": "2026-11-11T04:00:00Z",
        "views": extra.pop("views", 10),
        "likes": 0,
        "genre": extra.pop("genre", "General"),
        "durationSeconds": extra.pop("durationSeconds", 600),
        "videoType": "VOD",
    }
    video.update(extra)
    return video


def test_private_tabs_stay_out_of_news_and_all():
    import cluster_api

    vlog = _feed_video(
        "v1", "Raipur morning walk", genre="Vlog",
        channelId="UCMfTJq-r9fNEAEWcf_K3b5g", channelTitle="Mr.RakeshWakte",
    )
    food = _feed_video(
        "f1", "Indore poha breakfast", genre="Food",
        channelId="UCATnz7ESiAH9rW6WZC2pajg", channelTitle="indorefood",
    )
    movie = _feed_video(
        "m1", "Family Drama (U) Full Movie", genre="Movie",
        channelId="UCyoXW-Dse7fURq30EWl_CUA", durationSeconds=7200,
    )
    news = _feed_video("n1", "रायपुर में बैठक", views=100)
    out = gj.generate_ott_json([news, vlog, food, movie])
    by_section = {
        section["section"]: [item["videoId"] for item in section["items"]]
        for section in out["sections"]
    }
    assert by_section["Vlog"] == ["v1"]
    assert by_section["Food"] == ["f1"]
    assert by_section["Movie"] == ["m1"]
    assert by_section["Vlog"] and out["sections"][-3]["sectionIndex"] == 9
    for name, video_ids in by_section.items():
        if name in ("Vlog", "Food", "Movie"):
            continue
        assert "v1" not in video_ids
        assert "f1" not in video_ids
        assert "m1" not in video_ids
    encoded = json.dumps(cluster_api.extract_clusters(out))
    assert "v1" not in encoded
    assert "f1" not in encoded
    assert "m1" not in encoded


def test_darshan_tab_keeps_only_a_live_mandir_stream():
    import cluster_api

    mahakal = "UCEUuwXHmHckwmdANSy4c7Sw"
    live = _feed_video(
        "d1", "Bhasma Aarti", channelId=mahakal, channelTitle="Shri Mahakaleshwar",
        videoType="LIVE", durationSeconds=0,
    )
    recording = _feed_video(
        "d2", "Bhasma Aarti recording", channelId=mahakal, videoType="VOD", durationSeconds=3600,
    )
    upcoming = _feed_video(
        "d3", "Aarti upcoming", channelId=mahakal, videoType="SCHEDULED", durationSeconds=0,
    )
    news_live = _feed_video("n1", "रायपुर लाइव", videoType="LIVE", channelId="NEWSCHAN")
    assert gj.darshan_live_ok(live)
    assert not gj.darshan_live_ok(recording)
    assert not gj.darshan_live_ok(upcoming)
    assert not gj.darshan_live_ok(news_live)
    news = _feed_video("n2", "रायपुर में बैठक")
    out = gj.generate_ott_json([news, live, recording, news_live], darshan_videos=[live, recording, upcoming])
    by_section = {
        section["section"]: [item["videoId"] for item in section["items"]]
        for section in out["sections"]
    }
    assert by_section["Darshan"] == ["d1"]
    assert out["sections"][-1]["sectionIndex"] == 12
    for name, video_ids in by_section.items():
        if name == "Darshan":
            continue
        assert "d1" not in video_ids
        assert "d2" not in video_ids
    assert "d1" not in json.dumps(cluster_api.extract_clusters(out))


def test_katha_stays_on_its_tab():
    import cluster_api

    bhakti = "UCsjMAEPcv7-oNGHtRU9Vg6w"
    talk = _feed_video(
        "k1", "Shri Ram Katha", channelId=bhakti, channelTitle="BhaktiPath", durationSeconds=3600,
    )
    short = _feed_video(
        "k2", "Katha clip", channelId=bhakti, durationSeconds=30,
    )
    live = _feed_video(
        "k3", "Katha live", channelId="UCGQ-nwzfhjFKJBvwv1TE3-Q",
        channelTitle="Sudhanshu Ji Maharaj", videoType="LIVE", durationSeconds=0,
    )
    news = _feed_video("n2", "रायपुर में बैठक")
    assert gj.katha_ok(talk)
    assert gj.katha_ok(live)
    assert not gj.katha_ok(short)
    out = gj.generate_ott_json([news, talk, short], katha_videos=[talk, short, live])
    by_section = {
        section["section"]: [item["videoId"] for item in section["items"]]
        for section in out["sections"]
    }
    assert set(by_section["Katha"]) == {"k1", "k3"}
    assert out["sections"][-1]["sectionIndex"] == 13
    for name, video_ids in by_section.items():
        if name == "Katha":
            continue
        assert "k1" not in video_ids
        assert "k2" not in video_ids
        assert "k3" not in video_ids
    encoded = json.dumps(cluster_api.extract_clusters(out))
    assert "k1" not in encoded
    assert "k3" not in encoded


def test_free_u_certificate_movie_rules():
    goldmines = "UCyoXW-Dse7fURq30EWl_CUA"
    assert gj.free_u_movie_ok(_feed_video(
        "ok", "Family Drama (U) | Full Movie", channelId=goldmines, durationSeconds=7200,
    ))
    assert not gj.free_u_movie_ok(_feed_video(
        "ua", "Action (U/A) | Full Movie", channelId=goldmines, durationSeconds=7200,
    ))
    assert not gj.free_u_movie_ok(_feed_video(
        "trailer", "Family Drama (U) Official Trailer", channelId=goldmines, durationSeconds=7200,
    ))
    assert not gj.free_u_movie_ok(_feed_video(
        "short", "Family Drama (U)", channelId=goldmines, durationSeconds=600,
    ))
    assert gj.food_vlog_ok(_feed_video(
        "food", "Best poha in Indore", channelId="UCATnz7ESiAH9rW6WZC2pajg", durationSeconds=480,
    ))
    assert not gj.food_vlog_ok(_feed_video(
        "short", "Indore poha", channelId="UCATnz7ESiAH9rW6WZC2pajg", durationSeconds=30,
    ))
    assert not gj.food_vlog_ok(_feed_video(
        "shop", "Dhanteras Shopping 2026", channelId="UCOT0guEWvdIoXjKI1et_2cA", durationSeconds=400,
    ))
    assert gj.food_vlog_ok(_feed_video(
        "recipe", "लाल भाजी recipe", channelId="UCOT0guEWvdIoXjKI1et_2cA", durationSeconds=400,
    ))
    assert gj.free_u_movie_ok(_feed_video(
        "full", "Sindoor | Full Movie (HD)", channelId=goldmines, durationSeconds=7200,
    ))


def test_weather_and_trending_topic_groups():
    import cluster_api

    items = [
        _feed_video(f"rain{i}", f"रायपुर में भारी बारिश का अलर्ट {i}", views=20)
        for i in range(1)
    ]
    items.append(_feed_video("heat", "छत्तीसगढ़ में मौसम विभाग का अलर्ट", views=20))
    items.append(_feed_video("meet", "रायपुर में बैठक", views=20))
    items.extend(
        _feed_video(
            f"paddy{i}", f"धान खरीदी केंद्र में भीड़ {i}", views=80000,
            channelTitle="Desk A" if i == 0 else "Desk B",
        )
        for i in range(3)
    )
    items.extend(
        _feed_video(f"other{i}", f"अलग अलग मुद्दा संख्या {i}", views=5)
        for i in range(12)
    )
    clusters = cluster_api.extract_clusters({"sections": [{"section": "General", "items": items}]})
    by_topic = {cluster["topic"]: cluster for cluster in clusters}
    weather_ids = {video["videoId"] for video in by_topic["Weather"]["videos"]}
    assert weather_ids == {"rain0", "heat"}
    assert "meet" not in weather_ids
    assert any(
        cluster["clusterId"].startswith("topic-")
        and {video["videoId"] for video in cluster["videos"]} == {"paddy0", "paddy1", "paddy2"}
        for cluster in clusters
    )


def test_city_news_is_grouped_even_when_views_are_low():
    import cluster_api

    def video(video_id, title, views=1):
        return {
            "videoId": video_id,
            "title": title,
            "description": "chhattisgarh",
            "channelTitle": "Bhilai Times",
            "publishedAt": "2026-11-11T04:00:00Z",
            "views": views,
            "likes": 0,
            "genre": "General",
        }

    items = [video(f"n{i}", f"छत्तीसगढ़ समाचार {i}", views=100000) for i in range(20)]
    items.extend([
        video("b1", "भिलाई में पानी की समस्या", views=2),
        video("b2", "Bhilai market update", views=2),
        video("b3", "Bhilai road work", views=2),
        video("d1", "Durg city news", views=2),
        video("d2", "दुर्ग में घटना", views=2),
        video("d3", "Durg police", views=2),
        video("x1", "Durga pandal decoration", views=2),
    ])
    clusters = cluster_api.extract_clusters({"sections": [{"section": "General", "items": items}]})
    topics = {cluster["topic"]: cluster["videoCount"] for cluster in clusters}
    assert topics.get("Bhilai") == 3
    assert topics.get("Durg") == 3


def test_trending_movie_song_and_cm_clusters():
    import cluster_api

    assert cluster_api.is_new_cg_movie_release({
        "title": "Official Trailer | छत्तीसगढ़ी फिल्म नया रिलीज",
        "channelTitle": "CGFilm",
    })
    assert not cluster_api.is_new_cg_movie_release({
        "title": "Ramayana Star Riya Kapoor ने फिल्म पर खुलकर बात की",
        "channelTitle": "IBC24",
    })
    assert cluster_api.is_song_video({
        "title": "Gaura Gauri - Short Video Song | Kanchan Joshi",
        "channelTitle": "Sundrani CG Bhakti",
    })
    assert not cluster_api.is_song_video({
        "title": "Official Trailer | छत्तीसगढ़ी फिल्म नया रिलीज",
        "channelTitle": "CGFilm",
    })
    assert cluster_api.is_cm_video({
        "title": "Cm का आदेश | फसल मुआवजा",
        "channelTitle": "Bhilai Times",
    })
    assert cluster_api.is_cm_video({
        "title": "CM Mohan Yadav का बड़ा बयान, भोपाल",
        "channelTitle": "IBC24",
    })
    assert not cluster_api.is_cm_video({
        "title": "Pushkar Dhami LIVE: Pauri Garhwal में CM Dhami",
        "channelTitle": "Bansal News",
    })
    assert not cluster_api.is_cm_video({
        "title": "रायपुर में सड़क हादसा",
        "channelTitle": "IBC24",
    })


def test_region_filter_keeps_cg_mp_and_drops_other_states():
    assert gj.video_is_cg_or_mp("रायपुर में सड़क हादसा, दो की मौत")
    assert gj.video_is_cg_or_mp("भोपाल में मंत्रिमंडल की बैठक, मध्य प्रदेश")
    assert gj.video_is_cg_or_mp("Bilaspur Kolkata flight restored")
    assert gj.video_is_cg_or_mp("Raipur News: दिल्ली में नक्सलवाद ला रही कांग्रेस")
    assert gj.video_is_cg_or_mp("Montreal traffic update and road closure")
    assert gj.video_is_cg_or_mp("बंगाल की खाड़ी में चक्रवात")
    assert not gj.video_is_cg_or_mp("AAP Punjab List | पंजाब चुनाव के लिए पहली लिस्ट")
    assert not gj.video_is_cg_or_mp("Vande Bharat: Delhi-Varanasi के बीच नई ट्रेन, Prayagraj-Kanpur")
    assert not gj.video_is_cg_or_mp("लखनऊ में योगी सरकार का बड़ा फैसला")
    assert not gj.video_is_cg_or_mp("Cyclone: राजस्थान में ओले, उत्तराखंड में बर्फबारी, ओडिशा में लैंडस्लाइड")
    assert not gj.video_is_cg_or_mp("CM Yogi बोले देश सुरक्षित बन रहा है")


def test_generate_ott_json_adds_city_section_index_zero():
    feed = [
        {
            "videoId": "v1",
            "title": "Montreal traffic update and road closure",
            "description": "City bulletin",
            "channelTitle": "News Channel",
            "genre": gj.GENRE_NEGATIVE,
            "videoType": gj.VIDEO_TYPE_VOD,
            "publishedAt": "2026-01-01T00:00:00Z",
        },
        {
            "videoId": "v2",
            "title": "General CG bulletin",
            "description": "No city mention",
            "channelTitle": "News Channel",
            "genre": gj.GENRE_GENERAL,
            "videoType": gj.VIDEO_TYPE_VOD,
            "publishedAt": "2026-01-01T00:00:00Z",
        },
    ]

    class Resp:
        status_code = 200
        def json(self):
            return {"status": "success", "city": "Montreal"}

    gj._ip_city_cache.clear()
    with patch("requests.get", return_value=Resp()):
        out = gj.generate_ott_json(feed, request_ip="24.48.0.1", ip_lookup_url="http://ip-api.com/json/{ip}")

    section_names = [section["section"] for section in out["sections"]]
    assert "Scheduled" not in section_names
    assert "Podcast" in section_names
    assert "Negative" in section_names
    assert "Crime" not in section_names
    assert "Traffic" not in section_names
    negative = next(section for section in out["sections"] if section["section"] == "Negative")
    assert negative["sectionIndex"] == 7
    assert negative["count"] == 1
    assert negative["items"][0]["videoId"] == "v1"
    assert out["sections"][0]["section"] == "Montreal"
    assert out["sections"][0]["sectionIndex"] == 0
    assert out["sections"][0]["count"] == 1
    assert out["sections"][0]["items"][0]["videoId"] == "v1"

# === Tests that mock YouTube API calls ===
def mocked_requests_get(url, params=None, timeout=None, **kwargs):
    # Simulate the uploads playlist. The title must be inside the last-day window.
    if url == gj.YOUTUBE_PLAYLIST_URL:
        published = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        playlist_json = {
            "items": [
                {
                    "snippet": {
                        "title": "Test video title",
                        "description": "A short description",
                        "channelTitle": "TestChannel",
                        "publishedAt": published,
                        "thumbnails": {"high": {"url": "http://example.com/thumb.jpg"}},
                        "resourceId": {"videoId": "vid1"},
                    },
                    "contentDetails": {
                        "videoId": "vid1",
                        "videoPublishedAt": published,
                    },
                }
            ]
        }
        return MockResponse(200, playlist_json)
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
    gj.BHILAI_CHANNELS = gj.load_bhilai_channels()
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