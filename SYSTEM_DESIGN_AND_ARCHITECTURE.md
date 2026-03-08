# System Design & Architecture: YTCG Pipeline

## 1. Overview

### 1.1 Purpose

The **YTCG Pipeline** (YouTube Chhattisgarh / Bhilai News Pipeline) is a data pipeline that:

1. **Ingests** latest videos from configured Bhilai/news YouTube channels via YouTube Data API v3.
2. **Enriches** each video with genre classification (Live, Scheduled, Crime, Traffic, Jobs, Events, Civic, Politics, Podcast, General) using a hybrid approach: OpenAI for recent videos, rule-based keywords otherwise.
3. **Publishes** two JSON artifacts:
   - **output.json** — OTT-style feed with sections (Live, Scheduled, General, Podcast, Jobs, Politics, Events, Civic, Traffic, Crime) and full video metadata.
   - **trending_cluster.json** — Clusters of similar/topical content (PSE, Movie, Festival, content-based) sorted by trend score, with top videos per cluster.
4. **Serves** data via an optional **Cluster API** (Flask) for clusters, filtering, and raw file access (including proxy to GitHub raw for `trending_cluster.json`).

### 1.2 Scope

- **In scope**: Ingestion from YouTube, genre classification, sectioning, clustering, JSON generation, CI/CD (GitHub Actions), local run script, and Cluster API.
- **Out of scope**: User authentication, real-time streaming, storage beyond generated JSON files, and front-end OTT app logic.

### 1.3 High-Level Flow

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  YouTube API    │────▶│  generate_json   │────▶│  output.json         │
│  (channels)     │     │  + genre (OpenAI │     │  (sections + items)  │
└─────────────────┘     │   / keyword)     │     └──────────┬──────────┘
                         └──────────────────┘                │
                                    │                          │
                         ┌──────────▼──────────┐               │
                         │  OpenAI (optional)  │               │
                         │  recent videos only │               │
                         └─────────────────────┘               │
                                                                │
┌─────────────────┐     ┌──────────────────┐                  │
│  cluster_api    │◀────│  generate_        │◀─────────────────┘
│  (Flask server)  │     │  trending_        │   output.json
│  /api/...       │     │  clusters.py      │
└────────┬────────┘     └────────┬─────────┘
         │                        │
         │                        ▼
         │               trending_cluster.json
         │
         └──────────────▶ Clients (raw JSON / API)
```

---

## 2. Architecture

### 2.1 Layered View

| Layer | Components | Responsibility |
|-------|------------|----------------|
| **Orchestration** | GitHub Actions (`cg_pipeline.yml`), `run_pipeline.sh` | Schedule, env, run order, commit/push |
| **Ingestion & Enrichment** | `generate_json.py` | Fetch videos, classify genre, build OTT sections, write `output.json` |
| **Clustering & Trending** | `generate_trending_clusters.py`, `cluster_api.py` (lib) | Read `output.json`, extract clusters, score, write `trending_cluster.json` |
| **Serving** | `cluster_api.py` (Flask app) | REST API for clusters, filters, and raw/proxied JSON |
| **Data** | `output.json`, `trending_cluster.json` | Canonical outputs; optionally served via API or GitHub raw |

### 2.2 Component Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                  ORCHESTRATION                          │
                    │  GitHub Actions (cron / dispatch)  │  run_pipeline.sh    │
                    └────────────────────────┬────────────────────────────────┘
                                              │
                    ┌─────────────────────────▼────────────────────────────────┐
                    │              generate_json.py                             │
                    │  • load_api_keys, load_bhilai_channels                    │
                    │  • aggregate_bhilai_videos → fetch_latest_videos_for_channel│
                    │  • add_genres_to_feed (OpenAI + keyword fallback)          │
                    │  • generate_ott_json → output.json                        │
                    └─────────────────────────┬────────────────────────────────┘
                                              │ output.json
                    ┌─────────────────────────▼────────────────────────────────┐
                    │         generate_trending_clusters.py                     │
                    │  • load_output_json, extract_clusters                      │
                    │  • generate_trending_cluster_json, save_trending_cluster   │
                    └─────────────────────────┬────────────────────────────────┘
                                              │ trending_cluster.json
                    ┌─────────────────────────▼────────────────────────────────┐
                    │              cluster_api.py (Flask)                       │
                    │  • load_output_json (cached), extract_clusters             │
                    │  • GET /api/clusters, /trending, /filter, /<id>           │
                    │  • GET /api/data/output.json (local file)                 │
                    │  • GET /api/data/trending_cluster.json (GitHub raw proxy)  │
                    └──────────────────────────────────────────────────────────┘
```

### 2.3 Data Flow

1. **Config**: API keys and channel list from environment (`YOUTUBE_API_KEY*`, `BHILAI_CHANNELS`, `OPENAI_API_KEY`).
2. **Fetch**: For each channel, search latest videos (YouTube Search + Videos), merge, sort by `publishedAt`.
3. **Classify**: For each video, set `genre` (Live/Scheduled by `videoType`; else OpenAI for last 1h, else keyword-based). Podcast channels forced to genre Podcast.
4. **Section**: Build OTT sections from genres; write `output.json` (generatedAt, sections with sectionIndex, items).
5. **Cluster**: Read `output.json`; extract PSE / Movie / Festival + content-based clusters (min 4 videos); compute trend score; sort; top 5 videos per cluster; write `trending_cluster.json`.
6. **Optional**: Cluster API serves clusters and files; `trending_cluster.json` can be read from disk or proxied from GitHub raw.

---

## 3. Component Design

### 3.1 generate_json.py

**Role**: Single entry point to produce `output.json`.

**Main steps**:

1. **Config**
   - Load YouTube API keys (primary + `YOUTUBE_API_KEY_2`, etc.) and `BHILAI_CHANNELS` from env.
   - If no keys or no channels: write error payload to `output.json` (or skip update) and exit.

2. **Aggregation**
   - `aggregate_bhilai_videos(channels)`: for each channel, `fetch_latest_videos_for_channel(channel_id)`.
   - Per channel: YouTube Search (max 20, order by date) → extract `videoId` → `fetch_video_details(video_ids)` (batch Videos API for snippet, contentDetails, statistics) → merge into flat list; on quota/invalid-key, switch to next API key.
   - Merge all channel results, sort by `publishedAt` descending.

3. **Genre classification** (`add_genres_to_feed`)
   - Live/Scheduled: from `videoType` (LIVE, SCHEDULED/UPCOMING).
   - Others: if published in last 1 hour and OpenAI enabled → `classify_genre_with_openai`; else or on failure → `classify_genre_keyword_based(title, description, channel_title)`.
   - Circuit breaker: first OpenAI failure stops OpenAI for rest of batch; time cap (e.g. 50s) stops OpenAI after limit; Podcast channel override (channel allow-list) forces or rejects Podcast.

4. **OTT structure** (`generate_ott_json`)
   - Filter by genre into sections: Live, Scheduled, General, Podcast, Jobs, Politics, Events, Civic, Traffic, Crime (Podcast section only from allowed channels; other sections exclude those channels).
   - Attach `sectionIndex` per section; output `{ generatedAt, sections }`.

5. **Persistence**
   - Write `output.json` only when config is valid and feed non-empty (otherwise preserve existing file).

**Key design choices**:
- Multiple YouTube API keys for quota and key-rotation.
- Hybrid genre: OpenAI for recency, keyword for stability and cost control.
- Circuit breaker and time limit to avoid runaway OpenAI usage.

### 3.2 generate_trending_clusters.py

**Role**: Produce `trending_cluster.json` from current `output.json`.

**Steps**:
1. Load `output.json` via `cluster_api.load_output_json()`.
2. `extract_clusters(data)` (see cluster_api): PSE / Movie / Festival by rules; rest grouped by content similarity (min 4); each cluster gets `trendScore` via `calculate_trend_score`.
3. `generate_trending_cluster_json(clusters)`: sort by `trendScore` desc; for each cluster take top 5 videos by views; output `{ generatedAt, clusterCount, clusters }` with `clusterId`, `topic`, `videoCount`, `trendScore`, `topVideos`, `latestUpdateAt`.
4. `save_trending_cluster_json(trending_data)` → `trending_cluster.json`.

**Dependency**: Requires `output.json` and `cluster_api` (extract_clusters, scoring, save).

### 3.3 cluster_api.py

**Dual role**: (1) Library for clustering and trending; (2) Flask server for REST API.

**Library**:
- **load_output_json**: Read and parse `output.json`; optional in-memory cache (e.g. 5 min).
- **extract_clusters**: Flatten sections → classify each video as PSE / Movie / Festival / other; build PSE/Movie/Festival clusters; group “other” with `group_similar_videos` (min cluster size 4); assign `trendScore` (video count, views, engagement, recency, velocity).
- **calculate_trend_score**: Weighted formula (e.g. 30% count, 25% views, 20% engagement, 15% recency, 10% velocity), normalized to a 0–100 scale.
- **generate_trending_cluster_json**: Sort clusters, top 5 videos per cluster, structure for `trending_cluster.json`.
- **save_trending_cluster_json**: Write JSON to file.

**Flask API**:
- `GET /`: API info and endpoint list.
- `GET /health`: Health; checks if data loaded.
- `GET /api/clusters`: All clusters (summary, no full video list).
- `GET /api/clusters/trending`: Clusters sorted by trend score.
- `GET /api/clusters/trending/top/<n>`: Top N clusters.
- `GET /api/clusters/<clusterId>`: Single cluster with videos.
- `GET /api/clusters/filter?minScore=X`: Clusters with trend score ≥ X.
- `GET /api/data/output.json`: Serve local `output.json`.
- `GET /api/data/trending_cluster.json`: Fetch from GitHub raw URL and return (proxy).
- 404/500 handlers; CORS enabled; `@require_data` for endpoints that need loaded data.

**Config**: `OUTPUT_FILE`, `TRENDING_CLUSTER_FILE`, `GITHUB_RAW_BASE_URL`, `CACHE_TIMEOUT`, `PORT`/`HOST`/`DEBUG`.

### 3.4 Orchestration

**GitHub Actions** (`.github/workflows/cg_pipeline.yml`):
- **Trigger**: Schedule (`cron: '10 4-22/2 * * *'`) and `workflow_dispatch`.
- **Job**: Checkout → Python 3.11 → install deps → run `generate_json.py` then `generate_trending_clusters.py` with env (YouTube keys, BHILAI_CHANNELS, OPENAI_API_KEY).
- **Commit**: Only if `output.json` changed; then `git add output.json trending_cluster.json`, commit, push.
- **Artifacts**: Upload `output.json`; print raw/Pages URLs.

**Local** (`run_pipeline.sh`):
- Check Python3 → install deps → `generate_json.py` → `generate_trending_clusters.py` → print paths and validation command.

---

## 4. Data Model

### 4.1 output.json

- **generatedAt**: ISO8601 UTC.
- **sections**: Array of:
  - **section**: Name (Live, Scheduled, General, Podcast, Jobs, Politics, Events, Civic, Traffic, Crime).
  - **sectionIndex**: Numeric index for ordering (e.g. General=1, Podcast=2, …).
  - **count**: Number of videos in section.
  - **items**: Array of video objects.

**Video object** (per item):
- **videoId**, **title**, **description**, **channelTitle**, **publishedAt**, **thumbnail**
- **durationSeconds**, **durationFormatted**
- **views**, **likes**
- **videoType** (LIVE, VOD, SCHEDULED, UNKNOWN)
- **genre** (Live, Scheduled, Crime, Traffic, Politics, Jobs, Events, Civic, Podcast, General)

### 4.2 trending_cluster.json

- **generatedAt**: ISO8601 UTC.
- **clusterCount**: Number of clusters.
- **clusters**: Array of:
  - **clusterId**, **topic**, **videoCount**, **trendScore**, **latestUpdateAt**
  - **topVideos**: Up to 5 videos (same shape as in output, with index).

Cluster types:
- **public-sector-exam**, **movie**, **festival**: Rule-based from title/description.
- **Content-based**: Similarity grouping on “other” videos; min size 4; auto-generated topic and slug clusterId.

### 4.3 Genre and Section Mapping

- **Genres**: Live, Scheduled, Crime, Traffic, Politics, Jobs, Events, Civic, Podcast, General.
- **Sections** (OTT tabs): Same names; sectionIndex defines order; Podcast is channel-allow-list only; other sections exclude those channels.

---

## 5. External Integrations

### 5.1 YouTube Data API v3

- **Endpoints**: Search (list), Videos (list) for details and statistics.
- **Auth**: API key(s) in env; multiple keys for fallback on quota or invalid key.
- **Usage**: Per-channel search (maxResults 20, order by date) → batch video details by ID.
- **Error handling**: 403 quota → switch key; 400 invalid key → switch key; repeat until keys exhausted or success.

### 5.2 OpenAI API

- **Usage**: Optional genre classification for videos published in last 1 hour.
- **Model**: e.g. `gpt-4o-mini`; low temperature; max tokens for single genre label.
- **Resilience**: Circuit breaker on first failure; time-bound (e.g. 50s); rate/token checks with safety margin; fallback to keyword classification; no script failure on OpenAI errors.
- **Config**: `OPENAI_API_KEY`, rate/token limits, time limit in `generate_json.py`.

### 5.3 GitHub

- **Actions**: Run pipeline; commit and push `output.json` and `trending_cluster.json` when output changes.
- **Raw**: Cluster API can proxy `trending_cluster.json` from GitHub raw URL (e.g. `GITHUB_RAW_BASE_URL`).

---

## 6. Deployment & Execution

### 6.1 CI/CD (GitHub Actions)

- **When**: Scheduled every 2 hours (UTC) + manual dispatch.
- **Runner**: ubuntu-latest, Python 3.11.
- **Secrets/Vars**: YouTube keys, OPENAI_API_KEY (secrets); BHILAI_CHANNELS (vars).
- **Output**: Updated repo files; artifact for `output.json`; logs with URLs.

### 6.2 Local

- **Command**: `./run_pipeline.sh` or run `generate_json.py` then `generate_trending_clusters.py` manually.
- **Requires**: `output.json` for trending step; env set for keys and channels.

### 6.3 Cluster API Server

- **Run**: `python cluster_api.py` (or app server with Flask).
- **Config**: `PORT`, `HOST`, `DEBUG`; expects `output.json` (and optionally `trending_cluster.json`) in working directory or configured path; `trending_cluster` endpoint may use GitHub raw.
- **Use case**: Serve clusters and JSON to front-ends or internal services.

---

## 7. Resilience & Configuration

### 7.1 YouTube

- **Multiple keys**: `YOUTUBE_API_KEY`, `YOUTUBE_API_KEY_2`, …; automatic switch on quota or invalid key.
- **No overwrite on failure**: If all keys fail or no channels, script does not overwrite `output.json` with empty data.

### 7.2 OpenAI

- **Circuit breaker**: One failure (rate limit, error, timeout) disables OpenAI for the rest of the batch.
- **Time cap**: Stops calling OpenAI after N seconds so total pipeline time is bounded.
- **Fallback**: Keyword-based classification always available; no dependency on OpenAI for script success.

### 7.3 Cluster API

- **Cache**: In-memory cache for `output.json` and derived clusters; TTL (e.g. 5 min) to balance freshness and load.
- **Errors**: 404/500 handlers; `require_data` returns 500 if data cannot be loaded.

### 7.4 Environment Summary

| Variable | Used by | Purpose |
|----------|---------|---------|
| YOUTUBE_API_KEY, YOUTUBE_API_KEY_2, … | generate_json | YouTube API auth and fallback |
| BHILAI_CHANNELS | generate_json | Comma-separated channel IDs |
| OPENAI_API_KEY | generate_json | Optional OpenAI genre classification |
| PORT, HOST, DEBUG | cluster_api | Flask server config |
| GITHUB_RAW_BASE_URL | cluster_api | Base URL for trending_cluster proxy |

---

## 8. Document References

- **API**: `API_DOCUMENTATION.md` — Cluster API endpoints and examples.
- **Genre**: `GENRE_CLASSIFICATION_ANALYSIS.md` — Genre logic, accuracy, recommendations.
- **OpenAI**: `OPENAI_CLASSIFICATION.md` — OpenAI setup, limits, fallback.
- **Trending**: `IMPLEMENTATION_SUMMARY.md` — Automatic update of `trending_cluster.json`.

---

## 9. Summary

The system is a **batch pipeline** plus **optional REST API**:

- **Pipeline**: YouTube → genre tagging (OpenAI + keyword) → OTT sections → `output.json`; then clusters + trend scores → `trending_cluster.json`. Orchestrated by GitHub Actions or `run_pipeline.sh`.
- **Serving**: Cluster API provides clusters, filters, and access to both JSONs; `trending_cluster.json` can be read from disk or GitHub raw.
- **Resilience**: Multiple YouTube keys, OpenAI circuit breaker and time limit, and safe fallbacks keep the pipeline running and the data files consistent.
