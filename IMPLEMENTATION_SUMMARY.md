# Implementation Summary: Automatic Update of trending_cluster.json

## Problem Statement

The requirement: **"trending_cluster.json should also update once output.json is updated"**

### Root Cause

- The pipeline (GitHub Actions and local) was updating `output.json` but not `trending_cluster.json`.
- Logic to build trending clusters lived in `cluster_api.py` but was never invoked in the pipeline.

## Solution Overview

`trending_cluster.json` is now generated automatically whenever the pipeline runs, so it stays in sync with `output.json` in both GitHub Actions and local runs.

## Implementation Details

### 1. Script: `generate_trending_clusters.py`

**Purpose**: Build `trending_cluster.json` from the current `output.json`.

**Flow**:

1. Load `output.json` via `cluster_api.load_output_json()`.
2. Extract clusters via `cluster_api.extract_clusters(data)` (PSE, Movie, Festival, and content-based clusters with ≥4 videos).
3. Build trending payload with `cluster_api.generate_trending_cluster_json(clusters)` (sorted by trend score, top 5 videos per cluster).
4. Write result with `cluster_api.save_trending_cluster_json(trending_data)` to `trending_cluster.json`.

**Trend score** (in `cluster_api.calculate_trend_score()`):

- Video count (30%) — normalized 0–100.
- Total views (25%) — log scale, normalized.
- Engagement rate (20%) — likes/views.
- Recency (15%) — exponential decay by age.
- Velocity (10%) — views per video, log scale.

**Output file**: `trending_cluster.json` with `generatedAt`, `clusterCount`, and `clusters` (each: `clusterId`, `topic`, `videoCount`, `trendScore`, `topVideos`, `latestUpdateAt`).

**Usage**:

```bash
python generate_trending_clusters.py
```

### 2. GitHub Actions Workflow

**File**: `.github/workflows/cg_pipeline.yml`

**Schedule**: Every 2 hours (cron `10 4-22/2 * * *` UTC).

**Steps**:

1. Checkout, set up Python 3.11, install dependencies.
2. Run `python generate_json.py` (env: `YOUTUBE_API_KEY*`, `BHILAI_CHANNELS`, `OPENAI_API_KEY`).
3. Run `python generate_trending_clusters.py`.
4. **Commit and push**: Only if `output.json` changed (`git diff --quiet output.json`). If changed, `git add output.json trending_cluster.json`, commit, push.
5. Upload `output.json` as artifact; optional display and URL hints.

Both files are committed together when `output.json` changes so `trending_cluster.json` always matches the latest data.

### 3. Local Pipeline

**File**: `run_pipeline.sh`

**Flow**:

1. Check Python, install dependencies.
2. `python3 generate_json.py`.
3. `python3 generate_trending_clusters.py`.
4. Print paths to `output.json` and `trending_cluster.json` and validation hint.

**Usage**:

```bash
./run_pipeline.sh
```

### 4. Cluster API and API Endpoint

**File**: `cluster_api.py`

- **`extract_clusters(data)`**: Builds PSE, Movie, Festival, and content-based clusters (similarity grouping, min size 4). Each cluster gets `trendScore` via `calculate_trend_score()`.
- **`generate_trending_cluster_json(clusters)`**: Sorts by `trendScore`, keeps top 5 videos per cluster, returns structure for `trending_cluster.json`.
- **`save_trending_cluster_json(trending_data, filename)`**: Writes JSON to `trending_cluster.json` (or given filename).

**Endpoint**: `GET /api/data/trending_cluster.json` — serves the file (e.g. from GitHub raw URL when configured). See `API_DOCUMENTATION.md` for base URL and usage.

### 5. Integration Test

**File**: `test_trending_clusters_generation.py`

- Runs `generate_trending_clusters.py` and checks:
  - Script exits successfully.
  - `trending_cluster.json` exists and is valid JSON.
  - Required fields and structure (e.g. `generatedAt`, `clusterCount`, `clusters`, `topVideos`).
  - Timestamp freshness (optional warning if file is old).

## Validation

- **Code**: Script and workflow follow project style.
- **JSON**: `output.json` and `trending_cluster.json` are valid.
- **Integration**: `test_trending_clusters_generation.py` validates end-to-end generation.
- **Workflow**: `.github/workflows/cg_pipeline.yml` is valid and runs on schedule and `workflow_dispatch`.

## Testing

**Local**:

```bash
# Full pipeline (output.json + trending_cluster.json)
./run_pipeline.sh

# Only trending clusters (requires existing output.json)
python generate_trending_clusters.py

# Validate JSONs
python validate_json.py

# Integration test for trending generation
python test_trending_clusters_generation.py
```

**Manual checks**: Confirm both files have current timestamps, cluster count and top videos look correct, and trend scores are populated.

## Files Involved

| File | Role |
|------|------|
| `generate_trending_clusters.py` | Script that generates `trending_cluster.json` from `output.json`. |
| `cluster_api.py` | `load_output_json`, `extract_clusters`, `calculate_trend_score`, `generate_trending_cluster_json`, `save_trending_cluster_json`; API route for `trending_cluster.json`. |
| `.github/workflows/cg_pipeline.yml` | Runs `generate_json.py` then `generate_trending_clusters.py`; commits both JSONs when `output.json` changes. |
| `run_pipeline.sh` | Local pipeline: `generate_json.py` → `generate_trending_clusters.py`. |
| `test_trending_clusters_generation.py` | Integration test for trending cluster generation. |
| `IMPLEMENTATION_SUMMARY.md` | This summary. |

## Benefits

- **Sync**: `trending_cluster.json` updates whenever `output.json` is updated (CI and local).
- **Minimal surface**: Reuses existing cluster and scoring logic in `cluster_api.py`.
- **Single commit**: One commit contains both JSONs when the workflow pushes.
- **Tested**: Integration test and manual checks ensure the pipeline and output shape stay correct.

## End-to-End Flow

**GitHub Actions**:

```
Schedule (every 2h) or workflow_dispatch
  → generate_json.py → output.json
  → generate_trending_clusters.py → trending_cluster.json
  → If output.json changed: commit & push both files
```

**Local**:

```
./run_pipeline.sh
  → generate_json.py → output.json
  → generate_trending_clusters.py → trending_cluster.json
  → Print paths and validation command
```

## Optional Future Enhancements

- Trending history or time-series data.
- Configurable trend-score weights.
- Trend-change or anomaly detection.
- More cluster types or filters.

## Conclusion

`trending_cluster.json` is now updated automatically whenever `output.json` is updated, in both GitHub Actions (`.github/workflows/cg_pipeline.yml`) and local runs (`run_pipeline.sh`). The behavior is documented, tested, and aligned with the current cluster and scoring logic in `cluster_api.py`.
