# Cluster Trending API Documentation

## Overview

The Cluster Trending API is a RESTful web service that analyzes and serves trending news clusters based on YouTube video data. It provides endpoints to query clusters, filter by trending scores, and access detailed cluster information.

## Features

- 🔍 **Cluster Analysis**: Automatically analyzes video clusters and calculates trending scores
- 📊 **Multiple Endpoints**: Access data through various filtering and sorting options
- ⚡ **Caching**: Built-in caching for improved performance
- 🌐 **CORS Support**: Cross-origin requests enabled for web applications
- 📝 **Detailed Metrics**: Engagement rates, trending velocity, and more
- 🔒 **Error Handling**: Comprehensive error handling and validation

## Getting Started

### Prerequisites

- Python 3.7+
- pip package manager

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the API (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the server:
```bash
python cluster_api.py
```

The API will start on `http://localhost:5000` by default.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `5000` | Server port number |
| `DEBUG` | `False` | Enable debug mode |
| `OUTPUT_FILE` | `output.json` | Path to data file |
| `CACHE_TIMEOUT` | `300` | Cache timeout in seconds |

## API Endpoints

### Root Endpoint

**GET /**

Returns API information and available endpoints.

**Response:**
```json
{
  "name": "Cluster Trending API",
  "version": "1.0.0",
  "description": "REST API for analyzing and serving trending news clusters",
  "endpoints": { ... },
  "timestamp": "2026-01-04T10:00:00Z"
}
```

---

### Health Check

**GET /health**

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "dataLoaded": true,
  "timestamp": "2026-01-04T10:00:00Z"
}
```

---

### Get All Clusters

**GET /api/clusters**

Returns all available clusters with summary information.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "clusterId": "general",
      "topic": "General",
      "videoCount": 104,
      "trendScore": 85.3,
      "latestUpdateAt": "2026-01-04T09:41:22Z",
      "totalViews": 12500,
      "totalLikes": 350,
      "engagementRate": 2.8,
      "trendingVelocity": 120.2
    },
    ...
  ],
  "timestamp": "2026-01-04T10:00:00Z"
}
```

---

### Get Trending Clusters

**GET /api/clusters/trending**

Returns all clusters sorted by trend score in descending order.

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "clusterId": "politics",
      "topic": "Politics",
      "videoCount": 87,
      "trendScore": 92.4,
      "latestUpdateAt": "2026-01-04T09:41:22Z",
      "totalViews": 45000,
      "totalLikes": 1250,
      "engagementRate": 2.78,
      "trendingVelocity": 517.2
    },
    ...
  ],
  "timestamp": "2026-01-04T10:00:00Z"
}
```

---

### Get Top N Trending Clusters

**GET /api/clusters/trending/top/:n**

Returns the top N clusters with highest trend scores.

**Parameters:**
- `n` (path parameter): Number of top clusters to return (1-100)

**Example:** `GET /api/clusters/trending/top/5`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "clusterId": "politics",
      "topic": "Politics",
      "videoCount": 87,
      "trendScore": 92.4,
      ...
    }
  ],
  "timestamp": "2026-01-04T10:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: If n < 1 or n > 100

---

### Get Cluster By ID

**GET /api/clusters/:clusterId**

Returns detailed information about a specific cluster, including all videos.

**Parameters:**
- `clusterId` (path parameter): Cluster identifier (e.g., "politics", "crime", "general")

**Example:** `GET /api/clusters/politics`

**Response:**
```json
{
  "success": true,
  "data": {
    "clusterId": "politics",
    "topic": "Politics",
    "videoCount": 87,
    "trendScore": 92.4,
    "latestUpdateAt": "2026-01-04T09:41:22Z",
    "totalViews": 45000,
    "totalLikes": 1250,
    "engagementRate": 2.78,
    "trendingVelocity": 517.2,
    "videos": [
      {
        "videoId": "abc123",
        "title": "Video Title",
        "description": "Video description...",
        "channelTitle": "Channel Name",
        "publishedAt": "2026-01-04T09:00:00Z",
        "thumbnail": "https://...",
        "durationSeconds": 300,
        "durationFormatted": "05:00",
        "views": 1500,
        "likes": 45,
        "videoType": "VOD",
        "genre": "Politics"
      },
      ...
    ]
  },
  "timestamp": "2026-01-04T10:00:00Z"
}
```

**Error Responses:**
- `404 Not Found`: If cluster with specified ID doesn't exist

---

### Filter Clusters By Score

**GET /api/clusters/filter?minScore=X**

Returns clusters with trend score greater than or equal to the specified minimum score.

**Query Parameters:**
- `minScore` (required): Minimum trend score (0-100)

**Example:** `GET /api/clusters/filter?minScore=85`

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "clusterId": "politics",
      "topic": "Politics",
      "videoCount": 87,
      "trendScore": 92.4,
      ...
    },
    {
      "clusterId": "general",
      "topic": "General",
      "videoCount": 104,
      "trendScore": 85.3,
      ...
    }
  ],
  "timestamp": "2026-01-04T10:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: If minScore is missing, invalid, or out of range (0-100)

---

### Get Raw Data File

**GET /api/data/output.json**

Returns the raw output.json file.

**Response:** Raw JSON file content

**Error Responses:**
- `404 Not Found`: If output.json file doesn't exist
- `500 Internal Server Error`: If file cannot be read

---

### Get Trending Cluster JSON File

**GET /api/data/trending_cluster.json**

Fetches and returns the `trending_cluster.json` file from the GitHub repository's main branch. This endpoint retrieves the latest committed version of the trending clusters.

**GitHub URL:** `https://raw.githubusercontent.com/MridulEcolab/TestPipeline/main/trending_cluster.json`

**Response:**
```json
{
  "success": true,
  "data": {
    "generatedAt": "2026-01-04T10:29:03Z",
    "totalClusters": 5,
    "clusters": [
      {
        "clusterId": "politics",
        "topic": "Politics",
        "videoCount": 42,
        "trendScore": 69.0,
        "latestUpdateAt": "2026-01-04T09:35:07Z",
        "totalViews": 67337,
        "totalLikes": 3211,
        "engagementRate": 4.77,
        "trendingVelocity": 1603.3
      },
      ...
    ]
  },
  "timestamp": "2026-01-04T10:42:05Z"
}
```

**Features:**
- Fetches from GitHub's raw content URL
- Returns the committed version (stable snapshots)
- Clusters are sorted by `trendScore` in descending order (highest first)
- Contains summary information (no video details)
- Includes metadata: `generatedAt` timestamp and `totalClusters` count
- Uses standardized success response format
- 10-second timeout for requests

**Error Responses:**
- `404 Not Found`: If file doesn't exist on GitHub yet
- `403 Forbidden`: Rate limiting or access issues
- `502 Bad Gateway`: Other HTTP errors from GitHub
- `504 Gateway Timeout`: Request timeout

**Use Cases:**
- Access the latest committed trending data from GitHub
- Integrate with external systems using GitHub as data source
- Retrieve historical snapshots from version control
- Share stable trending data via GitHub CDN

**Example:**
```bash
curl http://localhost:5000/api/data/trending_cluster.json
```

---

## Data Model
- Returns the committed version (not dynamically generated)
- Uses standardized success response format
- 10-second timeout for requests

**Error Responses:**
- `404 Not Found`: If file doesn't exist on GitHub yet
- `500 Internal Server Error`: If fetch fails or invalid JSON
- `504 Gateway Timeout`: If GitHub request times out

**Use Cases:**
- Access the latest committed trending data
- Integrate with external systems using GitHub as data source
- Retrieve historical snapshots from version control
- Share stable trending data via GitHub CDN

**Example:**
```bash
curl http://localhost:5000/api/data/trending_cluster.json
```

---

## Data Model

### Cluster Object

| Field | Type | Description |
|-------|------|-------------|
| `clusterId` | string | Unique identifier for the cluster |
| `topic` | string | Auto-generated topic title from video content |
| `originalCategory` | string | Original category/section name |
| `videoCount` | integer | Number of videos in the cluster |
| `trendScore` | number | Trending score (0-100) |
| `latestUpdateAt` | string | ISO 8601 timestamp of latest video |
| `totalViews` | integer | Total views across all videos |
| `totalLikes` | integer | Total likes across all videos |
| `engagementRate` | number | Percentage of likes per view |
| `trendingVelocity` | number | Average views per video |
| `videos` | array | Array of video objects (only in detail view) |

### Auto-Generated Cluster Titles

The API automatically generates descriptive cluster titles by analyzing video content:

**Algorithm:**
1. Extracts words from all video titles in the cluster
2. Filters out common stop words (the, and, in, news, etc.)
3. Scores words based on:
   - **Frequency**: How often the word appears
   - **Proper nouns**: Words that are capitalized (names, places, events)
   - **Meaningfulness**: Length and alphabetic content
4. Selects top 2-3 highest-scored words

**Preferences:**
- **Names**: Person names, organization names (e.g., "Mohan Yadav", "BJP")
- **Places**: City and location names (e.g., "Indore", "Chhattisgarh")
- **Events**: Event-related nouns (e.g., "Accident", "Election")
- **Mixed languages**: Supports both English and Hindi content

**Benefits:**
- More descriptive than generic categories
- Reflects actual content themes
- Dynamically adapts to trending topics
- Preserves original category in `originalCategory` field

### Trend Score Calculation

The trend score is calculated using multiple weighted factors:

- **30%** - Video count (more videos = higher score)
- **25%** - Total views (logarithmic scale)
- **20%** - Engagement rate (likes/views ratio)
- **15%** - Recency (newer content scores higher)
- **10%** - Trending velocity (views per video)

Formula ensures scores range from 0 to 100, with 100 representing maximum trending potential.

### Derived Metrics

**Engagement Rate**: `(totalLikes / totalViews) * 100`
- Measures how engaged viewers are with the content
- Higher rate indicates quality content

**Trending Velocity**: `totalViews / videoCount`
- Average views per video in the cluster
- Indicates per-video popularity

---

## Response Format

All API responses follow a consistent format:

### Success Response
```json
{
  "success": true,
  "data": <response data>,
  "timestamp": "2026-01-04T10:00:00Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message describing what went wrong",
  "timestamp": "2026-01-04T10:00:00Z"
}
```

---

## HTTP Status Codes

| Code | Description |
|------|-------------|
| `200 OK` | Request successful |
| `400 Bad Request` | Invalid request parameters |
| `404 Not Found` | Resource not found |
| `500 Internal Server Error` | Server error |

---

## Usage Examples

### Python

```python
import requests

# Get all trending clusters
response = requests.get('http://localhost:5000/api/clusters/trending')
data = response.json()

for cluster in data['data']:
    print(f"{cluster['topic']}: {cluster['trendScore']}")
```

### JavaScript (fetch)

```javascript
// Get top 5 trending clusters
fetch('http://localhost:5000/api/clusters/trending/top/5')
  .then(response => response.json())
  .then(data => {
    data.data.forEach(cluster => {
      console.log(`${cluster.topic}: ${cluster.trendScore}`);
    });
  });
```

### cURL

```bash
# Get cluster by ID
curl http://localhost:5000/api/clusters/politics

# Filter clusters with score >= 80
curl 'http://localhost:5000/api/clusters/filter?minScore=80'
```

---

## Performance

### Caching

The API implements intelligent caching:
- Data is cached for 5 minutes (configurable via `CACHE_TIMEOUT`)
- Automatic cache invalidation when new data is detected
- Cluster calculations are cached after first computation

### Best Practices

1. **Use filtering endpoints** to reduce response size
2. **Cache responses** on client-side for frequently accessed data
3. **Use summary endpoints** (`/api/clusters/trending`) when full video details aren't needed
4. **Implement pagination** on client-side for large result sets

---

## Error Handling

The API provides detailed error messages for debugging:

```json
{
  "success": false,
  "error": "Parameter 'minScore' must be between 0 and 100",
  "timestamp": "2026-01-04T10:00:00Z"
}
```

Common error scenarios:
- Missing or invalid parameters
- Data file not found
- Invalid JSON format
- Resource not found

---

## Development

### Running in Debug Mode

```bash
export DEBUG=True
python cluster_api.py
```

Debug mode enables:
- Detailed error messages
- Auto-reload on code changes
- Enhanced logging

### Testing

Run manual tests using cURL or a REST client like Postman.

Example test commands:
```bash
# Test health endpoint
curl http://localhost:5000/health

# Test all clusters
curl http://localhost:5000/api/clusters

# Test filtering
curl 'http://localhost:5000/api/clusters/filter?minScore=50'
```

---

## Deployment

### Production Considerations

1. **Use a production WSGI server** (e.g., Gunicorn, uWSGI)
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 cluster_api:app
   ```

2. **Set environment variables**
   ```bash
   export DEBUG=False
   export PORT=5000
   ```

3. **Use reverse proxy** (e.g., Nginx) for SSL and load balancing

4. **Monitor logs** for errors and performance issues

5. **Configure firewall** to allow traffic on your chosen port

---

## License

This project is part of the TestPipeline repository.

---

## Support

For issues or questions, please create an issue in the GitHub repository.
