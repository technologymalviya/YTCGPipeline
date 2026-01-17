#!/usr/bin/env python3
"""
Cluster API Server
A RESTful API for analyzing and serving trending news clusters based on output.json
"""

import json
import os
import logging
import math
import re
import requests
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set, Tuple
from functools import wraps
from urllib.parse import urljoin
from collections import defaultdict

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# Configuration
OUTPUT_FILE = "output.json"
TRENDING_CLUSTER_FILE = "trending_cluster.json"
GITHUB_RAW_BASE_URL = "https://raw.githubusercontent.com/MridulEcolab/TestPipeline/main/"
GITHUB_REQUEST_TIMEOUT = 10  # seconds
DEFAULT_PORT = 5000
DEFAULT_HOST = "0.0.0.0"
CACHE_TIMEOUT = 300  # 5 minutes cache timeout

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Simple cache for frequently accessed data
cache = {
    'data': None,
    'timestamp': None,
    'clusters': None
}


def load_output_json() -> Dict[str, Any]:
    """Load and parse the output.json file with caching."""
    try:
        # Check cache
        if cache['data'] and cache['timestamp']:
            elapsed = (datetime.now(timezone.utc) - cache['timestamp']).total_seconds()
            if elapsed < CACHE_TIMEOUT:
                logger.debug("Returning cached data")
                return cache['data']
        
        # Load fresh data
        if not os.path.exists(OUTPUT_FILE):
            logger.error(f"{OUTPUT_FILE} not found")
            return None
        
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Update cache
        cache['data'] = data
        cache['timestamp'] = datetime.now(timezone.utc)
        cache['clusters'] = None  # Invalidate cluster cache
        
        logger.info(f"Loaded {OUTPUT_FILE} successfully")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {OUTPUT_FILE}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading {OUTPUT_FILE}: {e}")
        return None


def calculate_trend_score(section: Dict[str, Any], videos: List[Dict[str, Any]]) -> float:
    """
    Calculate trend score for a cluster based on multiple factors.
    
    Factors considered:
    - Video count (higher is better)
    - Total views (engagement)
    - Total likes (engagement quality)
    - Recency (newer content scores higher)
    - View-to-like ratio (engagement rate)
    """
    if not videos:
        return 0.0
    
    video_count = len(videos)
    total_views = sum(v.get('views', 0) for v in videos)
    total_likes = sum(v.get('likes', 0) for v in videos)
    
    # Calculate recency score (newer videos get higher scores)
    now = datetime.now(timezone.utc)
    recency_scores = []
    for video in videos:
        published_at = video.get('publishedAt', '')
        try:
            pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            hours_old = (now - pub_date).total_seconds() / 3600
            # Exponential decay: newer = higher score
            recency_score = max(0, 100 * (0.95 ** hours_old))
            recency_scores.append(recency_score)
        except (ValueError, AttributeError):
            recency_scores.append(0)
    
    avg_recency = sum(recency_scores) / len(recency_scores) if recency_scores else 0
    
    # Calculate engagement rate
    engagement_rate = (total_likes / total_views * 100) if total_views > 0 else 0
    
    # Calculate trending velocity (views per video)
    velocity = total_views / video_count if video_count > 0 else 0
    
    # Weighted trend score formula
    # - 30% video count (normalized to 0-100 scale)
    # - 25% total views (log scale, normalized)
    # - 20% engagement rate
    # - 15% recency
    # - 10% velocity (log scale)
    
    video_count_score = min(100, video_count * 3.7)  # 27 videos = ~100 points
    views_score = min(100, math.log10(max(1, total_views)) * 10)
    engagement_score = min(100, engagement_rate * 10)
    velocity_score = min(100, math.log10(max(1, velocity)) * 20)
    
    trend_score = (
        video_count_score * 0.30 +
        views_score * 0.25 +
        engagement_score * 0.20 +
        avg_recency * 0.15 +
        velocity_score * 0.10
    )
    
    return round(trend_score, 1)


def generate_cluster_title(videos: List[Dict[str, Any]], default_title: str = "Unknown") -> str:
    """
    Auto-generate short, news-style headline for cluster based on video content.
    
    Creates human-readable, short titles by:
    - Extracting most frequent meaningful words from TOP-VIEWED videos only
    - Preferring nouns (names, places, events)
    - Strongly weighting by video views to prioritize trending content
    - Formatting as news-style headline
    
    Args:
        videos: List of video dictionaries with 'title' and 'views' fields
        default_title: Fallback title if generation fails
    
    Returns:
        Short, news-style headline string (2-4 words max)
    """
    if not videos:
        return default_title
    
    # Sort by views and focus on top videos (where the real trending content is)
    sorted_videos = sorted(videos, key=lambda v: v.get('views', 0), reverse=True)
    # Only use top 15 videos or videos with at least 10% of max views
    max_views = sorted_videos[0].get('views', 1) if sorted_videos else 1
    threshold_views = max_views * 0.05  # 5% of top video views
    
    # Filter to high-performing videos only
    top_videos = [v for v in sorted_videos[:15] if v.get('views', 0) >= threshold_views]
    if not top_videos:
        top_videos = sorted_videos[:5]  # Fallback to at least top 5
    
    # Common stop words to filter out (English and Hindi transliterations)
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'video', 'news', 'live', 'latest', 'breaking', 'today', 'update', 'hindi',
        'exclusive', 'full', 'new', 'big', 'बड़ी', 'ताजा', 'खबर', 'न्यूज़',
        'में', 'को', 'की', 'का', 'के', 'से', 'ने', 'है', 'हैं', 'था', 'थी',
        'पर', 'और', 'या', 'बड़ा', 'मध्य', 'प्रदेश', 'madhya', 'pradesh',
        'ibc24', 'news24', 'mp24', 'mp', 'cg', 'india'
    }
    
    # Collect all words from titles weighted by video views
    word_info = {}  # {lowercase: {'weighted_count': float, 'capitalized_count': int, 'sample': str}}
    
    # Calculate max views for normalization (from top videos only)
    max_views = max((v.get('views', 0) for v in top_videos), default=1)
    if max_views == 0:
        max_views = 1
    
    for video in top_videos:  # Only process top-performing videos
        title = video.get('title', '')
        if not title:
            continue
        
        # Weight by views: videos with more views contribute MORE to word frequency
        # Use SQUARED log scale to make high-view videos MUCH more influential
        # This ensures words from viral content dominate the headline
        views = video.get('views', 0)
        log_weight = math.log(views + 1) / math.log(max_views + 1) if max_views > 0 else 0.1
        # Square the weight to make differences more pronounced
        # E.g., 26K views (weight ~1.0) vs 3K views (weight ~0.6) becomes 1.0 vs 0.36
        view_weight = log_weight ** 2
        # Ensure minimum weight so all videos contribute something
        view_weight = max(view_weight, 0.05)
        
        # Split on common delimiters
        words = title.replace('|', ' ').replace(':', ' ').replace('-', ' ').replace('...', ' ').split()
        
        for word in words:
            # Remove punctuation but keep the word
            cleaned = word.strip('.,!?;()[]{}"\'-')
            if not cleaned:
                continue
            
            cleaned_lower = cleaned.lower()
            
            # Skip if empty, too short, or is a stop word
            if len(cleaned_lower) < 3 or cleaned_lower in stop_words:
                continue
            
            # Skip if it's all numbers
            if cleaned.isdigit():
                continue
            
            # Track word info
            if cleaned_lower not in word_info:
                word_info[cleaned_lower] = {
                    'weighted_count': 0.0,
                    'capitalized_count': 0,
                    'sample': cleaned
                }
            
            word_info[cleaned_lower]['weighted_count'] += view_weight
            
            # Check if word starts with capital (likely proper noun)
            if cleaned[0].isupper():
                word_info[cleaned_lower]['capitalized_count'] += 1
                # Prefer capitalized version as sample
                word_info[cleaned_lower]['sample'] = cleaned
    
    if not word_info:
        return default_title
    
    # Score words: prefer frequently occurring + capitalized words (proper nouns)
    # with emphasis on high-view videos
    scored_words = []
    for word_lower, info in word_info.items():
        # Base score on weighted frequency (accounts for video views)
        # This is the PRIMARY factor - words from high-view videos get higher base scores
        score = info['weighted_count']
        
        # Apply MULTIPLIERS for proper nouns and English words
        # This way, the view weighting remains dominant
        multiplier = 1.0
        
        # Boost multiplier if it's often capitalized (likely a proper noun - name, place, event)
        if info['capitalized_count'] >= 1:  # Appears capitalized at least once
            multiplier *= 2.0  # 2x multiplier for proper nouns (was +15 additive)
        
        # Boost multiplier for English words (likely names/places in transliteration)
        if word_lower.isascii() and word_lower.isalpha():
            multiplier *= 1.5  # 1.5x multiplier for English words (was +8 additive)
        
        final_score = score * multiplier
        scored_words.append((info['sample'], final_score, info['weighted_count']))
    
    # Sort by score (highest first)
    scored_words.sort(key=lambda x: x[1], reverse=True)
    
    # Build short, news-style headline from top scored words (max 2-3 words)
    top_words = []
    for word, score, weighted_count in scored_words[:8]:  # Look at top 8 candidates
        # Must have meaningful weight to be included
        if weighted_count >= 0.5:  # Lowered threshold for weighted count
            top_words.append(word)
            
            # Keep it short: max 2-3 words for news-style headline
            if len(top_words) >= 2:
                break
    
    if top_words:
        # Create short, news-style headline (2-3 words max)
        generated_title = ' '.join(top_words[:3])
        logger.debug(f"Generated headline: '{generated_title}' from {len(videos)} videos")
        return generated_title
    
    return default_title


def normalize_text_for_clustering(text: str) -> str:
    """Normalize text for similarity comparison."""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text


def extract_significant_words(text: str) -> Set[str]:
    """
    Extract significant words from text for similarity comparison.
    Filters out stop words and short words.
    """
    if not text:
        return set()
    
    normalized = normalize_text_for_clustering(text)
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'video', 'news', 'live', 'latest', 'breaking', 'today', 'update', 'hindi',
        'exclusive', 'full', 'new', 'big', 'mp', 'cg', 'india', 'news18', 'ndtv',
        'ibc24', 'breaking', 'top', 'news', 'latest', 'update', 'live', 'today',
        'में', 'को', 'की', 'का', 'के', 'से', 'ने', 'है', 'हैं', 'था', 'थी',
        'पर', 'और', 'या', 'बड़ा', 'मध्य', 'प्रदेश', 'madhya', 'pradesh'
    }
    
    words = normalized.split()
    # Filter: min 3 chars, not stop word, not all digits
    significant = {
        word for word in words
        if len(word) >= 3 and word not in stop_words and not word.isdigit()
    }
    
    return significant


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts based on significant words.
    Returns a value between 0.0 (no similarity) and 1.0 (identical).
    """
    words1 = extract_significant_words(text1)
    words2 = extract_significant_words(text2)
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity: intersection / union
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def group_similar_videos(videos: List[Dict[str, Any]], similarity_threshold: float = 0.3, min_cluster_size: int = 4) -> List[List[Dict[str, Any]]]:
    """
    Group videos into clusters based on content similarity.
    
    Uses a greedy clustering approach with transitive similarity:
    1. For each video, find all videos with similarity >= threshold
    2. Merge clusters that have overlapping videos (transitive closure)
    3. Only keep clusters with at least min_cluster_size videos
    
    Args:
        videos: List of video dictionaries with 'title' and optionally 'description'
        similarity_threshold: Minimum similarity (0.0-1.0) to consider videos similar
        min_cluster_size: Minimum number of videos required to form a cluster
    
    Returns:
        List of video clusters (each cluster is a list of videos)
    """
    if not videos:
        return []
    
    logger.info(f"Grouping {len(videos)} videos into similar clusters (threshold={similarity_threshold}, min_size={min_cluster_size})")
    
    # Combine title and description for comparison
    video_texts = []
    for video in videos:
        title = video.get('title', '')
        description = video.get('description', '')
        combined_text = f"{title} {description}".strip()
        video_texts.append(combined_text)
    
    # Build similarity graph: for each video, find all similar videos
    similarity_graph = defaultdict(set)
    for i in range(len(videos)):
        for j in range(i + 1, len(videos)):
            similarity = calculate_text_similarity(video_texts[i], video_texts[j])
            if similarity >= similarity_threshold:
                similarity_graph[i].add(j)
                similarity_graph[j].add(i)
    
    # Find connected components using BFS (transitive similarity)
    visited = set()
    components = []
    
    def bfs(start_idx):
        """Find all connected videos starting from start_idx."""
        component = set()
        queue = [start_idx]
        visited.add(start_idx)
        component.add(start_idx)
        
        while queue:
            current = queue.pop(0)
            for neighbor in similarity_graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        
        return component
    
    # Find all connected components
    for i in range(len(videos)):
        if i not in visited:
            component = bfs(i)
            if len(component) >= min_cluster_size:
                cluster_videos = [videos[idx] for idx in component]
                components.append(cluster_videos)
                logger.debug(f"Created cluster with {len(cluster_videos)} similar videos (similarity >= {similarity_threshold})")
    
    logger.info(f"Created {len(components)} clusters from {len(videos)} videos (min_size={min_cluster_size})")
    return components


def extract_clusters(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract clusters from output.json based on content similarity.
    
    Groups similar videos across all sections and only creates clusters
    with 4+ similar videos.
    """
    if cache['clusters'] and cache['data'] == data:
        logger.debug("Returning cached clusters")
        return cache['clusters']
    
    clusters = []
    
    # Collect all videos from all sections
    all_videos = []
    sections = data.get('sections', [])
    
    for section in sections:
        items = section.get('items', [])
        all_videos.extend(items)
    
    if not all_videos:
        logger.info("No videos found in data")
        return []
    
    logger.info(f"Processing {len(all_videos)} total videos for content-based clustering")
    
    # Group similar videos (only clusters with 4+ videos)
    video_clusters = group_similar_videos(all_videos, similarity_threshold=0.3, min_cluster_size=4)
    
    # Create cluster objects for each similar video group
    for cluster_idx, items in enumerate(video_clusters):
        if len(items) < 4:  # Skip clusters with less than 4 videos
            continue
        
        # Auto-generate cluster title from video content
        generated_title = generate_cluster_title(items, "Unknown")
        
        # Generate cluster ID from title
        cluster_id = re.sub(r'[^\w\s-]', '', generated_title.lower())
        cluster_id = re.sub(r'[-\s]+', '-', cluster_id).strip('-')
        if not cluster_id:
            cluster_id = f"cluster-{cluster_idx + 1}"
        
        # Calculate trend score
        # Create a dummy section for compatibility
        dummy_section = {'section': generated_title}
        trend_score = calculate_trend_score(dummy_section, items)
        
        # Find latest update time
        latest_update = None
        for item in items:
            published_at = item.get('publishedAt')
            if published_at:
                try:
                    pub_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                    if not latest_update or pub_date > latest_update:
                        latest_update = pub_date
                except (ValueError, AttributeError):
                    pass
        
        latest_update_str = latest_update.isoformat().replace('+00:00', 'Z') if latest_update else None
        
        # Calculate derived metrics
        total_views = sum(v.get('views', 0) for v in items)
        total_likes = sum(v.get('likes', 0) for v in items)
        engagement_rate = round((total_likes / total_views * 100), 2) if total_views > 0 else 0
        trending_velocity = round(total_views / len(items), 1) if items else 0
        
        # Determine most common genre from items
        genres = [v.get('genre', 'General') for v in items]
        most_common_genre = max(set(genres), key=genres.count) if genres else 'General'
        
        cluster = {
            'clusterId': cluster_id,
            'topic': generated_title,
            'originalCategory': most_common_genre,
            'videoCount': len(items),
            'trendScore': trend_score,
            'latestUpdateAt': latest_update_str,
            'totalViews': total_views,
            'totalLikes': total_likes,
            'engagementRate': engagement_rate,
            'trendingVelocity': trending_velocity,
            'videos': items  # Include full video data
        }
        
        clusters.append(cluster)
    
    # Cache the clusters
    cache['clusters'] = clusters
    
    logger.info(f"Extracted {len(clusters)} clusters (all with 4+ similar videos)")
    return clusters


def generate_trending_cluster_json(clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate trending cluster JSON structure sorted by trend score.
    
    Creates clean, news-focused output with:
    - Short, human-readable titles
    - Top videos for each cluster
    - Essential metrics only
    
    Args:
        clusters: List of cluster dictionaries with metadata and videos
    
    Returns:
        Dictionary with 'generatedAt', 'clusterCount', and sorted 'clusters' list
    """
    # Sort clusters by trend score descending
    sorted_clusters = sorted(clusters, key=lambda x: x['trendScore'], reverse=True)
    
    # Create summary clusters with topVideos
    summary_clusters = []
    for cluster in sorted_clusters:
        # Get top 3-5 videos by views
        videos = cluster.get('videos', [])
        sorted_videos = sorted(videos, key=lambda v: v.get('views', 0), reverse=True)
        top_videos = sorted_videos[:5]  # Top 5 videos
        
        # Create clean video summaries with full fields from output.json
        top_video_summaries = []
        for index, video in enumerate(top_videos, start=1):
            video_summary = {
                'index': index,
                'videoId': video.get('videoId'),
                'title': video.get('title'),
                'description': video.get('description'),
                'channelTitle': video.get('channelTitle'),
                'publishedAt': video.get('publishedAt'),
                'thumbnail': video.get('thumbnail'),
                'durationSeconds': video.get('durationSeconds', 0),
                'durationFormatted': video.get('durationFormatted'),
                'views': video.get('views', 0),
                'likes': video.get('likes', 0),
                'videoType': video.get('videoType'),
                'genre': video.get('genre')
            }
            top_video_summaries.append(video_summary)
        
        # Create cluster summary with required fields
        summary = {
            'clusterId': cluster.get('clusterId'),
            'topic': cluster.get('topic'),  # Auto-generated short headline
            'videoCount': cluster.get('videoCount'),
            'trendScore': cluster.get('trendScore'),
            'topVideos': top_video_summaries,
            'latestUpdateAt': cluster.get('latestUpdateAt')
        }
        summary_clusters.append(summary)
    
    trending_data = {
        'generatedAt': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        'clusterCount': len(summary_clusters),  # Changed from totalClusters
        'clusters': summary_clusters
    }
    
    return trending_data


def save_trending_cluster_json(trending_data: Dict[str, Any], filename: str = TRENDING_CLUSTER_FILE) -> bool:
    """
    Save trending cluster data to a JSON file.
    
    Args:
        trending_data: The trending cluster data to save
        filename: The filename to save to (default: trending_cluster.json)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(trending_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved trending clusters to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving trending clusters to {filename}: {e}")
        return False


def error_response(message: str, status_code: int = 400) -> tuple:
    """Create a standardized error response."""
    return jsonify({
        'success': False,
        'error': message,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    }), status_code


def success_response(data: Any) -> Dict[str, Any]:
    """Create a standardized success response."""
    return jsonify({
        'success': True,
        'data': data,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    })


def require_data(f):
    """Decorator to ensure data is loaded before handling request."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        data = load_output_json()
        if not data:
            return error_response("Failed to load cluster data", 500)
        return f(data, *args, **kwargs)
    return decorated_function


# API Endpoints

@app.route('/')
def index():
    """API root endpoint with information."""
    return jsonify({
        'name': 'Cluster Trending API',
        'version': '1.0.0',
        'description': 'REST API for analyzing and serving trending news clusters',
        'endpoints': {
            'GET /api/clusters': 'Get all clusters',
            'GET /api/clusters/trending': 'Get clusters sorted by trend score',
            'GET /api/clusters/trending/top/<n>': 'Get top N trending clusters',
            'GET /api/clusters/<clusterId>': 'Get specific cluster by ID',
            'GET /api/clusters/filter?minScore=X': 'Get clusters with trend score >= X',
            'GET /api/data/output.json': 'Get raw output.json file',
            'GET /api/data/trending_cluster.json': 'Get trending clusters from GitHub',
            'GET /health': 'Health check endpoint'
        },
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    data = load_output_json()
    return jsonify({
        'status': 'healthy' if data else 'unhealthy',
        'dataLoaded': data is not None,
        'timestamp': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    })


@app.route('/api/clusters', methods=['GET'])
@require_data
def get_all_clusters(data):
    """Get all clusters."""
    logger.info("GET /api/clusters")
    clusters = extract_clusters(data)
    
    # Remove video details for summary view
    summary_clusters = []
    for cluster in clusters:
        summary = {k: v for k, v in cluster.items() if k != 'videos'}
        summary_clusters.append(summary)
    
    return success_response(summary_clusters)


@app.route('/api/clusters/trending', methods=['GET'])
@require_data
def get_trending_clusters(data):
    """Get clusters sorted by trend score (descending)."""
    logger.info("GET /api/clusters/trending")
    clusters = extract_clusters(data)
    
    # Sort by trend score descending
    sorted_clusters = sorted(clusters, key=lambda x: x['trendScore'], reverse=True)
    
    # Remove video details for summary view
    summary_clusters = []
    for cluster in sorted_clusters:
        summary = {k: v for k, v in cluster.items() if k != 'videos'}
        summary_clusters.append(summary)
    
    return success_response(summary_clusters)


@app.route('/api/clusters/trending/top/<int:n>', methods=['GET'])
@require_data
def get_top_trending_clusters(data, n):
    """Get top N trending clusters."""
    logger.info(f"GET /api/clusters/trending/top/{n}")
    
    # Validate input
    if n < 1:
        return error_response("Parameter 'n' must be at least 1", 400)
    if n > 100:
        return error_response("Parameter 'n' cannot exceed 100", 400)
    
    clusters = extract_clusters(data)
    
    # Sort by trend score descending and take top N
    sorted_clusters = sorted(clusters, key=lambda x: x['trendScore'], reverse=True)
    top_clusters = sorted_clusters[:n]
    
    # Remove video details for summary view
    summary_clusters = []
    for cluster in top_clusters:
        summary = {k: v for k, v in cluster.items() if k != 'videos'}
        summary_clusters.append(summary)
    
    return success_response(summary_clusters)


@app.route('/api/clusters/<cluster_id>', methods=['GET'])
@require_data
def get_cluster_by_id(data, cluster_id):
    """Get specific cluster by ID."""
    logger.info(f"GET /api/clusters/{cluster_id}")
    clusters = extract_clusters(data)
    
    # Find cluster with matching ID
    for cluster in clusters:
        if cluster['clusterId'] == cluster_id:
            return success_response(cluster)
    
    return error_response(f"Cluster with ID '{cluster_id}' not found", 404)


@app.route('/api/clusters/filter', methods=['GET'])
@require_data
def filter_clusters(data):
    """Filter clusters by minimum trend score."""
    logger.info("GET /api/clusters/filter")
    
    # Get minScore parameter
    min_score_str = request.args.get('minScore')
    if not min_score_str:
        return error_response("Missing required parameter 'minScore'", 400)
    
    try:
        min_score = float(min_score_str)
    except ValueError:
        return error_response("Parameter 'minScore' must be a number", 400)
    
    if min_score < 0 or min_score > 100:
        return error_response("Parameter 'minScore' must be between 0 and 100", 400)
    
    clusters = extract_clusters(data)
    
    # Filter clusters
    filtered_clusters = [c for c in clusters if c['trendScore'] >= min_score]
    
    # Sort by trend score descending
    filtered_clusters.sort(key=lambda x: x['trendScore'], reverse=True)
    
    # Remove video details for summary view
    summary_clusters = []
    for cluster in filtered_clusters:
        summary = {k: v for k, v in cluster.items() if k != 'videos'}
        summary_clusters.append(summary)
    
    logger.info(f"Filtered {len(summary_clusters)} clusters with minScore >= {min_score}")
    return success_response(summary_clusters)


@app.route('/api/data/output.json', methods=['GET'])
def get_output_json():
    """Expose the raw output.json file."""
    logger.info("GET /api/data/output.json")
    
    if not os.path.exists(OUTPUT_FILE):
        return error_response(f"{OUTPUT_FILE} not found", 404)
    
    try:
        return send_file(OUTPUT_FILE, mimetype='application/json')
    except Exception as e:
        logger.error(f"Error serving {OUTPUT_FILE}: {e}")
        return error_response(f"Error serving file: {str(e)}", 500)


@app.route('/api/data/trending_cluster.json', methods=['GET'])
def get_trending_cluster_json():
    """
    Fetch and return the trending_cluster.json file from GitHub.
    
    This endpoint fetches the file from the GitHub repository's main branch
    using the raw content URL: https://raw.githubusercontent.com/MridulEcolab/TestPipeline/main/trending_cluster.json
    """
    logger.info("GET /api/data/trending_cluster.json")
    
    # Construct the full GitHub URL using urljoin for robustness
    github_url = urljoin(GITHUB_RAW_BASE_URL, TRENDING_CLUSTER_FILE)
    
    try:
        # Fetch the file from GitHub
        logger.info(f"Fetching from: {github_url}")
        response = requests.get(github_url, timeout=GITHUB_REQUEST_TIMEOUT)
        
        if response.status_code == 404:
            return error_response(f"File not found on GitHub: {TRENDING_CLUSTER_FILE}", 404)
        
        if response.status_code == 403:
            logger.warning("GitHub rate limit may be exceeded")
            return error_response("GitHub API rate limit exceeded or access forbidden", 403)
        
        if response.status_code != 200:
            logger.error(f"GitHub returned status {response.status_code}")
            return error_response(f"Failed to fetch from GitHub. Status: {response.status_code}", 502)
        
        # Parse and validate the JSON
        try:
            trending_data = response.json()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from GitHub: {e}")
            return error_response("Invalid JSON format from GitHub", 500)
        
        # Return the data with success response format
        return success_response(trending_data)
        
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching from GitHub: {github_url}")
        return error_response("Request timeout while fetching from GitHub", 504)
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching from GitHub: {e}")
        return error_response(f"Error fetching from GitHub: {str(e)}", 500)


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return error_response("Endpoint not found", 404)


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {error}")
    return error_response("Internal server error", 500)


def main():
    """Run the Flask server."""
    port = int(os.environ.get('PORT', DEFAULT_PORT))
    host = os.environ.get('HOST', DEFAULT_HOST)
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Cluster API server on {host}:{port}")
    logger.info(f"Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
