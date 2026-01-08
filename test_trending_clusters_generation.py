#!/usr/bin/env python3
"""
Test script to verify that trending_cluster.json generation works correctly
"""

import json
import os
import sys
import subprocess
from datetime import datetime, timezone


def test_trending_clusters_generation():
    """Test that generate_trending_clusters.py works correctly"""
    print("=" * 60)
    print("Testing Trending Clusters Generation")
    print("=" * 60)
    print()
    
    # Check that output.json exists
    if not os.path.exists('output.json'):
        print("❌ FAILED: output.json not found")
        return False
    print("✅ output.json exists")
    
    # Load output.json and get its timestamp
    with open('output.json', 'r') as f:
        output_data = json.load(f)
    output_timestamp = output_data.get('generatedAt')
    print(f"   Generated at: {output_timestamp}")
    
    # Run the generation script
    print("\nRunning generate_trending_clusters.py...")
    result = subprocess.run(
        ['python3', 'generate_trending_clusters.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ FAILED: Script exited with code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        return False
    print("✅ Script executed successfully")
    
    # Check that trending_cluster.json was created/updated
    if not os.path.exists('trending_cluster.json'):
        print("❌ FAILED: trending_cluster.json not created")
        return False
    print("✅ trending_cluster.json exists")
    
    # Load and validate trending_cluster.json
    with open('trending_cluster.json', 'r') as f:
        trending_data = json.load(f)
    
    # Validate structure
    required_fields = ['generatedAt', 'clusterCount', 'clusters']
    for field in required_fields:
        if field not in trending_data:
            print(f"❌ FAILED: Missing required field '{field}'")
            return False
    print(f"✅ All required fields present")
    
    # Validate timestamp is recent
    trending_timestamp = trending_data.get('generatedAt')
    print(f"   Generated at: {trending_timestamp}")
    
    # Parse timestamps
    try:
        trending_dt = datetime.fromisoformat(trending_timestamp.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        time_diff = (now - trending_dt).total_seconds()
        
        if time_diff > 300:  # More than 5 minutes old
            print(f"⚠️  WARNING: trending_cluster.json is {time_diff:.0f} seconds old")
        else:
            print(f"✅ Timestamp is recent ({time_diff:.1f} seconds ago)")
    except Exception as e:
        print(f"⚠️  WARNING: Could not parse timestamp: {e}")
    
    # Validate cluster structure
    if not isinstance(trending_data['clusters'], list):
        print("❌ FAILED: 'clusters' must be a list")
        return False
    
    cluster_count = len(trending_data['clusters'])
    declared_count = trending_data['clusterCount']
    
    if cluster_count != declared_count:
        print(f"❌ FAILED: Cluster count mismatch. Declared: {declared_count}, Actual: {cluster_count}")
        return False
    print(f"✅ Cluster count is consistent: {cluster_count}")
    
    # Validate first cluster if exists
    if cluster_count > 0:
        cluster = trending_data['clusters'][0]
        required_cluster_fields = ['clusterId', 'topic', 'videoCount', 'trendScore', 'topVideos', 'latestUpdateAt']
        
        for field in required_cluster_fields:
            if field not in cluster:
                print(f"❌ FAILED: Missing cluster field '{field}'")
                return False
        
        print(f"✅ Cluster structure is valid")
        print(f"   Top cluster: '{cluster['topic']}' (score: {cluster['trendScore']}, videos: {cluster['videoCount']})")
        
        # Validate topVideos
        if not isinstance(cluster['topVideos'], list):
            print("❌ FAILED: 'topVideos' must be a list")
            return False
        
        if cluster['topVideos']:
            video = cluster['topVideos'][0]
            required_video_fields = ['index', 'videoId', 'title', 'views', 'likes']
            for field in required_video_fields:
                if field not in video:
                    print(f"❌ FAILED: Missing video field '{field}'")
                    return False
            print(f"✅ Top video structure is valid")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_trending_clusters_generation()
    sys.exit(0 if success else 1)
