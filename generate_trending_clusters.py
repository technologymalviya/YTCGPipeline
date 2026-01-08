#!/usr/bin/env python3
"""
Generate trending_cluster.json from output.json
This script is called after generate_json.py to create trending clusters.
"""

import sys
import os

# Add the current directory to the path so we can import cluster_api
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cluster_api import (
    load_output_json,
    extract_clusters,
    generate_trending_cluster_json,
    save_trending_cluster_json,
    OUTPUT_FILE,
    TRENDING_CLUSTER_FILE
)

def main():
    """Main function to generate trending clusters."""
    print("========================================")
    print("Generating trending_cluster.json")
    print("========================================")
    print()
    
    # Load output.json
    print(f"Loading {OUTPUT_FILE}...")
    data = load_output_json()
    
    if not data:
        print(f"Error: Failed to load {OUTPUT_FILE}")
        sys.exit(1)
    
    print(f"✓ Successfully loaded {OUTPUT_FILE}")
    
    # Extract clusters from the data
    print("Extracting clusters...")
    clusters = extract_clusters(data)
    print(f"✓ Extracted {len(clusters)} clusters")
    
    # Generate trending cluster JSON
    print("Generating trending cluster data...")
    trending_data = generate_trending_cluster_json(clusters)
    print(f"✓ Generated trending data with {trending_data['clusterCount']} clusters")
    
    # Save to file
    print(f"Saving to {TRENDING_CLUSTER_FILE}...")
    success = save_trending_cluster_json(trending_data)
    
    if not success:
        print(f"Error: Failed to save {TRENDING_CLUSTER_FILE}")
        sys.exit(1)
    
    print(f"✓ Successfully saved {TRENDING_CLUSTER_FILE}")
    print()
    print("========================================")
    print("Trending cluster generation complete!")
    print("========================================")
    print()
    print(f"Generated at: {trending_data['generatedAt']}")
    print(f"Total clusters: {trending_data['clusterCount']}")
    print()
    
    # Print top 3 trending clusters
    if trending_data['clusters']:
        print("Top 3 trending clusters:")
        for i, cluster in enumerate(trending_data['clusters'][:3], 1):
            print(f"  {i}. {cluster['topic']} (score: {cluster['trendScore']}, videos: {cluster['videoCount']})")
    print()

if __name__ == "__main__":
    main()
