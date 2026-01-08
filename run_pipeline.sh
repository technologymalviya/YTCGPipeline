#!/bin/bash
# Script to run the YouTube aggregator pipeline locally
# This script runs the same logic that GitHub Actions executes

echo "=========================================="
echo "Running YouTube Aggregator Pipeline"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Check if requirements are installed
echo ""
echo "Installing dependencies..."
python3 -m pip install -r requirements.txt -q

# Run the pipeline
echo ""
echo "Executing pipeline script..."
python3 generate_json.py

# Generate trending clusters
echo ""
echo "Generating trending clusters..."
python3 generate_trending_clusters.py

# Show results
echo ""
echo "=========================================="
echo "Pipeline Execution Complete"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - output.json"
echo "  - trending_cluster.json"
echo ""
echo "To view the output:"
echo "  cat output.json"
echo "  cat trending_cluster.json"
echo ""

# Get repository info dynamically
REPO_URL=$(git config --get remote.origin.url 2>/dev/null)
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")

if [ -n "$REPO_URL" ]; then
    # Convert SSH URL to HTTPS if needed
    REPO_URL=$(echo "$REPO_URL" | sed 's/git@github.com:/https:\/\/github.com\//' | sed 's/\.git$//')
    echo "To view it online (after GitHub Actions runs):"
    echo "  ${REPO_URL}/raw/${BRANCH}/output.json"
else
    echo "To view it online (after GitHub Actions runs):"
    echo "  https://raw.githubusercontent.com/<owner>/<repo>/<branch>/output.json"
fi
echo ""
echo "To validate the JSON output:"
echo "  python3 validate_json.py"
echo ""
