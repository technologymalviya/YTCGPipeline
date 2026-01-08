#!/usr/bin/env python3
"""
Simple tests for the Cluster API
Tests basic functionality of all endpoints
"""

import json
import os
import sys
import time
import requests
from subprocess import Popen, PIPE
import signal

# Configuration
API_HOST = "localhost"
API_PORT = 5000
BASE_URL = f"http://{API_HOST}:{API_PORT}"
STARTUP_TIMEOUT = 5
TEST_TIMEOUT = 10


class TestClusterAPI:
    """Test suite for Cluster API"""
    
    def __init__(self):
        self.server_process = None
        self.passed_tests = 0
        self.failed_tests = 0
        
    def start_server(self):
        """Start the API server for testing"""
        print(f"Starting API server on {API_HOST}:{API_PORT}...")
        
        env = os.environ.copy()
        env['HOST'] = API_HOST
        env['PORT'] = str(API_PORT)
        
        self.server_process = Popen(
            ['python', 'cluster_api.py'],
            stdout=PIPE,
            stderr=PIPE,
            env=env
        )
        
        # Wait for server to start
        for _ in range(STARTUP_TIMEOUT * 2):
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=1)
                if response.status_code == 200:
                    print("✓ Server started successfully")
                    return True
            except requests.exceptions.RequestException:
                time.sleep(0.5)
        
        print("✗ Server failed to start")
        return False
    
    def stop_server(self):
        """Stop the API server"""
        if self.server_process:
            print("\nStopping API server...")
            self.server_process.send_signal(signal.SIGTERM)
            self.server_process.wait(timeout=5)
            print("✓ Server stopped")
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        print("\n--- Testing /health endpoint ---")
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=TEST_TIMEOUT)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert 'status' in data, "Missing 'status' field"
            assert data['status'] == 'healthy', f"Expected healthy status, got {data['status']}"
            assert data['dataLoaded'] is True, "Data should be loaded"
            
            print("✓ Health endpoint test passed")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Health endpoint test failed: {e}")
            self.failed_tests += 1
    
    def test_get_all_clusters(self):
        """Test GET /api/clusters endpoint"""
        print("\n--- Testing GET /api/clusters ---")
        try:
            response = requests.get(f"{BASE_URL}/api/clusters", timeout=TEST_TIMEOUT)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert data['success'] is True, "Response should indicate success"
            assert isinstance(data['data'], list), "Data should be a list"
            assert len(data['data']) > 0, "Should have at least one cluster"
            
            # Check cluster structure
            cluster = data['data'][0]
            required_fields = ['clusterId', 'topic', 'videoCount', 'trendScore', 
                             'totalViews', 'totalLikes', 'engagementRate', 'trendingVelocity']
            for field in required_fields:
                assert field in cluster, f"Missing field: {field}"
            
            print(f"✓ Get all clusters test passed ({len(data['data'])} clusters found)")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Get all clusters test failed: {e}")
            self.failed_tests += 1
    
    def test_get_trending_clusters(self):
        """Test GET /api/clusters/trending endpoint"""
        print("\n--- Testing GET /api/clusters/trending ---")
        try:
            response = requests.get(f"{BASE_URL}/api/clusters/trending", timeout=TEST_TIMEOUT)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert data['success'] is True, "Response should indicate success"
            assert isinstance(data['data'], list), "Data should be a list"
            
            # Check that results are sorted by trendScore descending
            if len(data['data']) > 1:
                for i in range(len(data['data']) - 1):
                    current = data['data'][i]['trendScore']
                    next_item = data['data'][i + 1]['trendScore']
                    assert current >= next_item, "Results should be sorted by trendScore descending"
            
            print("✓ Get trending clusters test passed")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Get trending clusters test failed: {e}")
            self.failed_tests += 1
    
    def test_get_top_n_trending(self):
        """Test GET /api/clusters/trending/top/:n endpoint"""
        print("\n--- Testing GET /api/clusters/trending/top/3 ---")
        try:
            response = requests.get(f"{BASE_URL}/api/clusters/trending/top/3", timeout=TEST_TIMEOUT)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert data['success'] is True, "Response should indicate success"
            assert len(data['data']) <= 3, "Should return at most 3 clusters"
            
            print(f"✓ Get top N trending test passed (returned {len(data['data'])} clusters)")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Get top N trending test failed: {e}")
            self.failed_tests += 1
    
    def test_get_cluster_by_id(self):
        """Test GET /api/clusters/:clusterId endpoint"""
        print("\n--- Testing GET /api/clusters/:clusterId ---")
        try:
            # First get all clusters to find a valid ID
            response = requests.get(f"{BASE_URL}/api/clusters", timeout=TEST_TIMEOUT)
            clusters = response.json()['data']
            
            if len(clusters) > 0:
                cluster_id = clusters[0]['clusterId']
                
                # Now test getting that specific cluster
                response = requests.get(f"{BASE_URL}/api/clusters/{cluster_id}", timeout=TEST_TIMEOUT)
                assert response.status_code == 200, f"Expected 200, got {response.status_code}"
                
                data = response.json()
                assert data['success'] is True, "Response should indicate success"
                assert data['data']['clusterId'] == cluster_id, "Should return correct cluster"
                assert 'videos' in data['data'], "Detailed view should include videos"
                
                print(f"✓ Get cluster by ID test passed (cluster: {cluster_id})")
                self.passed_tests += 1
            else:
                print("⊘ Skipped: No clusters available")
        except Exception as e:
            print(f"✗ Get cluster by ID test failed: {e}")
            self.failed_tests += 1
    
    def test_filter_clusters(self):
        """Test GET /api/clusters/filter?minScore=X endpoint"""
        print("\n--- Testing GET /api/clusters/filter?minScore=50 ---")
        try:
            response = requests.get(f"{BASE_URL}/api/clusters/filter?minScore=50", timeout=TEST_TIMEOUT)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert data['success'] is True, "Response should indicate success"
            
            # Check that all returned clusters have score >= 50
            for cluster in data['data']:
                assert cluster['trendScore'] >= 50, f"Cluster {cluster['clusterId']} has score {cluster['trendScore']} < 50"
            
            print(f"✓ Filter clusters test passed ({len(data['data'])} clusters with score >= 50)")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Filter clusters test failed: {e}")
            self.failed_tests += 1
    
    def test_error_handling(self):
        """Test error handling for invalid requests"""
        print("\n--- Testing error handling ---")
        try:
            # Test non-existent cluster
            response = requests.get(f"{BASE_URL}/api/clusters/nonexistent", timeout=TEST_TIMEOUT)
            assert response.status_code == 404, f"Expected 404, got {response.status_code}"
            data = response.json()
            assert data['success'] is False, "Error response should have success=false"
            
            # Test invalid minScore parameter
            response = requests.get(f"{BASE_URL}/api/clusters/filter?minScore=abc", timeout=TEST_TIMEOUT)
            assert response.status_code == 400, f"Expected 400, got {response.status_code}"
            
            # Test invalid N value
            response = requests.get(f"{BASE_URL}/api/clusters/trending/top/0", timeout=TEST_TIMEOUT)
            assert response.status_code == 400, f"Expected 400, got {response.status_code}"
            
            print("✓ Error handling test passed")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            self.failed_tests += 1
    
    def test_output_json_endpoint(self):
        """Test GET /api/data/output.json endpoint"""
        print("\n--- Testing GET /api/data/output.json ---")
        try:
            response = requests.get(f"{BASE_URL}/api/data/output.json", timeout=TEST_TIMEOUT)
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            # Should be valid JSON
            data = response.json()
            assert 'sections' in data, "Should contain sections"
            assert 'generatedAt' in data, "Should contain generatedAt timestamp"
            
            print("✓ Output JSON endpoint test passed")
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Output JSON endpoint test failed: {e}")
            self.failed_tests += 1
    
    def test_trending_cluster_json_endpoint(self):
        """Test GET /api/data/trending_cluster.json endpoint (fetches from GitHub)"""
        print("\n--- Testing GET /api/data/trending_cluster.json ---")
        try:
            response = requests.get(f"{BASE_URL}/api/data/trending_cluster.json", timeout=TEST_TIMEOUT)
            
            # The endpoint now fetches from GitHub, so it should return either:
            # 1. Success (200) with data if file exists on GitHub
            # 2. 404 if file doesn't exist on GitHub yet
            # Both are valid responses
            
            if response.status_code == 200:
                data = response.json()
                assert data['success'] is True, "Response should indicate success"
                
                # If successful, verify the structure
                trending_data = data['data']
                assert 'clusters' in trending_data, "Should contain clusters"
                assert 'generatedAt' in trending_data, "Should contain generatedAt timestamp"
                assert 'clusterCount' in trending_data, "Should contain clusterCount"
                
                # Verify clusters are sorted by trendScore descending
                clusters = trending_data['clusters']
                if len(clusters) > 1:
                    for i in range(len(clusters) - 1):
                        current = clusters[i]['trendScore']
                        next_item = clusters[i + 1]['trendScore']
                        assert current >= next_item, "Clusters should be sorted by trendScore descending"
                
                # Verify each cluster has topVideos
                if clusters:
                    assert 'topVideos' in clusters[0], "Each cluster should contain topVideos"
                
                print(f"✓ Trending cluster endpoint test passed (file found on GitHub, {trending_data['clusterCount']} clusters)")
            elif response.status_code == 404:
                data = response.json()
                assert data['success'] is False, "Error response should have success=false"
                print("✓ Trending cluster endpoint test passed (file not yet on GitHub - expected)")
            else:
                raise AssertionError(f"Unexpected status code: {response.status_code}")
            
            self.passed_tests += 1
        except Exception as e:
            print(f"✗ Trending cluster endpoint test failed: {e}")
            self.failed_tests += 1
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("CLUSTER API TEST SUITE")
        print("=" * 60)
        
        # Start server
        if not self.start_server():
            print("\nERROR: Could not start API server")
            return 1
        
        try:
            # Run tests
            self.test_health_endpoint()
            self.test_get_all_clusters()
            self.test_get_trending_clusters()
            self.test_get_top_n_trending()
            self.test_get_cluster_by_id()
            self.test_filter_clusters()
            self.test_error_handling()
            self.test_output_json_endpoint()
            self.test_trending_cluster_json_endpoint()
        finally:
            # Always stop server
            self.stop_server()
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        total_tests = self.passed_tests + self.failed_tests
        print(f"Total tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        
        if self.failed_tests == 0:
            print("\n✓ ALL TESTS PASSED")
            return 0
        else:
            print(f"\n✗ {self.failed_tests} TEST(S) FAILED")
            return 1


def main():
    """Main test entry point"""
    # Check if output.json exists
    if not os.path.exists('output.json'):
        print("ERROR: output.json not found. Please run generate_json.py first.")
        return 1
    
    tester = TestClusterAPI()
    return tester.run_all_tests()


if __name__ == '__main__':
    sys.exit(main())
