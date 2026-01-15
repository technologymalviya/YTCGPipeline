#!/usr/bin/env python3
"""
Test cases for genre classification functionality.
Tests OpenAI classification with mock data without hitting YouTube API.
"""

import os
import sys
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta, timezone

# Add current directory to path
sys.path.insert(0, '.')

from generate_json import (
    classify_genre_keyword_based,
    classify_genre_with_openai,
    add_genres_to_feed,
    _check_openai_rate_limits,
    _check_openai_token_limit,
    OPENAI_MODEL,
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT,
    OPENAI_SAFETY_MARGIN_RPM,
    OPENAI_SAFETY_MARGIN_TPM,
    OPENAI_MAX_EXECUTION_TIME,
    GENRE_CRIME,
    GENRE_TRAFFIC,
    GENRE_JOBS,
    GENRE_EVENTS,
    GENRE_CIVIC,
    GENRE_POLITICS,
    GENRE_GENERAL,
    GENRE_LIVE,
    GENRE_SCHEDULED,
    VIDEO_TYPE_LIVE,
    VIDEO_TYPE_VOD,
    VIDEO_TYPE_SCHEDULED,
)


class TestKeywordClassification(unittest.TestCase):
    """Test keyword-based genre classification."""
    
    def test_crime_classification(self):
        """Test crime genre classification."""
        test_cases = [
            ("Murder case in city", "A murder happened", GENRE_CRIME),
            ("हत्या का मामला", "कत्ल", GENRE_CRIME),
            ("Police arrested suspect", "गिरफ्तार", GENRE_CRIME),
            ("Kidnapping case reported", "अपहरण", GENRE_CRIME),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")
    
    def test_traffic_classification(self):
        """Test traffic genre classification."""
        test_cases = [
            ("Road accident on highway", "कार हादसा", GENRE_TRAFFIC),
            ("Traffic jam in city", "सड़क जाम", GENRE_TRAFFIC),
            ("Car crash on expressway", "दुर्घटना", GENRE_TRAFFIC),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")
    
    def test_jobs_classification(self):
        """Test jobs genre classification."""
        test_cases = [
            ("Job vacancy notification", "नौकरी सूचना", GENRE_JOBS),
            ("Government job recruitment", "सरकारी नौकरी", GENRE_JOBS),
            ("Walk-in interview tomorrow", "भर्ती इंटरव्यू", GENRE_JOBS),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")
    
    def test_events_classification(self):
        """Test events genre classification."""
        test_cases = [
            ("Republic Day celebration", "26 जनवरी", GENRE_EVENTS),
            ("Festival celebration", "त्योहार", GENRE_EVENTS),
            ("Makar Sankranti festival", "मकर संक्रांति", GENRE_EVENTS),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")
    
    def test_civic_classification(self):
        """Test civic genre classification."""
        test_cases = [
            ("Municipal corporation notice", "नगर निगम", GENRE_CIVIC),
            ("Road construction work", "सड़क निर्माण", GENRE_CIVIC),
            ("NRDA encroachment removal", "अतिक्रमण हटाने", GENRE_CIVIC),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")
    
    def test_politics_classification(self):
        """Test politics genre classification."""
        test_cases = [
            ("PM Modi speech today", "प्रधानमंत्री", GENRE_POLITICS),
            ("Election campaign rally", "चुनाव", GENRE_POLITICS),
            ("CM announcement", "मुख्यमंत्री", GENRE_POLITICS),
            ("Owaisi statement", "ओवैसी", GENRE_POLITICS),
            ("Land jihad issue", "लैंड जिहाद", GENRE_POLITICS),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")
    
    def test_general_classification(self):
        """Test general genre classification for non-specific content."""
        test_cases = [
            ("Weather update today", "मौसम", GENRE_GENERAL),
            ("General news update", "सामान्य खबर", GENRE_GENERAL),
        ]
        
        for title, description, expected in test_cases:
            with self.subTest(title=title):
                result = classify_genre_keyword_based(title, description)
                self.assertEqual(result, expected, f"Expected {expected}, got {result} for '{title}'")


class TestOpenAIClassification(unittest.TestCase):
    """Test OpenAI classification with mocked API calls."""
    
    def setUp(self):
        """Set up test environment."""
        # Set a mock API key
        os.environ['OPENAI_API_KEY'] = 'sk-test-mock-key-12345'
        # Reset rate limit tracker
        from generate_json import _openai_rate_limit_tracker
        _openai_rate_limit_tracker["requests"] = []
        _openai_rate_limit_tracker["tokens"] = []
        _openai_rate_limit_tracker["last_reset"] = time.time()
    
    def tearDown(self):
        """Clean up after tests."""
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
    
    @patch('generate_json.openai')
    def test_openai_successful_classification(self, mock_openai):
        """Test successful OpenAI classification."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Crime"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client
        
        result = classify_genre_with_openai("Murder case reported", "A crime happened")
        
        self.assertEqual(result, GENRE_CRIME)
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('generate_json.openai')
    def test_openai_rate_limit_error(self, mock_openai):
        """Test OpenAI rate limit error handling."""
        # Mock OpenAI client to raise RateLimitError
        mock_client = MagicMock()
        mock_openai.RateLimitError = Exception
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        mock_openai.OpenAI.return_value = mock_client
        
        result = classify_genre_with_openai("Test title", "Test description")
        
        self.assertIsNone(result)  # Should return None on rate limit error
    
    @patch('generate_json.openai')
    def test_openai_connection_error(self, mock_openai):
        """Test OpenAI connection error handling."""
        # Mock OpenAI client to raise APIConnectionError
        mock_client = MagicMock()
        mock_openai.APIConnectionError = Exception
        mock_client.chat.completions.create.side_effect = Exception("Connection error")
        mock_openai.OpenAI.return_value = mock_client
        
        result = classify_genre_with_openai("Test title", "Test description")
        
        # Should return None after retry
        self.assertIsNone(result)
    
    @patch('generate_json.openai')
    def test_openai_timeout(self, mock_openai):
        """Test OpenAI timeout handling."""
        # Mock OpenAI client to simulate timeout
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TimeoutError("Request timeout")
        mock_openai.OpenAI.return_value = mock_client
        
        result = classify_genre_with_openai("Test title", "Test description")
        
        # Should return None on timeout
        self.assertIsNone(result)
    
    def test_openai_no_api_key(self):
        """Test OpenAI classification without API key."""
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
        
        result = classify_genre_with_openai("Test title", "Test description")
        
        self.assertIsNone(result)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        from generate_json import _openai_rate_limit_tracker
        _openai_rate_limit_tracker["requests"] = []
        _openai_rate_limit_tracker["tokens"] = []
        _openai_rate_limit_tracker["last_reset"] = time.time()
    
    def test_rate_limit_check_under_limit(self):
        """Test rate limit check when under limit."""
        result = _check_openai_rate_limits()
        self.assertTrue(result)
    
    def test_rate_limit_check_at_limit(self):
        """Test rate limit check when at safety threshold."""
        from generate_json import _openai_rate_limit_tracker, OPENAI_RATE_LIMIT_RPM, OPENAI_SAFETY_MARGIN_RPM
        
        # Add requests up to safety threshold
        safe_limit = int(OPENAI_RATE_LIMIT_RPM * OPENAI_SAFETY_MARGIN_RPM)
        current_time = time.time()
        for i in range(safe_limit):
            _openai_rate_limit_tracker["requests"].append(current_time - 10)  # Recent requests
        
        result = _check_openai_rate_limits()
        self.assertFalse(result)  # Should fail at safety threshold
    
    def test_token_limit_check_under_limit(self):
        """Test token limit check when under limit."""
        result = _check_openai_token_limit(1000)
        self.assertTrue(result)
    
    def test_token_limit_check_at_limit(self):
        """Test token limit check when at safety threshold."""
        from generate_json import _openai_rate_limit_tracker, OPENAI_RATE_LIMIT_TPM, OPENAI_SAFETY_MARGIN_TPM
        
        # Add tokens up to safety threshold
        safe_limit = int(OPENAI_RATE_LIMIT_TPM * OPENAI_SAFETY_MARGIN_TPM)
        current_time = time.time()
        _openai_rate_limit_tracker["tokens"].append({"time": current_time - 10, "tokens": safe_limit})
        
        result = _check_openai_token_limit(1)
        self.assertFalse(result)  # Should fail at safety threshold


class TestAddGenresToFeed(unittest.TestCase):
    """Test add_genres_to_feed function with mock data."""
    
    def setUp(self):
        """Set up test environment."""
        # Set a mock API key
        os.environ['OPENAI_API_KEY'] = 'sk-test-mock-key-12345'
        # Reset rate limit tracker
        from generate_json import _openai_rate_limit_tracker
        _openai_rate_limit_tracker["requests"] = []
        _openai_rate_limit_tracker["tokens"] = []
        _openai_rate_limit_tracker["last_reset"] = time.time()
    
    def tearDown(self):
        """Clean up after tests."""
        if 'OPENAI_API_KEY' in os.environ:
            del os.environ['OPENAI_API_KEY']
    
    def create_mock_video(self, video_id, title, description, published_at, video_type=VIDEO_TYPE_VOD):
        """Helper to create mock video data."""
        return {
            "videoId": video_id,
            "title": title,
            "description": description,
            "publishedAt": published_at,
            "videoType": video_type,
            "channelTitle": "Test Channel",
            "thumbnail": "https://example.com/thumb.jpg",
            "durationSeconds": 100,
            "durationFormatted": "01:40",
            "views": 1000,
            "likes": 50,
        }
    
    def test_live_video_classification(self):
        """Test Live video classification."""
        now = datetime.now(timezone.utc)
        feed = [
            self.create_mock_video("live1", "Live News", "Live broadcast", now.isoformat(), VIDEO_TYPE_LIVE)
        ]
        
        result = add_genres_to_feed(feed)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["genre"], GENRE_LIVE)
    
    def test_scheduled_video_classification(self):
        """Test Scheduled video classification."""
        now = datetime.now(timezone.utc)
        feed = [
            self.create_mock_video("sched1", "Upcoming Event", "Scheduled", now.isoformat(), VIDEO_TYPE_SCHEDULED)
        ]
        
        result = add_genres_to_feed(feed)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["genre"], GENRE_SCHEDULED)
    
    def test_keyword_based_classification(self):
        """Test keyword-based classification for older videos."""
        # Video published 2 hours ago (not recent)
        two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        feed = [
            self.create_mock_video("old1", "Murder case reported", "Crime news", two_hours_ago)
        ]
        
        result = add_genres_to_feed(feed)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["genre"], GENRE_CRIME)  # Should use keyword-based
    
    @patch('generate_json.classify_genre_with_openai')
    def test_openai_classification_for_recent_videos(self, mock_openai):
        """Test OpenAI classification for recent videos."""
        # Video published 30 minutes ago (recent)
        thirty_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        feed = [
            self.create_mock_video("recent1", "Breaking news", "Latest update", thirty_min_ago)
        ]
        
        # Mock OpenAI to return Politics
        mock_openai.return_value = GENRE_POLITICS
        
        result = add_genres_to_feed(feed)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["genre"], GENRE_POLITICS)
        mock_openai.assert_called_once()
    
    @patch('generate_json.classify_genre_with_openai')
    def test_circuit_breaker_on_openai_failure(self, mock_openai):
        """Test circuit breaker activates on OpenAI failure."""
        # Recent videos
        thirty_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        feed = [
            self.create_mock_video("recent1", "News 1", "Description 1", thirty_min_ago),
            self.create_mock_video("recent2", "News 2", "Description 2", thirty_min_ago),
            self.create_mock_video("recent3", "News 3", "Description 3", thirty_min_ago),
        ]
        
        # Mock OpenAI to fail on first call, then return None
        mock_openai.return_value = None
        
        result = add_genres_to_feed(feed)
        
        # All videos should be classified (using keyword-based after first failure)
        self.assertEqual(len(result), 3)
        # First call should attempt OpenAI, subsequent should use keyword-based
        self.assertLessEqual(mock_openai.call_count, 1)  # Circuit breaker should stop after first failure
    
    @patch('generate_json.classify_genre_with_openai')
    def test_rate_limit_safety_threshold(self, mock_openai):
        """Test rate limit safety threshold activation."""
        from generate_json import _openai_rate_limit_tracker, OPENAI_RATE_LIMIT_RPM, OPENAI_SAFETY_MARGIN_RPM
        
        # Fill up rate limit tracker to safety threshold
        safe_limit = int(OPENAI_RATE_LIMIT_RPM * OPENAI_SAFETY_MARGIN_RPM)
        current_time = time.time()
        _openai_rate_limit_tracker["requests"] = [current_time - 10] * safe_limit
        
        # Recent video
        thirty_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        feed = [
            self.create_mock_video("recent1", "News 1", "Description 1", thirty_min_ago)
        ]
        
        result = add_genres_to_feed(feed)
        
        # Should use keyword-based due to rate limit
        self.assertEqual(len(result), 1)
        # OpenAI should not be called due to rate limit check
        mock_openai.assert_not_called()
    
    @patch('generate_json.classify_genre_with_openai')
    def test_execution_time_limit(self, mock_openai):
        """Test execution time limit enforcement."""
        # Use real time but patch time.sleep to speed up test
        # We'll manually check the time limit logic by using a very short limit
        from generate_json import OPENAI_MAX_EXECUTION_TIME
        
        # Create a feed with recent videos
        thirty_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        feed = [
            self.create_mock_video("recent1", "News 1", "Description 1", thirty_min_ago),
        ]
        
        # Mock OpenAI to take a long time (simulate slow API)
        def slow_openai(*args, **kwargs):
            time.sleep(0.1)  # Small delay to simulate API call
            return None  # Return None to trigger fallback
        
        mock_openai.side_effect = slow_openai
        
        # Temporarily set a very short execution time limit for this test
        import generate_json
        original_limit = generate_json.OPENAI_MAX_EXECUTION_TIME
        generate_json.OPENAI_MAX_EXECUTION_TIME = 0.05  # 50ms limit
        
        try:
            result = add_genres_to_feed(feed)
            
            # Video should be classified (using keyword-based after time limit)
            self.assertEqual(len(result), 1)
            # OpenAI might be called once before time limit, but should stop after
            self.assertLessEqual(mock_openai.call_count, 1)
        finally:
            # Restore original limit
            generate_json.OPENAI_MAX_EXECUTION_TIME = original_limit
    
    def test_mixed_video_types(self):
        """Test classification with mixed video types."""
        now = datetime.now(timezone.utc)
        thirty_min_ago = (now - timedelta(minutes=30)).isoformat()
        two_hours_ago = (now - timedelta(hours=2)).isoformat()
        
        feed = [
            self.create_mock_video("live1", "Live", "Live", now.isoformat(), VIDEO_TYPE_LIVE),
            self.create_mock_video("sched1", "Scheduled", "Scheduled", now.isoformat(), VIDEO_TYPE_SCHEDULED),
            self.create_mock_video("recent1", "Crime news", "Murder case", thirty_min_ago),
            self.create_mock_video("old1", "Traffic accident", "Road crash", two_hours_ago),
        ]
        
        result = add_genres_to_feed(feed)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0]["genre"], GENRE_LIVE)
        self.assertEqual(result[1]["genre"], GENRE_SCHEDULED)
        # Recent video might use OpenAI (if available) or keyword-based
        # Old video should use keyword-based
        self.assertEqual(result[3]["genre"], GENRE_TRAFFIC)


def run_tests():
    """Run all test cases."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestKeywordClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestOpenAIClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiting))
    suite.addTests(loader.loadTestsFromTestCase(TestAddGenresToFeed))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
