#!/usr/bin/env python3
"""
Unit tests for HTTP 400 API key error handling
Tests the new is_invalid_api_key_error function and error handling logic.
"""

import json
import unittest
from unittest.mock import Mock


class TestHTTP400Handling(unittest.TestCase):
    """Test cases for HTTP 400 invalid API key error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Import the function we're testing
        import generate_json
        self.is_invalid_api_key_error = generate_json.is_invalid_api_key_error
    
    def test_http_400_with_key_invalid_reason(self):
        """Test that HTTP 400 with keyInvalid reason is detected."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "errors": [
                    {
                        "domain": "global",
                        "reason": "keyInvalid",
                        "message": "Bad Request"
                    }
                ]
            }
        }
        
        self.assertTrue(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_with_bad_request_reason(self):
        """Test that HTTP 400 with badRequest reason is detected."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "errors": [
                    {
                        "domain": "usageLimits",
                        "reason": "badRequest",
                        "message": "Invalid API key"
                    }
                ]
            }
        }
        
        self.assertTrue(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_with_api_key_in_message(self):
        """Test that HTTP 400 with 'api key' in message is detected."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "errors": [
                    {
                        "domain": "global",
                        "reason": "invalid",
                        "message": "API key not valid"
                    }
                ],
                "message": "API key not valid. Please pass a valid API key."
            }
        }
        
        self.assertTrue(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_with_invalid_in_message(self):
        """Test that HTTP 400 with 'invalid' in message is detected."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "errors": [],
                "message": "Invalid key provided"
            }
        }
        
        self.assertTrue(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_with_invalid_key_in_message(self):
        """Test that HTTP 400 with 'invalid' and 'key' in message is detected."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "errors": [],
                "message": "The provided key is invalid"
            }
        }
        
        self.assertTrue(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_unrelated_error(self):
        """Test that HTTP 400 without API key-related error is not detected."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {
            "error": {
                "errors": [
                    {
                        "domain": "youtube.parameter",
                        "reason": "missingRequiredParameter",
                        "message": "No filter selected."
                    }
                ],
                "message": "No filter selected."
            }
        }
        
        self.assertFalse(self.is_invalid_api_key_error(mock_response))
    
    def test_http_200_not_detected(self):
        """Test that HTTP 200 is not detected as invalid API key error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}
        
        self.assertFalse(self.is_invalid_api_key_error(mock_response))
    
    def test_http_403_not_detected(self):
        """Test that HTTP 403 (quota) is not detected as invalid API key error."""
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "error": {
                "errors": [
                    {
                        "domain": "usageLimits",
                        "reason": "quotaExceeded",
                        "message": "Quota exceeded"
                    }
                ]
            }
        }
        
        self.assertFalse(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_with_malformed_json(self):
        """Test that malformed JSON in HTTP 400 doesn't crash."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.side_effect = json.JSONDecodeError("", "", 0)
        
        # Should return False without raising an exception
        self.assertFalse(self.is_invalid_api_key_error(mock_response))
    
    def test_http_400_with_missing_error_field(self):
        """Test that HTTP 400 with missing error field doesn't crash."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"status": "error"}
        
        # Should return False without raising an exception
        self.assertFalse(self.is_invalid_api_key_error(mock_response))


if __name__ == "__main__":
    unittest.main()
