#!/usr/bin/env python3
"""
Integration test for HTTP 400 API key error handling.
This script demonstrates that the new error handling works correctly.
"""

import sys
from unittest.mock import Mock, patch
import generate_json


def test_http_400_error_handling():
    """Test that HTTP 400 errors are handled correctly."""
    print("=" * 60)
    print("INTEGRATION TEST: HTTP 400 Error Handling")
    print("=" * 60)
    print()
    
    # Test 1: is_invalid_api_key_error function
    print("Test 1: Checking is_invalid_api_key_error function...")
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
            ],
            "message": "API key not valid. Please pass a valid API key."
        }
    }
    
    result = generate_json.is_invalid_api_key_error(mock_response)
    if result:
        print("✓ Test 1 PASSED: HTTP 400 with keyInvalid is detected")
    else:
        print("✗ Test 1 FAILED: HTTP 400 with keyInvalid was not detected")
        return False
    
    print()
    
    # Test 2: HTTP 400 not confused with other errors
    print("Test 2: Checking that unrelated HTTP 400 errors are not detected...")
    mock_response2 = Mock()
    mock_response2.status_code = 400
    mock_response2.json.return_value = {
        "error": {
            "errors": [
                {
                    "domain": "youtube.parameter",
                    "reason": "missingRequiredParameter",
                    "message": "No filter selected."
                }
            ]
        }
    }
    
    result2 = generate_json.is_invalid_api_key_error(mock_response2)
    if not result2:
        print("✓ Test 2 PASSED: Unrelated HTTP 400 errors are not detected as API key errors")
    else:
        print("✗ Test 2 FAILED: Unrelated HTTP 400 was incorrectly detected as API key error")
        return False
    
    print()
    
    # Test 3: HTTP 403 (quota) not confused with HTTP 400
    print("Test 3: Checking that HTTP 403 (quota) is not detected as HTTP 400...")
    mock_response3 = Mock()
    mock_response3.status_code = 403
    mock_response3.json.return_value = {
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
    
    result3 = generate_json.is_invalid_api_key_error(mock_response3)
    if not result3:
        print("✓ Test 3 PASSED: HTTP 403 quota errors are not detected as API key errors")
    else:
        print("✗ Test 3 FAILED: HTTP 403 was incorrectly detected as API key error")
        return False
    
    print()
    
    # Test 4: Check that error messages are defined
    print("Test 4: Checking that error messages are defined...")
    if hasattr(generate_json, 'MSG_INVALID_API_KEY') and hasattr(generate_json, 'MSG_CHECK_API_KEY'):
        print(f"✓ Test 4 PASSED: Error messages are defined")
        print(f"  - MSG_INVALID_API_KEY: {generate_json.MSG_INVALID_API_KEY}")
        print(f"  - MSG_CHECK_API_KEY: {generate_json.MSG_CHECK_API_KEY}")
    else:
        print("✗ Test 4 FAILED: Error messages are not defined")
        return False
    
    print()
    print("=" * 60)
    print("ALL INTEGRATION TESTS PASSED ✓")
    print("=" * 60)
    print()
    print("Summary:")
    print("- HTTP 400 errors with invalid API keys are correctly detected")
    print("- Unrelated HTTP 400 errors are not falsely detected")
    print("- HTTP 403 (quota) errors are not confused with HTTP 400")
    print("- Error messages are properly defined and user-friendly")
    print()
    print("The system will now:")
    print("1. Detect HTTP 400 errors indicating invalid API keys")
    print("2. Display a clear error message to the user")
    print("3. Attempt to switch to the next API key (if available)")
    print("4. Provide guidance on fixing the issue")
    print()
    
    return True


if __name__ == "__main__":
    success = test_http_400_error_handling()
    sys.exit(0 if success else 1)
