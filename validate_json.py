#!/usr/bin/env python3
"""
JSON Output Validation Script
Cross-checks the output.json file for correctness and completeness.
"""

import json
import sys
from datetime import datetime
from typing import Dict, List, Any


# File Constants
OUTPUT_FILE_NAME = "output.json"

# Section Names
EXPECTED_SECTIONS = ["Live", "Scheduled", "General", "Politics", "Traffic", "Crime"]

# Required Fields
REQUIRED_VIDEO_FIELDS = [
    "videoId", "title", "description", "channelTitle", "publishedAt",
    "thumbnail", "durationSeconds", "durationFormatted", "views", "likes",
    "videoType", "genre"
]

# Error Messages
ERR_FILE_NOT_FOUND = "❌ ERROR: {} not found"
ERR_INVALID_JSON = "❌ ERROR: Invalid JSON format: {}"
ERR_MISSING_FIELD = "Missing required field: '{}'"
ERR_INVALID_TIMESTAMP = "Invalid timestamp format: '{}'"
ERR_MUST_BE_LIST = "Field '{}' must be a list"
ERR_MISSING_SECTION = "Missing expected section: '{}'"
ERR_SECTION_MISSING_FIELD = "Section {}: Missing '{}' field"
ERR_SECTION_WRONG_TYPE = "Section {}: '{}' must be {}"
ERR_COUNT_MISMATCH = "Section {} ('{}'): count ({}) doesn't match items length ({})"
ERR_ITEM_MISSING_FIELD = "Section '{}', item {}: Missing field '{}'"
ERR_ITEM_WRONG_TYPE = "Section '{}', item {}: '{}' must be an integer"

# Info Messages
MSG_NOTE_NO_API_KEY = "ℹ️  Note: Pipeline ran without YouTube API key."
MSG_SET_API_KEY = "   To fetch actual data, set YOUTUBE_API_KEY environment variable."


def validate_json_structure(data: Dict[str, Any]) -> List[str]:
    """Validate the structure of the JSON output."""
    errors = []
    
    # Check required top-level fields
    if "generatedAt" not in data:
        errors.append(ERR_MISSING_FIELD.format("generatedAt"))
    else:
        # Validate ISO 8601 timestamp format
        try:
            datetime.fromisoformat(data["generatedAt"].replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            errors.append(ERR_INVALID_TIMESTAMP.format(data.get('generatedAt')))
    
    if "sections" not in data:
        errors.append(ERR_MISSING_FIELD.format("sections"))
    elif not isinstance(data["sections"], list):
        errors.append(ERR_MUST_BE_LIST.format("sections"))
    
    return errors


def validate_sections(sections: List[Dict]) -> List[str]:
    """Validate the sections array."""
    errors = []
    
    if len(sections) == 0:
        # Empty sections are valid when API key is not configured
        return errors
    
    found_sections = [s.get("section") for s in sections]
    
    for section_name in EXPECTED_SECTIONS:
        if section_name not in found_sections:
            errors.append(ERR_MISSING_SECTION.format(section_name))
    
    for i, section in enumerate(sections):
        if "section" not in section:
            errors.append(ERR_SECTION_MISSING_FIELD.format(i, "section"))
        if "count" not in section:
            errors.append(ERR_SECTION_MISSING_FIELD.format(i, "count"))
        elif not isinstance(section["count"], int):
            errors.append(ERR_SECTION_WRONG_TYPE.format(i, "count", "an integer"))
        if "items" not in section:
            errors.append(ERR_SECTION_MISSING_FIELD.format(i, "items"))
        elif not isinstance(section["items"], list):
            errors.append(ERR_SECTION_WRONG_TYPE.format(i, "items", "a list"))
        else:
            # Verify count matches items length
            if section["count"] != len(section["items"]):
                errors.append(ERR_COUNT_MISMATCH.format(
                    i,
                    section.get('section'),
                    section['count'],
                    len(section['items'])
                ))
    
    return errors


def validate_video_items(sections: List[Dict]) -> List[str]:
    """Validate video items in sections."""
    errors = []
    
    for section in sections:
        section_name = section.get("section", "Unknown")
        items = section.get("items", [])
        
        for i, item in enumerate(items):
            for field in REQUIRED_VIDEO_FIELDS:
                if field not in item:
                    errors.append(ERR_ITEM_MISSING_FIELD.format(section_name, i, field))
            
            # Validate specific field types
            if "views" in item and not isinstance(item["views"], int):
                errors.append(ERR_ITEM_WRONG_TYPE.format(section_name, i, "views"))
            if "likes" in item and not isinstance(item["likes"], int):
                errors.append(ERR_ITEM_WRONG_TYPE.format(section_name, i, "likes"))
            if "durationSeconds" in item and item["durationSeconds"] is not None and not isinstance(item["durationSeconds"], int):
                errors.append(ERR_ITEM_WRONG_TYPE.format(section_name, i, "durationSeconds"))
    
    return errors


def main():
    """Main validation function."""
    try:
        with open(OUTPUT_FILE_NAME, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(ERR_FILE_NOT_FOUND.format(OUTPUT_FILE_NAME))
        return 1
    except json.JSONDecodeError as e:
        print(ERR_INVALID_JSON.format(e))
        return 1
    
    print("=" * 60)
    print("JSON OUTPUT VALIDATION REPORT")
    print("=" * 60)
    print()
    
    # Basic info
    print(f"📄 File: {OUTPUT_FILE_NAME}")
    print(f"📅 Generated At: {data.get('generatedAt', 'N/A')}")
    
    # Check for error field
    if "error" in data:
        print(f"⚠️  Error Message: {data['error']}")
        print()
        print(MSG_NOTE_NO_API_KEY)
        print(MSG_SET_API_KEY)
        print()
    
    print(f"📊 Sections: {len(data.get('sections', []))}")
    
    # Display section summary
    sections = data.get("sections", [])
    if sections:
        print()
        print("Section Summary:")
        total_items = 0
        for section in sections:
            count = section.get("count", 0)
            total_items += count
            print(f"  • {section.get('section', 'Unknown')}: {count} videos")
        print(f"  Total Videos: {total_items}")
    
    print()
    print("-" * 60)
    print("VALIDATION RESULTS")
    print("-" * 60)
    
    # Run validations
    all_errors = []
    
    # Validate structure
    structure_errors = validate_json_structure(data)
    all_errors.extend(structure_errors)
    
    # Validate sections
    if "sections" in data and isinstance(data["sections"], list):
        section_errors = validate_sections(data["sections"])
        all_errors.extend(section_errors)
        
        # Validate video items (only if sections have items)
        if any(len(s.get("items", [])) > 0 for s in data["sections"]):
            item_errors = validate_video_items(data["sections"])
            all_errors.extend(item_errors)
    
    # Display results
    if all_errors:
        print(f"❌ VALIDATION FAILED: {len(all_errors)} error(s) found")
        print()
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print()
        return 1
    else:
        print("✅ VALIDATION PASSED")
        print()
        print("All checks completed successfully:")
        print("  ✓ Valid JSON syntax")
        print("  ✓ Required fields present")
        print("  ✓ Correct data types")
        print("  ✓ Timestamp format valid")
        
        if sections:
            print("  ✓ All expected sections present")
            print("  ✓ Section counts match item lengths")
            if any(len(s.get("items", [])) > 0 for s in sections):
                print("  ✓ Video items have required fields")
        
        print()
        print("JSON output is valid and well-formed.")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
