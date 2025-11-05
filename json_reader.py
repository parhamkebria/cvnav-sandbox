#!/usr/bin/env python3
"""
Controls:
- Press 'S' or 's' + Enter: Save current section to a new text file
- Press Enter only: Skip saving and continue to next section
"""

import json
import os
import sys
from datetime import datetime


def print_separator():
    """Print a visual separator line."""
    print("=" * 80)


def format_json_section(key, value, indent=0):
    """Format a JSON section for display."""
    indent_str = "  " * indent
    
    if isinstance(value, dict):
        result = f"{indent_str}{key}:\n"
        for sub_key, sub_value in value.items():
            result += format_json_section(sub_key, sub_value, indent + 1)
        return result
    elif isinstance(value, list):
        result = f"{indent_str}{key}: [\n"
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                result += f"{indent_str}  Item {i + 1}:\n"
                result += format_json_section("", item, indent + 2)
            else:
                result += f"{indent_str}  {item}\n"
        result += f"{indent_str}]\n"
        return result
    else:
        return f"{indent_str}{key}: {value}\n"


def save_section_to_file(section_name, section_content, base_filename):
    """Save a section to a text file."""
    # Create a safe filename
    safe_section_name = "".join(c for c in section_name if c.isalnum() or c in (' ', '_', '-')).strip()
    safe_section_name = safe_section_name.replace(' ', '_')
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"{base_filename}_{safe_section_name}_{timestamp}.txt"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Section: {section_name}\n")
            f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n\n")
            f.write(section_content)
        
        print(f"✅ Section saved to: {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return False


def extract_frame_id_from_image_name(image_name):
    """Extract frame ID from image name like 'frame_20190829091111_x_0001973.jpg'."""
    try:
        # Remove file extension
        name_without_ext = os.path.splitext(image_name)[0]
        
        # Split by underscores and find the frame part
        parts = name_without_ext.split('_')
        if len(parts) >= 4 and parts[0] == 'frame':
            # Reconstruct frame ID: frame_timestamp_x_number
            frame_id = f"frame_{parts[1]}_{parts[2]}_{parts[3]}"
            return frame_id
        else:
            # Fallback: use the whole name without extension
            return name_without_ext
    except Exception:
        # If anything goes wrong, use the original name without extension
        return os.path.splitext(image_name)[0]


def save_annotation_item_to_file(annotation_item, index):
    """Save individual annotation item to a file named after the image."""
    try:
        # Get image_name from the annotation item
        image_name = annotation_item.get('image_name', f'unknown_image_{index}')
        
        # Extract frame ID for filename
        frame_id = extract_frame_id_from_image_name(image_name)
        
        # Create filename
        filename = f"{frame_id}.txt"
        
        # Format the content
        content = f"Image Annotation Data\n"
        content += f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += "=" * 50 + "\n\n"
        
        # Add all fields from the annotation item
        for key, value in annotation_item.items():
            if isinstance(value, (dict, list)):
                content += f"{key}:\n"
                content += format_json_section("", value, 1)
            else:
                content += f"{key}: {value}\n"
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  ✅ Saved annotation for {image_name} to: {filename}")
        return True
        
    except Exception as e:
        print(f"  ❌ Error saving annotation item {index}: {e}")
        return False


def handle_annotations_section(annotations_data):
    """Special handler for annotations section - saves each image annotation individually."""
    print(f"\n🎯 Special handling for 'annotations' section detected!")
    print(f"📊 Found {len(annotations_data)} image annotations")
    
    choice = input("\nPress 'S' to save ALL image annotations as individual files, or Enter to skip: ").strip().lower()
    
    if choice == 's':
        saved_count = 0
        print(f"\n💾 Saving individual annotation files...")
        
        for i, annotation in enumerate(annotations_data, 1):
            if save_annotation_item_to_file(annotation, i):
                saved_count += 1
        
        print(f"\n🎉 Saved {saved_count} out of {len(annotations_data)} annotation files")
        return saved_count
    else:
        print("⏭️ Skipped saving annotation files")
        return 0


def get_user_choice():
    """Get user choice for saving the section."""
    while True:
        choice = input("\nPress 'S' to save this section, or just Enter to skip: ").strip().lower()
        if choice == 's':
            return True
        elif choice == '':
            return False
        else:
            print("Invalid input. Please press 'S' to save or just Enter to skip.")


def main():
    """Main function to run the JSON section viewer."""
    print("JSON Section Viewer and Saver")
    print_separator()
    
    # Get JSON file path from user
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = input("Enter the path to your JSON file: ").strip().strip('"')
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"❌ Error: File '{json_file_path}' not found.")
        return
    
    # Read the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON file. {e}")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Get base filename for saving
    base_filename = os.path.splitext(os.path.basename(json_file_path))[0]
    
    print(f"✅ Successfully loaded JSON file: {json_file_path}")
    print(f"📊 Found {len(data)} major sections")
    print_separator()
    
    # Process each major section
    section_count = 0
    saved_count = 0
    individual_annotations_saved = 0
    
    if isinstance(data, dict):
        for key, value in data.items():
            section_count += 1
            print(f"\n📋 Section {section_count}: {key}")
            print("-" * 40)
            
            # Special handling for annotations section
            if key.lower() == "annotations" and isinstance(value, list):
                # Show a preview of the annotations
                print(f"📸 This section contains {len(value)} image annotations")
                if len(value) > 0:
                    print(f"📋 Sample annotation preview:")
                    sample = value[0]
                    if isinstance(sample, dict) and 'image_name' in sample:
                        print(f"   Image: {sample.get('image_name', 'N/A')}")
                        print(f"   Fields: {list(sample.keys())}")
                    print(f"   ... and {len(value) - 1} more annotations")
                
                print_separator()
                
                # Handle annotations specially
                annotations_saved = handle_annotations_section(value)
                individual_annotations_saved += annotations_saved
                
            else:
                # Regular section handling
                # Format and display the section
                section_content = format_json_section(key, value)
                print(section_content)
                
                print_separator()
                
                # Ask user if they want to save this section
                if get_user_choice():
                    if save_section_to_file(key, section_content, base_filename):
                        saved_count += 1
            
            print_separator()
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            section_count += 1
            section_name = f"Item_{i + 1}"
            print(f"\n📋 Section {section_count}: {section_name}")
            print("-" * 40)
            
            # Format and display the section
            section_content = format_json_section(section_name, item)
            print(section_content)
            
            print_separator()
            
            # Ask user if they want to save this section
            if get_user_choice():
                if save_section_to_file(section_name, section_content, base_filename):
                    saved_count += 1
            
            print_separator()
    
    else:
        print("❌ Error: JSON file does not contain a dictionary or list at the root level.")
        return
    
    # Summary
    print(f"\n🎉 Processing complete!")
    print(f"📊 Total sections processed: {section_count}")
    print(f"💾 Regular sections saved: {saved_count}")
    if individual_annotations_saved > 0:
        print(f"📸 Individual annotation files saved: {individual_annotations_saved}")
    print(f"⏭️  Sections skipped: {section_count - saved_count}")
    
    total_files_saved = saved_count + individual_annotations_saved
    if total_files_saved > 0:
        print(f"\n📁 Total files created: {total_files_saved}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
