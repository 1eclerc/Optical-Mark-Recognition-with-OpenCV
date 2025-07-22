import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def show_image(title, image, width=None):
    """Function to display images during processing for debugging"""
    # Convert from BGR to RGB for matplotlib display
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(image, cmap='gray')
    plt.show()

def process_omr_form(image_path, answer_key, debug=False):
    """Process an OMR form with better grid positioning and booklet type detection"""
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return None
    
    original = image.copy()
    if debug:
        show_image("Original Image", original)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        show_image("Grayscale Image", gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding to get binary image
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if debug:
        show_image("Thresholded Image", thresh)
    
    # Updated regions with more precise coordinates based on the image
    regions = {
        "student_info": (409, 611, 171, 1313),
        "name_grid": (760, 2213, 171, 1313),
        "student_id_grid": (760, 1258, 1418, 1816),
        "booklet_type_grid": (1407, 1457, 1508, 1769),
        "answers_section": (2365, 3365, 225, 935),
    }
    
    # Extract regions using the coordinates from the region selector
    student_info_region = thresh[regions["student_info"][0]:regions["student_info"][1], 
                               regions["student_info"][2]:regions["student_info"][3]]
    if debug:
        show_image("Student Info Region", student_info_region)
    
    name_grid = thresh[regions["name_grid"][0]:regions["name_grid"][1], 
                      regions["name_grid"][2]:regions["name_grid"][3]]
    if debug:
        show_image("Name Grid", name_grid)
    
    student_id_grid = thresh[regions["student_id_grid"][0]:regions["student_id_grid"][1], 
                           regions["student_id_grid"][2]:regions["student_id_grid"][3]]
    if debug:
        show_image("Student ID Grid", student_id_grid)
    
    booklet_type_grid = thresh[regions["booklet_type_grid"][0]:regions["booklet_type_grid"][1], 
                             regions["booklet_type_grid"][2]:regions["booklet_type_grid"][3]]
    if debug:
        show_image("Booklet Type Grid", booklet_type_grid)
    
    answers_section = thresh[regions["answers_section"][0]:regions["answers_section"][1], 
                           regions["answers_section"][2]:regions["answers_section"][3]]
    if debug:
        show_image("Answers Section", answers_section)
    
    # Process answers using our updated method
    answers = updated_process_answer_grid(answers_section, debug)
    
    # The rest of the function remains the same...
    # Process student ID
    student_id = process_student_id_grid(student_id_grid, debug)
    
    # Process booklet type
    booklet_type = process_booklet_type(booklet_type_grid, debug)
    
    # Extract student name from name grid (improved function)
    student_name = improved_extract_student_name(name_grid, debug)
    
    # Adjust answer key based on booklet type
    final_answer_key = adjust_answer_key(answer_key, booklet_type)
    
    # Grade the exam by comparing with the answer key
    correct_count = 0
    incorrect_count = 0
    results = []
    
    for q_num in range(1, 26):  # 25 questions
        student_answer = answers.get(q_num, "BLANK")
        correct_answer = final_answer_key.get(q_num, "")
        
        if student_answer == correct_answer:
            result = "CORRECT"
            correct_count += 1
        elif student_answer in ["BLANK", "MULTIPLE"]:
            result = student_answer
            incorrect_count += 1
        else:
            result = "INCORRECT"
            incorrect_count += 1
        
        results.append({
            'question': q_num,
            'student_answer': student_answer,
            'correct_answer': correct_answer,
            'result': result
        })
    
    # Prepare the final results
    form_results = {
        'student_id': student_id,
        'student_name': student_name,
        'booklet_type': booklet_type,
        'answers': answers,
        'correct_count': correct_count,
        'incorrect_count': incorrect_count,
        'detailed_results': results
    }
    
    return form_results

def improved_extract_student_name(name_grid, debug=False):
    """
    Improved function to extract student name correctly from the name grid bubbles
    Specifically designed to handle Turkish names with special characters
    """
    # Get dimensions of the name grid
    height, width = name_grid.shape
    
    # Define the character grid layout
    # Turkish characters: A,B,C,Ç,D,E,F,G,Ğ,H,I,İ,J,K,L,M,N,O,Ö,P,R,S,Ş,T,U,Ü,V,Y,Z
    num_rows = 29  # Full Turkish alphabet with 29 letters
    num_cols = 22  # Maximum name length (based on the columns in your grid image)
    
    # Calculate dimensions for each bubble
    row_height = height // num_rows
    col_width = width // num_cols
    
    # Turkish alphabet mapping (index to character)
    turkish_alphabet = {
        0: 'A', 1: 'B', 2: 'C', 3: 'Ç', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 
        8: 'Ğ', 9: 'H', 10: 'I', 11: 'İ', 12: 'J', 13: 'K', 14: 'L', 
        15: 'M', 16: 'N', 17: 'O', 18: 'Ö', 19: 'P', 20: 'R', 21: 'S', 
        22: 'Ş', 23: 'T', 24: 'U', 25: 'Ü', 26: 'V', 27: 'Y', 28: 'Z'
    }
    
    # Draw grid lines for debugging
    if debug:
        grid_image = cv2.cvtColor(name_grid.copy(), cv2.COLOR_GRAY2BGR)
        for i in range(1, num_rows):
            cv2.line(grid_image, (0, i * row_height), (width, i * row_height), (0, 255, 0), 1)
        for i in range(1, num_cols):
            cv2.line(grid_image, (i * col_width, 0), (i * col_width, height), (0, 255, 0), 1)
        show_image("Name Grid with Lines", grid_image)
    
    # Store the detected characters by position
    detected_chars = {}
    
    # For each column position in the name
    for col in range(num_cols):
        col_values = []
        
        # Calculate this column's boundaries
        x_start = col * col_width
        x_end = (col + 1) * col_width
        
        # For each possible letter in this column
        for row in range(num_rows):
            # Calculate the bubble position
            y_start = row * row_height
            y_end = (row + 1) * row_height
            
            # Extract the bubble
            bubble = name_grid[y_start:y_end, x_start:x_end]
            
            # Count filled pixels
            total_pixels = bubble.size
            filled_pixels = cv2.countNonZero(bubble)
            filled_percentage = filled_pixels / total_pixels
            
            if debug and col < 5:  # Only show debug for first few columns
                print(f"Column {col}, Letter {turkish_alphabet.get(row, '?')}: {filled_percentage:.2f}")
            
            # Lower threshold to capture more potential marks
            if filled_percentage > 0.15:  # Reduced from 0.25
                col_values.append((filled_percentage, row))
    
        # Only keep the bubble with the highest fill percentage for this column
        if col_values:
            col_values.sort(reverse=True)  # Sort by fill percentage (highest first)
            best_match = col_values[0]
            
            # Less strict criterion for accepting a match
            if len(col_values) == 1 or (col_values[0][0] - col_values[1][0] > 0.05):  # Reduced from 0.1
                char_row = best_match[1]
                detected_chars[col] = turkish_alphabet.get(char_row, '?')
    
    # Convert to a list of (position, character) tuples and sort by position
    char_list = [(pos, char) for pos, char in detected_chars.items()]
    char_list.sort()
    
    if debug:
        print(f"Detected characters: {char_list}")
    
    # Build the name string with improved spacing logic
    name = ""
    prev_pos = -1
    
    # First pass: construct the raw name
    for pos, char in char_list:
        # Add space if there's a gap of 2 or more positions
        if prev_pos != -1 and pos - prev_pos >= 2:
            name += " "
        name += char
        prev_pos = pos
    
    # Clean up the name
    parts = name.split()
    
    # If we have multiple parts but no clear spacing
    if len(parts) == 1 and len(name) > 10:
        # Try to intelligently split into first name and last name
        # Find common Turkish surname prefixes/suffixes as split hints
        suffixes = ["OĞLU", "KIZI", "GİL"]
        prefixes = ["KAR", "DEMİR", "KARA", "YIL"]
        
        best_split = None
        # Try to find natural break points in long names
        for i in range(3, len(name) - 3):
            # Check if this could be a natural break point
            first_part = name[:i]
            second_part = name[i:]
            
            # Look for surname patterns
            for suffix in suffixes:
                if second_part.startswith(suffix) or second_part.endswith(suffix):
                    best_split = (first_part, second_part)
                    break
            
            for prefix in prefixes:
                if second_part.startswith(prefix):
                    best_split = (first_part, second_part)
                    break
        
        # If we found a good split point, use it
        if best_split:
            name = f"{best_split[0]} {best_split[1]}"
        # Otherwise try a position-based split for names without spaces
        elif len(name) >= 6:
            # Try to find a vowel followed by a consonant as a potential split point
            vowels = "AEIİOÖUÜ"
            consonants = "BCÇDFGĞHJKLMNPRŞSTVYZ"
            split_indices = []
            
            for i in range(2, len(name) - 3):
                if (name[i] in vowels and name[i+1] in consonants) or (name[i] in consonants and name[i-1] in vowels):
                    split_indices.append(i+1)
            
            # If we found potential splits, choose the one closest to middle
            if split_indices:
                middle = len(name) // 2
                best_idx = min(split_indices, key=lambda x: abs(x - middle))
                name = f"{name[:best_idx]} {name[best_idx:]}"
            else:
                # Fallback: split at 60% of the name length if it's a long name
                split_point = int(len(name) * 0.6)
                name = f"{name[:split_point]} {name[split_point:]}"
    
    # Handle cases like "HAVVA REYHAN  A KAN" -> "HAVVA REYHAN KALKAN"
    # Look for single letters that might be part of a surname
    parts = name.split()
    if len(parts) >= 3:
        # Check for isolated single letters that might be part of next/previous word
        for i in range(1, len(parts) - 1):
            if len(parts[i]) == 1:
                # Merge with the next part if it exists
                if i < len(parts) - 1:
                    parts[i+1] = parts[i] + parts[i+1]
                    parts[i] = ""
        
        # Remove empty parts
        parts = [p for p in parts if p]
    
    # Rejoin the parts
    name = " ".join(parts)
    
    # Special cases for multi-part Turkish names
    if debug:
        print(f"Raw name: {name}")
    
    return name.strip()

def updated_process_answer_grid(answers_section, debug=False):
    """
    Improved approach for accurately detecting bubble answers focusing on precise bubble detection
    """
    height, width = answers_section.shape
    
    if debug:
        print(f"Answer section dimensions: {width}x{height}")
        show_image("Full Answers Section", answers_section)
    
    # Make a copy for visualization if debugging
    visual_copy = cv2.cvtColor(answers_section.copy(), cv2.COLOR_GRAY2BGR) if debug else None
    
    # Apply additional preprocessing to enhance bubbles
    enhanced = cv2.GaussianBlur(answers_section, (5, 5), 0)
    _, enhanced = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Define a more precise grid structure based on the form layout
    # Two columns with questions 1-20 on left, 21-40 on right
    left_col_x = width * 0.177  # Center of left column
    right_col_x = width * 0.80  # Center of right column
    
    q_per_col = 20
    row_height = height / q_per_col
    
    # Option positions relative to question center
    option_offsets = [-0.14, -0.07, 0, 0.07, 0.14] # A,B,C,D,E spacing coefficients
    option_width = width * 0.14  # Approximate width between options
    
    # Store all detected answers
    answers = {}
    
    # Process first 25 questions (1-20 in left column, 21-25 in right column)
    for q_idx in range(25):
        q_num = q_idx + 1
        
        # Determine column and row
        if q_num <= 20:
            col_x = left_col_x
            row_idx = q_num - 1
        else:
            col_x = right_col_x
            row_idx = q_num - 21
        
        # Calculate row center y-position
        row_y = (row_idx + 0.5) * row_height
        
        # Store fill ratios for each option
        option_fills = []
        
        for opt_idx, offset in enumerate(option_offsets):
            # Calculate bubble center
            x = int(col_x + offset * width)
            y = int(row_y)
            
            # Define bubble area
            radius = int(min(row_height, option_width) * 0.35)
            bubble_rect = (
                max(0, x - radius),
                max(0, y - radius),
                min(width - 1, x + radius) - max(0, x - radius),
                min(height - 1, y + radius) - max(0, y - radius)
            )
            
            # Extract bubble region
            x_start, y_start, w, h = bubble_rect
            bubble = enhanced[y_start:y_start+h, x_start:x_start+w]
            
            if bubble.size == 0:
                fill_ratio = 0
            else:
                # Calculate fill ratio
                fill_ratio = 1 - (np.count_nonzero(bubble) / bubble.size)
            
            option_fills.append((opt_idx, fill_ratio))
            
            # Draw for debugging if needed
            if debug and visual_copy is not None:
                cv2.rectangle(visual_copy, 
                             (x_start, y_start), 
                             (x_start + w, y_start + h), 
                             (0, 255, 0), 2)
                cv2.putText(visual_copy, f"{chr(65+opt_idx)}:{fill_ratio:.2f}", 
                          (x_start, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Sort by fill ratio
        option_fills.sort(key=lambda x: x[1], reverse=True)
        
        # Better determination of the answer with adjusted thresholds
        if option_fills[0][1] < 0.2:  # All bubbles relatively empty
            answers[q_num] = "BLANK"
        elif len(option_fills) > 1 and option_fills[1][1] > 0.2 and (option_fills[0][1] - option_fills[1][1]) < 0.025:
            answers[q_num] = "MULTIPLE"  # Multiple bubbles filled
        else:
            # Convert to A, B, C, D, E
            selected_option = option_fills[0][0]
            answers[q_num] = chr(65 + selected_option)
        
        if debug:
            top_options = [f"{chr(65+opt[0])}:{opt[1]:.2f}" for opt in option_fills[:2]]
            print(f"Q{q_num}: Top fills {', '.join(top_options)} → Answer: {answers.get(q_num, 'UNKNOWN')}")
            
            # Visualize each question's detection for debugging
            if q_num <= 5 or (q_num >= 21 and q_num <= 25):
                show_image(f"Q{q_num} Analysis", visual_copy.copy())
    
    # If debugging, show the full analysis with all bubbles marked
    if debug and visual_copy is not None:
        show_image("Full Analysis", visual_copy)
    
    return answers

def process_student_id_grid(student_id_grid, debug=False):
    """Process the student ID grid to extract the ID number"""
    # In the form, the student ID appears to be a 8-digit number
    # Each digit has 10 possible values (0-9)
    
    # Get dimensions
    height, width = student_id_grid.shape
    
    # We have 8 digits with 10 options each
    num_digits = 8
    num_options = 10  # 0-9
    
    # Calculate dimensions
    digit_width = width // num_digits
    option_height = height // num_options
    
    student_id = ""
    
    for digit in range(num_digits):
        marked_value = None
        max_filled_percentage = 0
        
        for option in range(num_options):
            # Calculate the bubble position
            x_start = digit * digit_width
            x_end = (digit + 1) * digit_width
            y_start = option * option_height
            y_end = (option + 1) * option_height
            
            # Extract the bubble
            bubble = student_id_grid[y_start:y_end, x_start:x_end]
            
            # Count filled pixels
            total_pixels = bubble.size
            filled_pixels = cv2.countNonZero(bubble)
            filled_percentage = filled_pixels / total_pixels
            
            if filled_percentage > max_filled_percentage and filled_percentage > 0.3:
                max_filled_percentage = filled_percentage
                marked_value = option
        
        # Add the digit to the ID
        if marked_value is not None:
            student_id += str(marked_value)
        else:
            student_id += "X"  # No mark detected
    
    return student_id

def process_booklet_type(booklet_grid, debug=False):
    """Process the booklet type grid to determine which type (A, B, C, D) was used"""
    # The booklet type typically has 4 options: A, B, C, D
    
    # Get dimensions
    height, width = booklet_grid.shape
    
    # There are 4 possible booklet types
    num_options = 4  # A, B, C, D
    
    # Calculate the width of each option
    option_width = width // num_options
    
    marked_type = None
    max_filled_percentage = 0
    
    for option in range(num_options):
        # Calculate the bubble position
        x_start = option * option_width
        x_end = (option + 1) * option_width
        
        # Extract the bubble
        bubble = booklet_grid[:, x_start:x_end]
        
        # Count filled pixels
        total_pixels = bubble.size
        filled_pixels = cv2.countNonZero(bubble)
        filled_percentage = filled_pixels / total_pixels
        
        if debug:
            print(f"Booklet Type {chr(65+option)}: {filled_percentage:.2f}")
        
        if filled_percentage > max_filled_percentage and filled_percentage > 0.3:
            max_filled_percentage = filled_percentage
            marked_type = chr(65 + option)  # Convert to A, B, C, D
    
    # If no type is detected, default to a placeholder
    if marked_type is None:
        marked_type = "A"  # Default value if none detected
        
    return marked_type

def adjust_answer_key(answer_key, booklet_type):
    """Return the appropriate answer key based on the booklet type"""
    
    # Define the answer keys for each booklet type
    booklet_answers = {
        "A": {
            1: "A", 2: "E", 3: "C", 4: "B", 5: "B", 
            6: "A", 7: "A", 8: "B", 9: "A", 10: "A",
            11: "B", 12: "D", 13: "A", 14: "C", 15: "D", 
            16: "B", 17: "E", 18: "E", 19: "B", 20: "E",
            21: "D", 22: "D", 23: "B", 24: "B", 25: "A"
        },
        "B": {
            1: "B", 2: "E", 3: "A", 4: "E", 5: "A", 
            6: "D", 7: "C", 8: "A", 9: "B", 10: "C",
            11: "C", 12: "A", 13: "A", 14: "C", 15: "C", 
            16: "E", 17: "D", 18: "E", 19: "A", 20: "E",
            21: "A", 22: "B", 23: "B", 24: "D", 25: "B"
        },
        "C": {
            1: "C", 2: "E", 3: "E", 4: "D", 5: "E", 
            6: "E", 7: "E", 8: "D", 9: "C", 10: "E",
            11: "B", 12: "C", 13: "A", 14: "C", 15: "D", 
            16: "C", 17: "B", 18: "E", 19: "C", 20: "E",
            21: "E", 22: "C", 23: "B", 24: "A", 25: "C"
        },
        "D": {
            1: "D", 2: "E", 3: "D", 4: "C", 5: "C", 
            6: "B", 7: "B", 8: "C", 9: "D", 10: "B",
            11: "B", 12: "B", 13: "B", 14: "E", 15: "C", 
            16: "D", 17: "A", 18: "E", 19: "D", 20: "E",
            21: "C", 22: "A", 23: "B", 24: "C", 25: "D"
        }
    }
    
    # Return the appropriate answer key or default to type A if not found
    return booklet_answers.get(booklet_type, booklet_answers["D"])

def batch_process_forms(image_dir, answer_key, output_file, debug=False):
    from xlsxwriter.utility import xl_col_to_name
    """Process all OMR forms in a directory and save results to Excel"""
    all_results = []
    error_files = []
    
    # Track processing progress
    total_files = len([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    processed = 0
    
    # Process each image in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            print(f"Processing {filename}... ({processed+1}/{total_files})")
            
            try:
                # Process the form with minimal debugging
                result = process_omr_form(image_path, answer_key, debug=debug)
                
                if result:
                    # Add filename to the result
                    result['filename'] = filename
                    all_results.append(result)
                    print(f"✓ Successfully processed {filename}")
                else:
                    print(f"✗ Failed to process {filename} - returned None")
                    error_files.append(filename)
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")
                error_files.append(filename)
            
            processed += 1
    
    # Prepare data for Excel
    excel_data = []
    for result in all_results:
        row = {
            'Filename': result['filename'],
            'Student ID': result['student_id'],
            'Name': result['student_name'],
            'Booklet Type': result['booklet_type'],
            'Correct Answers': result['correct_count'],
            'Incorrect Answers': result['incorrect_count'],
            'Score': result['correct_count'] * 4  # Assuming 4 points per correct answer
        }
        
        # Add each answer
        for q_num in sorted(result['answers'].keys()):
            row[f'Q{q_num}'] = result['answers'][q_num]
        
        # Add if answer was correct or not
        for detail in result['detailed_results']:
            q_num = detail['question']
            row[f'Q{q_num}_Result'] = detail['result']
        
        excel_data.append(row)
    
    # Create DataFrame and save to Excel
    if excel_data:
        df = pd.DataFrame(excel_data)
        
        # Save to Excel with properly formatted columns
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            
            # Get the xlsxwriter workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # Add a format for highlighting correct/incorrect answers
            correct_format = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            incorrect_format = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
            
            # Apply conditional formatting to result columns
            for col_num, col_name in enumerate(df.columns):
                if col_name.endswith('_Result'):
                    col_letter = chr(65 + col_num)
                    col_letter = xl_col_to_name(col_num)
                    worksheet.conditional_format(f'{col_letter}2:{col_letter}{len(df)+1}', 
                                               {'type': 'text',
                                                'criteria': 'containing',
                                                'value': 'CORRECT',
                                                'format': correct_format})
                    worksheet.conditional_format(f'{col_letter}2:{col_letter}{len(df)+1}', 
                                               {'type': 'text',
                                                'criteria': 'containing',
                                                'value': 'INCORRECT',
                                                'format': incorrect_format})
        
        print(f"\nResults saved to {output_file}")
        
        # Report on errors
        if error_files:
            print(f"\nFailed to process {len(error_files)} files:")
            for err_file in error_files:
                print(f" - {err_file}")
        
        return df
    else:
        print("No results to save.")
        return None

def create_summary_report(results_df, output_file):
    """Create a summary report with statistics about the exam results"""
    # Create a new Excel file for the summary
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Student performance summary
        student_summary = pd.DataFrame({
            'Student ID': results_df['Student ID'],
            'Name': results_df['Name'],
            'Correct': results_df['Correct Answers'],
            'Incorrect': results_df['Incorrect Answers'],
            'Score': results_df['Score']
        })
        student_summary = student_summary.sort_values('Score', ascending=False)
        student_summary.to_excel(writer, index=False, sheet_name='Student Scores')
        
        # Question analysis
        question_data = []
        for q_num in range(1, 26):
            q_col = f'Q{q_num}'
            result_col = f'Q{q_num}_Result'
            
            if q_col in results_df.columns and result_col in results_df.columns:
                correct_count = len(results_df[results_df[result_col] == 'CORRECT'])
                incorrect_count = len(results_df[results_df[result_col] == 'INCORRECT'])
                blank_count = len(results_df[results_df[q_col] == 'BLANK'])
                multiple_count = len(results_df[results_df[q_col] == 'MULTIPLE'])
                
                total_attempts = len(results_df)
                difficulty = 1 - (correct_count / total_attempts) if total_attempts > 0 else 0
                
                question_data.append({
                    'Question': q_num,
                    'Correct': correct_count,
                    'Incorrect': incorrect_count,
                    'Blank': blank_count,
                    'Multiple': multiple_count,
                    'Difficulty': difficulty,
                    'Success Rate': correct_count / total_attempts if total_attempts > 0 else 0
                })
        
        question_df = pd.DataFrame(question_data)
        question_df.to_excel(writer, index=False, sheet_name='Question Analysis')
        
        # Format workbook
        workbook = writer.book
        
        # Format the Student Scores sheet
        student_sheet = writer.sheets['Student Scores']
        header_format = workbook.add_format({'bold': True, 'bg_color': '#D9E1F2', 'border': 1})
        
        # Add header formatting
        for col_num, col_name in enumerate(student_summary.columns):
            student_sheet.write(0, col_num, col_name, header_format)
        
        # Format the Question Analysis sheet
        question_sheet = writer.sheets['Question Analysis']
        
        # Add header formatting
        for col_num, col_name in enumerate(question_df.columns):
            question_sheet.write(0, col_num, col_name, header_format)
        
        # Add chart for question difficulty
        chart = workbook.add_chart({'type': 'column'})
        chart.add_series({
            'name': 'Difficulty',
            'categories': ['Question Analysis', 1, 0, len(question_df), 0],
            'values': ['Question Analysis', 1, 5, len(question_df), 5],
        })
        chart.set_title({'name': 'Question Difficulty'})
        chart.set_x_axis({'name': 'Question Number'})
        chart.set_y_axis({'name': 'Difficulty (0-1)'})
        question_sheet.insert_chart('I2', chart, {'x_scale': 1.5, 'y_scale': 1.5})
        
        # Add chart for success rate
        success_chart = workbook.add_chart({'type': 'column'})
        success_chart.add_series({
            'name': 'Success Rate',
            'categories': ['Question Analysis', 1, 0, len(question_df), 0],
            'values': ['Question Analysis', 1, 6, len(question_df), 6],
        })
        success_chart.set_title({'name': 'Question Success Rate'})
        success_chart.set_x_axis({'name': 'Question Number'})
        success_chart.set_y_axis({'name': 'Success Rate (%)'})
        question_sheet.insert_chart('I18', success_chart, {'x_scale': 1.5, 'y_scale': 1.5})
    
    print(f"Summary report saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Example answer key (for booklet type A)
    answer_key = {
        1: "C", 2: "D", 3: "E", 4: "A", 5: "E",
        6: "E", 7: "D", 8: "C", 9: "B", 10: "B",
        11: "A", 12: "B", 13: "D", 14: "C", 15: "D",
        16: "A", 17: "C", 18: "D", 19: "A", 20: "D",
        21: "C", 22: "C", 23: "D", 24: "A", 25: "C"
    }
    
    # To process a single form with debugging output
    image_path = r"C:\Users\delph\Desktop\New folder (2)\form_gorselleri_dondurulmus\sayfa_6_rotated.jpg"  # Replace with your image path
    result = process_omr_form(image_path, answer_key, debug=True)
    
    if result:
        print(f"\nForm Analysis Results:")
        print(f"Student: {result['student_name']}")
        print(f"Student ID: {result['student_id']}")
        print(f"Booklet Type: {result['booklet_type']}")
        print(f"Correct answers: {result['correct_count']}")
        print(f"Incorrect answers: {result['incorrect_count']}")
        print("\nAnswers:")
        for q_num in sorted(result['answers'].keys()):
            print(f"Q{q_num}: {result['answers'][q_num]}")
        
        # Save results to Excel
        df = pd.DataFrame([{
            'Student ID': result['student_id'],
            'Name': result['student_name'],
            'Booklet Type': result['booklet_type'],
            'Correct Answers': result['correct_count'],
            'Incorrect Answers': result['incorrect_count'],
            'Score': result['correct_count'] * 4,  # Assuming 4 points per correct answer
            **{f'Q{q_num}': result['answers'][q_num] for q_num in sorted(result['answers'].keys())}
        }])
        df.to_excel("single_result.xlsx", index=False)
        print("\nResults saved to single_result.xlsx")
    
    # To process all forms in a directory:
    forms_directory = r"C:\Users\delph\Desktop\New folder (2)\form_gorselleri_dondurulmus"  # Replace with your directory path
    df = batch_process_forms(forms_directory, answer_key, "student_results.xlsx")
    
    # Create summary report
    if df is not None:
        create_summary_report(df, "exam_summary_report.xlsx")