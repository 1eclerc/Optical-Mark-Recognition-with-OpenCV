import cv2
import numpy as np
import os

# Global variables for mouse callback
start_point = None
end_point = None
drawing = False
current_image = None
region_name = ""
regions = {}

def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing, current_image, temp_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        # Create a copy of the current image to draw on
        temp_image = current_image.copy()
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Create a temporary image to show the rectangle being drawn
        temp_image = current_image.copy()
        cv2.rectangle(temp_image, start_point, (x, y), (0, 255, 0), 2)
        cv2.imshow("Select Region", temp_image)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        # Draw the final rectangle
        cv2.rectangle(current_image, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(current_image, region_name, (start_point[0], start_point[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Select Region", current_image)
        
        # Convert to y_start, y_end, x_start, x_end format
        x_start = min(start_point[0], end_point[0])
        x_end = max(start_point[0], end_point[0])
        y_start = min(start_point[1], end_point[1])
        y_end = max(start_point[1], end_point[1])
        
        # Save the coordinates
        regions[region_name] = (y_start, y_end, x_start, x_end)
        print(f"Selected coordinates for {region_name}: ({y_start}, {y_end}, {x_start}, {x_end})")

def select_regions(image_path):
    global current_image, region_name, regions
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    # Resize the image if it's too large for the screen
    height, width = image.shape[:2]
    max_height = 900
    max_width = 1600
    
    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
        print(f"Image resized by factor {scale:.2f} for display purposes")
        print(f"Original dimensions: {width}x{height}, New dimensions: {int(width * scale)}x{int(height * scale)}")
        print("Note: Coordinates will be scaled back to original size.")
        scaling_factor = 1/scale
    else:
        scaling_factor = 1
    
    # Create window and set mouse callback
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", mouse_callback)
    
    # Regions to select
    region_names = [
        "student_info",     # Name and ID fields at top
        "name_grid",        # "ADI, SOYADI" section  
        "student_id_grid",  # "ÖĞRENCİ NO" section
        "booklet_type_grid", # "KITAPÇIK TÜRÜ" section
        "answers_section"   # "CEVAPLAR" section
    ]
    
    for name in region_names:
        # Reset for new region
        region_name = name
        current_image = image.copy()
        
        # Display instructions
        instruction_text = f"Select region for: {name}"
        print("\n" + "="*50)
        print(instruction_text)
        print("Click and drag to select the region")
        print("Press 'r' to reset selection")
        print("Press 's' to save and continue to next region")
        print("Press 'q' to quit")
        
        cv2.putText(current_image, instruction_text, (10, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Select Region", current_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):  # Save and move to next region
                if region_name in regions:
                    print(f"Saved region: {name}")
                    break
                else:
                    print("Please select a region first!")
            
            elif key == ord('r'):  # Reset selection for current region
                print("Reset selection")
                if region_name in regions:
                    regions.pop(region_name)
                current_image = image.copy()
                cv2.putText(current_image, instruction_text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Select Region", current_image)
            
            elif key == ord('q'):  # Quit
                print("Selection canceled")
                cv2.destroyAllWindows()
                return
    
    cv2.destroyAllWindows()
    
    # Scale coordinates back to original size if image was resized
    if scaling_factor != 1:
        for name in regions:
            y_start, y_end, x_start, x_end = regions[name]
            regions[name] = (
                int(y_start * scaling_factor),
                int(y_end * scaling_factor),
                int(x_start * scaling_factor),
                int(x_end * scaling_factor)
            )
            print(f"Scaled coordinates for {name}: {regions[name]}")
    
    # Print final regions
    print("\n" + "="*50)
    print("Final selected regions (ready to copy into your code):")
    print("regions = {")
    for name, coords in regions.items():
        print(f'    "{name}": {coords},')
    print("}")
    
    # Display the extracted regions 
    # Load original image again
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    print("\nShowing extracted thresholded regions. Press any key to continue to next region.")
    for name, (y_start, y_end, x_start, x_end) in regions.items():
        # Extract region from thresholded image
        region_image = thresh[y_start:y_end, x_start:x_end]
        cv2.imshow(f"Extracted {name}", region_image)
        cv2.waitKey(0)
        cv2.destroyWindow(f"Extracted {name}")
    
    print("\nDone! You can now update your OMR script with these coordinates.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python interactive_region_selector.py [image_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    select_regions(image_path)