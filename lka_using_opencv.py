import cv2 as cv                     
import numpy as np                  
from collections import deque     
import math                        

window_size = 5                   # Number of recent steering angles to average for smoothing
angle_buffer = deque(maxlen=window_size)  # Buffer to store recent steering angles

def resize(image, scale=95):
    width = int(image.shape[1] * scale / 100) # Calculate new width based on scale percentage
    height = int(image.shape[0] * scale / 100) # Calculate new height based on scale percentage
    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA) # Resize image to new dimensions with interpolation for quality

def canny(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert color image to grayscale
    blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT) # Apply Gaussian Blur to smooth image and reduce noise
    return cv.Canny(blur, 30, 120) # Perform Canny edge detection on blurred image

def region_of_interest(image):
    height, width = image.shape[:2] # Get height and width of the image
    mask = np.zeros_like(image) # Create a black mask same size as image (all zeros)

    # Define polygon points for region of interest (ROI)
    polygon = np.array([[
        (int(0.1 * width), height),         # Bottom-left corner of polygon
        (int(0.4 * width), int(0.6 * height)),  # Upper-left corner of polygon
        (width, int(0.6 * height)),         # Upper-right corner of polygon
        (width, height)                     # Bottom-right corner of polygon
    ]], np.int32)
    # Fill the polygon area on the mask with white (255)
    if len(image.shape) == 2:
        cv.fillPoly(mask, polygon, 255)        # For grayscale images
    else:
        cv.fillPoly(mask, polygon, (255, 255, 255))  # For color images

    return cv.bitwise_and(image, mask) # Return image where only the ROI polygon is visible; rest is blacked out

def line_detection(image, original_image):
    # Lists to hold detected lines belonging to left and right lanes
    left_lines = []
    right_lines = []

    lines = cv.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=30) # Detect lines using Probabilistic Hough Line Transform
    line_detected = np.copy(image) if len(image.shape) == 3 else cv.cvtColor(image, cv.COLOR_GRAY2BGR) # Prepare a color image for drawing detected lines (if input is grayscale, convert to BGR)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]        # Extract line endpoints

            if x2 - x1 == 0: # Avoid division by zero if line is vertical
                continue

            slope = (y2 - y1) / (x2 - x1 + 1e-6) # Calculate slope of the line

            if abs(slope) < 0.5:             # Filter out near-horizontal lines to avoid noise
                continue

            # Classify line as left lane if slope is negative enough
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))

            # Classify line as right lane if slope is positive enough
            elif slope > 0:
                right_lines.append((x1, y1, x2, y2))

            # Draw all detected lines in red for visualization
            cv.line(line_detected, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Calculate average line for left lane if any left lines detected
        left_avg = np.mean(left_lines, axis=0).astype(int) if left_lines else None
        # Calculate average line for right lane if any right lines detected
        right_avg = np.mean(right_lines, axis=0).astype(int) if right_lines else None

        # Draw average left lane line in green if it exists
        if left_avg is not None:
            cv.line(line_detected, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 3)

        # Draw average right lane line in green if it exists
        if right_avg is not None:
            cv.line(line_detected, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 3)

        combined = cv.addWeighted(original_image, 0.8, line_detected, 1, 1) # Overlay the lines on the original image with transparency for better visualization
        return combined, line_detected, left_avg, right_avg

    else:
        return original_image, cv.cvtColor(image, cv.COLOR_GRAY2BGR), None, None

def compute_steering_angle(image, left_avg, right_avg):
    height, width = image.shape[:2]
    cx = width // 2
    bottom_y = height

    def bottom_point(line):
        if line is None:
            return None
        x1, y1, x2, y2 = line
        return (x1, y1) if y1 > y2 else (x2, y2)

    left_pt = bottom_point(left_avg)
    right_pt = bottom_point(right_avg)

    # Calculate midpoint between lane lines if both exist
    if left_pt and right_pt:
        mid_x = int((left_pt[0] + right_pt[0]) / 2)
        mid_y = int((left_pt[1] + right_pt[1]) / 2)
        
        # Calculate lane width for normalization
        lane_width = abs(right_pt[0] - left_pt[0])
        if lane_width < 100 or lane_width > width * 0.8:  # If lane width is unreasonable
            return 0
            
    elif right_pt:
        mid_x, mid_y = right_pt
        # If only right lane, estimate left lane position
        mid_x = mid_x - 350  # Approximate lane width
    elif left_pt:
        mid_x, mid_y = left_pt
        # If only left lane, estimate right lane position
        mid_x = mid_x + 350  # Approximate lane width
    else:
        return 0

    # Calculate horizontal deviation from center
    dx = mid_x - cx  # Reversed the subtraction to make right turns positive
    
    # Adaptive deadzone based on lane width
    deadzone = min(25, max(10, width * 0.03))  # 3% of image width, between 10-25 pixels
    if abs(dx) < deadzone:
        return 0
        
    # Calculate angle with improved scaling
    dy = bottom_y - mid_y if bottom_y - mid_y != 0 else 1
    
    # Normalize dx by lane width for more consistent angles
    if left_pt and right_pt:
        normalized_dx = dx / (lane_width * 0.5)  # Normalize by half lane width
    else:
        normalized_dx = dx / (width * 0.25)  # Use image width as reference for single lane
    
    # Calculate base angle
    angle_rad = math.atan2(normalized_dx, 1.0)  # Use normalized dx for more consistent angles
    angle_deg = math.degrees(angle_rad)
    
    # Adaptive scaling based on deviation
    scale_factor = 0.35  # Reduced base scale factor for more conservative steering
    if abs(normalized_dx) > 0.5:  # If deviation is large
        scale_factor = 0.25  # Further reduce scaling for large deviations
    
    scaled_angle = angle_deg * scale_factor
    
    # Limit maximum angle
    max_angle = 25  # Reduced maximum angle for more conservative steering
    scaled_angle = max(min(scaled_angle, max_angle), -max_angle)
    
    return scaled_angle

def smoothen_angle_display(new_angle, frame, image, prev_angle):
    alpha = 0.2 # Smoothing factor alpha for exponential moving average
    smoothed_angle = alpha * new_angle + (1 - alpha) * prev_angle  # Compute smoothed angle by blending previous angle and new measurement

    angle_buffer.append(smoothed_angle) # Add smoothed angle to buffer for further averaging
    avg_angle = sum(angle_buffer) / len(angle_buffer) # Calculate average over recent angles in buffer

    text = f'Angle: {int(avg_angle)} deg'
    cv.putText(frame, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)
    cv.putText(image, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 225), 2)
    return smoothed_angle

def draw_steering_arrow(image, angle, center_x, center_y, length=100):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate end point of arrow
    end_x = int(center_x + length * math.sin(angle_rad))
    end_y = int(center_y - length * math.cos(angle_rad))
    
    # Draw the main arrow line
    cv.arrowedLine(image, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
    
    # Add angle text using the same format as smoothen_angle_display
    text = f'Angle: {int(angle)} deg'
    cv.putText(image, text, (center_x - 50, center_y - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return image

cap= cv.VideoCapture("Raw Videos/challenge.mp4")
prev_angle = 0  # Initialize previous steering angle for smoothing

while True:
    isTrue, frame = cap.read()
    frame = resize(frame)
    canny_frame = canny(frame)
    roi = region_of_interest(canny_frame)
    line_original_image, line_roi_image, lavg, ravg = line_detection(roi, frame)
    angle = compute_steering_angle(line_roi_image, lavg, ravg)
    prev_angle = smoothen_angle_display(angle, line_original_image, line_roi_image, prev_angle)

    # Add steering arrow visualization using the smoothed angle
    height, width = line_roi_image.shape[:2]
    center_x = width // 2
    center_y = height - 50  # Position arrow near bottom center
    line_roi_image = draw_steering_arrow(line_roi_image, prev_angle, center_x, center_y)  # Use prev_angle instead of angle

    cv.imshow("Line Detection in Original Frame", line_original_image)
    cv.imshow("Line Detection in ROI Frame", line_roi_image)

    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    
cap.release()
cv.destroyAllWindows()
