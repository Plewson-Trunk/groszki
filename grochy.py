import cv2
import time
import numpy as np

def apply_filter(filter_id, frame):
    if filter_id == 1:
        b, g, r = cv2.split(frame)
        _, b_thresh = cv2.threshold(b, 72, 255, cv2.THRESH_BINARY)
        _, g_thresh = cv2.threshold(g, 72, 255, cv2.THRESH_BINARY)
        _, r_thresh = cv2.threshold(r, 72, 255, cv2.THRESH_BINARY)
        return cv2.merge((b_thresh, g_thresh, r_thresh))
    elif filter_id == 2:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif filter_id == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = cv2.bitwise_not(blurred)
        return cv2.divide(gray, inv_blur, scale=256.0)
    elif filter_id == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    elif filter_id == 5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif filter_id == 6:
        lower_bound = (50, 50, 50)
        upper_bound = (200, 200, 200)
        return cv2.inRange(frame, lower_bound, upper_bound)
    elif filter_id == 7:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    elif filter_id == 8:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        return cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
    elif filter_id == 9:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        frame_with_contours = frame.copy()
        cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 3)
        return frame_with_contours
    else:
        return frame
    
# Load the PNG overlay with an alpha channel
overlay = cv2.imread('mask_1920x1080.png', cv2.IMREAD_UNCHANGED)

# Open webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set webcam resolution
WIDTH = 1920
HEIGHT = 1080
FPS = 60
cap.set(cv2.CAP_PROP_FRAME_WIDTH,WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,HEIGHT)

# Set desired fps
# cap.set(cv2.CAP_PROP_FPS, FPS)
# Load overlay image
overlay = cv2.imread('mask_1920x1080.png', cv2.IMREAD_UNCHANGED)

# Get the actual FPS (to verify)
actual_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Actual FPS: {actual_fps}")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

filter_id = 1  # Start with the first filter
cv2.namedWindow("Filtered Webcam", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Filtered Webcam", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

start_time = time.time()
interval = 60  # Time in seconds before switching filters automatically

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply current filter
    filtered_frame = apply_filter(filter_id, frame)
    # DEBUG
    print("Filtered Frame Shape:", filtered_frame.shape)
    # Convert filtered frame to BGR (3-channel) to match overlay dimensions
    # Check if the image is grayscale before converting
    if len(filtered_frame.shape) == 2:
        filtered_frame = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)

    # Split overlay info BGR and Alpha channels
    overlay_bgr = overlay[:, :, :3] # BGR channels - first three
    overlay_mask = overlay[:, :, 3] # Alpha channel - fourth

    # Create inverse mask
    overlay_mask_inv = cv2.bitwise_not(overlay_mask)

    # Convert single channel mask to 3 channel
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR) / 255
    overlay_mask_inv = cv2.cvtColor(overlay_mask_inv, cv2.COLOR_GRAY2BGR) / 255

    # # Convert 1-channel mask to 3-channel for broadcasting
    # overlay_mask = cv2.cvtColor(overlay_mask.astype(np.float32), cv2.COLOR_GRAY2BGR)
    # overlay_mask_inv = cv2.cvtColor(overlay_mask_inv.astype(np.float32), cv2.COLOR_GRAY2BGR)

    # Blend overlay with filtered webcam frame
    filtered_frame_with_overlay = (filtered_frame * overlay_mask_inv + overlay_bgr * overlay_mask).astype(np.uint8)

    
    # Show the filtered frame in fullscreen mode
    cv2.imshow("Filtered Webcam", filtered_frame_with_overlay)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  # Manual switch
        filter_id = (filter_id % 3) + 1
        start_time = time.time()  # Reset timer
    elif key == ord('q'):  # Quit
        break

    # Automatically switch filter after interval seconds
    if time.time() - start_time > interval:
        filter_id = (filter_id % 3) + 1
        start_time = time.time()  # Reset timer

# Release resources
cap.release()
cv2.destroyAllWindows()
