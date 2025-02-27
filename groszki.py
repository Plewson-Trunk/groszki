import ctypes
import ctypes.wintypes
import toml
import sys
import time
import numpy as np
import cv2
 
# Constants for setting window position
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
 
def load_config():
    try:
        with open('config.toml', 'r') as f:
            config = toml.load(f)
            return config
    except Exception as e:
        print(f"Failed to read config file: {e}")
        sys.exit(1)
 
def hide_cursor(window_name):
    cv2.setMouseCallback(window_name, lambda *args: None)
    # Hide mouse cursor (Windows specific)
    if sys.platform.startswith('win'):
        ctypes.windll.user32.ShowCursor(False)
 
def set_window_on_top(window_name):
    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
    if hwnd:
        ctypes.windll.user32.SetWindowPos(hwnd, ctypes.wintypes.HWND(HWND_TOPMOST), 0, 0, 0, 0,
                                          SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
 
def apply_filter(filter_id, frame):
    if filter_id == 1:
        b, g, r = cv2.split(frame)
        _, b_thresh = cv2.threshold(b, 72, 255, cv2.THRESH_BINARY)
        _, g_thresh = cv2.threshold(g, 72, 255, cv2.THRESH_BINARY)
        _, r_thresh = cv2.threshold(r, 72, 255, cv2.THRESH_BINARY)
        thresh_frame = cv2.merge((b_thresh, g_thresh, r_thresh))
        return thresh_frame
    elif filter_id == 2:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    elif filter_id == 3:
        lower_bound = (50, 50, 50)
        upper_bound = (200, 200, 200)
        return cv2.inRange(frame, lower_bound, upper_bound)
    elif filter_id == 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh
    elif filter_id == 5:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif filter_id == 6:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = cv2.bitwise_not(gray)
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        inv_blur = cv2.bitwise_not(blurred)
        sketch = cv2.divide(gray, inv_blur, scale=256.0)
        return sketch
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
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
        return frame
    else:
        return frame
 
def resize_and_convert(frame, width, height, channels):
    resized_frame = cv2.resize(frame, (width, height))
    if len(resized_frame.shape) == 2 and channels == 3:  # grayscale to BGR
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2BGR)
    elif len(resized_frame.shape) == 3 and channels == 1:  # BGR to grayscale
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    return resized_frame
 
def fade_transition(frame1, frame2, alpha):
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
 
def ensure_bgr(frame):
    if len(frame.shape) == 2:  # grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame
 
def main():
    print("Starting app..")
    config = load_config()
    cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
 
    if not cam.isOpened():
        print("Failed to open camera")
        sys.exit(1)
 
    width = config['camera']['width']
    height = config['camera']['height']
    fps = config['camera']['fps']
 
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cam.set(cv2.CAP_PROP_FPS, fps)
 
    window_name = 'Camera'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    hide_cursor(window_name)
 
    # Set fullscreen first
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    #time.sleep(0.5)  # Wait a bit to ensure the window is in fullscreen
    # Then set the window on top
    set_window_on_top(window_name)
 
    filters = {k: v for k, v in config['filters'].items() if v['enabled']}
    filter_keys = sorted(filters.keys())
 
    transition_duration = config['transition']['duration']
    start_time = time.time()
    current_filter_index = 0
 
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        sys.exit(1)
    current_frame = resize_and_convert(frame, width, height, 3)
    next_frame = current_frame
 
    screen_width = cv2.getWindowImageRect(window_name)[2]
    screen_height = cv2.getWindowImageRect(window_name)[3]
 
    overlay = None
    if config['mask']['enabled']:
        overlay = cv2.imread(config['mask']['path'], cv2.IMREAD_UNCHANGED)
        overlay = cv2.resize(overlay, (screen_width, screen_height))
 
    print("Starting loop..")
 
    while True:
        #print("Loop iteration..")
 
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
 
        frame = resize_and_convert(frame, width, height, 3)
        current_time = time.time()
        elapsed_time = current_time - start_time
        filter_duration = filters[filter_keys[current_filter_index]]['duration']
 
        if elapsed_time < filter_duration:
            frame = apply_filter(int(filter_keys[current_filter_index]), frame)
        else:
            if elapsed_time < filter_duration + transition_duration:
                alpha = (elapsed_time - filter_duration) / transition_duration
                next_filter_index = (current_filter_index + 1) % len(filter_keys)
                next_frame = apply_filter(int(filter_keys[next_filter_index]), frame)
                next_frame = resize_and_convert(next_frame, width, height, 3)
                frame = fade_transition(current_frame, next_frame, alpha)
            else:
                current_filter_index = (current_filter_index + 1) % len(filter_keys)
                current_frame = next_frame
                start_time = current_time
                frame = apply_filter(int(filter_keys[current_filter_index]), frame)
                frame = resize_and_convert(frame, width, height, 3)
 
        frame = ensure_bgr(frame)
 
        # Ensure the frame is of the correct size
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
 
        # Create a black background
        background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
 
        # Calculate the position to center the camera frame
        x_offset = (screen_width - width) // 2
        y_offset = (screen_height - height) // 2
 
        # Ensure the frame fits within the background dimensions
        if y_offset + height <= screen_height and x_offset + width <= screen_width:
            background[y_offset:y_offset + height, x_offset:x_offset + width] = frame
        else:
            print("Frame size does not fit within the background dimensions")
            continue
 
        if overlay is not None:
            for c in range(0, 3):
                background[:, :, c] = np.where(
                    overlay[:, :, 3] == 0,
                    background[:, :, c],
                    overlay[:, :, c]
                )
 
        cv2.imshow(window_name, background)
 
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
 
        if key in [ord(str(i)) for i in range(10)]:
            current_filter = int(chr(key))
            if current_filter in filter_keys:
                current_filter_index = filter_keys.index(str(current_filter))
                current_frame = apply_filter(current_filter, frame)
                start_time = current_time
 
    cam.release()
    cv2.destroyAllWindows()
 
if __name__ == "__main__":
    main()
