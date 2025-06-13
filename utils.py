import numpy as np
import cv2
import pickle

# Global variables for lane fitting
left_a, left_b, left_c = [], [], []
right_a, right_b, right_c = [], [], []

def nothing(x):
    """Dummy function for trackbar callbacks"""
    pass

def undistort(img, cal_dir='cal_pickle.p'):
    """Remove camera distortion from image"""
    try:
        with open(cal_dir, 'rb') as f:
            calibration_data = pickle.load(f)
        mtx = calibration_data['mtx']
        dist = calibration_data['dist']
        return cv2.undistort(img, mtx, dist, None, mtx)
    except FileNotFoundError:
        print(f"[WARNING] Calibration file {cal_dir} not found. Using original image.")
        return img

def color_filter(img):
    """Filter image for yellow and white lane colors"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    lower_yellow = np.array([18, 94, 140])
    upper_yellow = np.array([48, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 255, 255])
    
    # Create masks and combine
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    return cv2.bitwise_or(mask_white, mask_yellow)

def thresholding(img):
    """Apply various thresholding techniques"""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5))
    
    # Edge detection
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_blur, 50, 100)
    img_dilated = cv2.dilate(img_canny, kernel, iterations=1)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=1)
    
    # Color filtering
    img_color = color_filter(img)
    
    # Combine edge and color information
    combined_img = cv2.bitwise_or(img_color, img_eroded)
    
    return combined_img, img_canny, img_color

def initialize_trackbars(initial_values):
    """Initialize trackbars for perspective transform adjustment"""
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", initial_values[0], 50, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", initial_values[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", initial_values[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", initial_values[3], 100, nothing)

def val_trackbars():
    """Get current trackbar values for perspective transform"""
    width_top = cv2.getTrackbarPos("Width Top", "Trackbars")
    height_top = cv2.getTrackbarPos("Height Top", "Trackbars")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    
    src = np.float32([
        (width_top/100, height_top/100), 
        (1-(width_top/100), height_top/100),
        (width_bottom/100, height_bottom/100), 
        (1-(width_bottom/100), height_bottom/100)
    ])
    
    return src

def draw_points(img, src):
    """Draw perspective transform points on image"""
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src_points = src * img_size
    
    for i in range(4):
        cv2.circle(img, (int(src_points[i][0]), int(src_points[i][1])), 15, (0, 0, 255), cv2.FILLED)
    
    return img

def perspective_warp(img, dst_size=(640, 480), 
                    src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                    dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
    """Apply perspective transform to get bird's eye view"""
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    
    return warped

def inv_perspective_warp(img, dst_size=(640, 480),
                        src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                        dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
    """Apply inverse perspective transform"""
    img_size = np.float32([(img.shape[1], img.shape[0])])
    src = src * img_size
    dst = dst * np.float32(dst_size)
    
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    
    return warped

def get_histogram(img):
    """Get histogram of bottom half of image"""
    return np.sum(img[img.shape[0]//2:, :], axis=0)

def sliding_window(img, nwindows=15, margin=50, minpix=1, draw_windows=True):
    """Detect lane lines using sliding window technique"""
    global left_a, left_b, left_c, right_a, right_b, right_c
    
    left_fit_ = np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img)) * 255
    
    # Find lane base positions
    histogram = get_histogram(img)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set up sliding windows
    window_height = int(img.shape[0] / nwindows)
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows
    for window in range(nwindows):
        # Identify window boundaries
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw windows
        if draw_windows:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (100, 255, 255), 1)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (100, 255, 255), 1)
        
        # Identify nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append indices to lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # Recenter next window on mean position if enough pixels found
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if leftx.size and rightx.size:
        # Fit second order polynomial to each lane
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        # Store coefficients for smoothing
        left_a.append(left_fit[0])
        left_b.append(left_fit[1])
        left_c.append(left_fit[2])
        
        right_a.append(right_fit[0])
        right_b.append(right_fit[1])
        right_c.append(right_fit[2])
        
        # Use moving average for smoothing
        left_fit_[0] = np.mean(left_a[-10:])
        left_fit_[1] = np.mean(left_b[-10:])
        left_fit_[2] = np.mean(left_c[-10:])
        
        right_fit_[0] = np.mean(right_a[-10:])
        right_fit_[1] = np.mean(right_b[-10:])
        right_fit_[2] = np.mean(right_c[-10:])
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit_[0] * ploty ** 2 + left_fit_[1] * ploty + left_fit_[2]
        right_fitx = right_fit_[0] * ploty ** 2 + right_fit_[1] * ploty + right_fit_[2]
        
        # Color the lane pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
        
        return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    else:
        return img, (0, 0), (0, 0), 0

def get_curve(img, leftx, rightx):
    """Calculate radius of curvature and vehicle position"""
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    y_eval = np.max(ploty)
    
    # Conversion factors from pixels to meters
    ym_per_pix = 1 / img.shape[0]
    xm_per_pix = 0.1 / img.shape[0]
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    
    # Calculate radius of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    # Calculate vehicle position
    car_pos = img.shape[1] / 2
    l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
    center_offset = (car_pos - lane_center_position) * xm_per_pix / 10
    
    return (l_fit_x_int, r_fit_x_int , center_offset)

def draw_lanes(img, left_fit, right_fit, frame_width, frame_height, src):
    """Draw detected lanes on original image"""
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    color_img = np.zeros_like(img)
    
    # Create points for lane area
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    # Fill lane area
    cv2.fillPoly(color_img, np.int_(points), (0, 255, 0))
    
    # Transform back to original perspective
    inv_perspective = inv_perspective_warp(color_img, (frame_width, frame_height), dst=src)
    result = cv2.addWeighted(img, 0.5, inv_perspective, 0.7, 0)
    
    return result

def draw_lines(img, lane_curve):
    """Draw steering indicator lines"""
    my_width = img.shape[1]
    my_height = img.shape[0]
    
    # Draw steering indicator lines
    for x in range(-30, 30):
        w = my_width // 20
        cv2.line(img, (w * x + int(lane_curve // 100), my_height - 30),
                (w * x + int(lane_curve // 100), my_height), (0, 0, 255), 2)
    
    # Draw center lines
    cv2.line(img, (int(lane_curve // 100) + my_width // 2, my_height - 30),
            (int(lane_curve // 100) + my_width // 2, my_height), (0, 255, 0), 3)
    cv2.line(img, (my_width // 2, my_height - 50), (my_width // 2, my_height), (0, 255, 255), 2)
    
    return img

def stack_images(scale, img_array):
    """Stack images in a grid for visualization"""
    rows = len(img_array)
    cols = len(img_array[0])
    rows_available = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    
    if rows_available:
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], 
                                               (img_array[0][0].shape[1], img_array[0][0].shape[0]), 
                                               None, scale, scale)
                
                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)
        
        image_blank = np.zeros((height, width, 3), np.uint8)
        hor = [image_blank] * rows
        
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])
        
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], 
                                        (img_array[0].shape[1], img_array[0].shape[0]), 
                                        None, scale, scale)
            
            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)
        
        ver = np.hstack(img_array)
    
    return ver