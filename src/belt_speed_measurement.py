import os, sys
import cv2
import numpy as np
import json
from collections import deque
import pyzed.sl as sl

PATH = os.path.dirname(os.path.abspath(__file__))

class BeltSpeedMeasurement:
    def __init__(self, roi_config_path=None, calibration_factor=1.0):
        """
        Initialize belt speed measurement system
        
        Args:
            roi_config_path: Path to ROI configuration file (JSON)
            calibration_factor: Pixels to mm conversion factor
        """
        self.roi = None
        self.belt_direction = None
        self.calibration_factor = calibration_factor
        self.previous_frame = None
        self.speed_history = deque(maxlen=10)
        self.belt_depth_range = None
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        if roi_config_path and os.path.exists(roi_config_path):
            self.load_roi_config(roi_config_path)
        
        print(f"Belt Speed Measurement initialized")
        print(f"Calibration factor: {self.calibration_factor} mm/pixel")
    
    def load_roi_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if 'roi' in config:
                self.roi = config['roi']
                print(f"Loaded ROI: x={self.roi['x']}, y={self.roi['y']}, "
                      f"width={self.roi['width']}, height={self.roi['height']}")
            
            if 'belt_depth' in config and 'depth_tolerance' in config:
                self.belt_depth_range = (config['belt_depth'], config['depth_tolerance'])
                print(f"Loaded belt depth: {config['belt_depth']:.1f}mm ±{config['depth_tolerance']:.1f}mm")
            
            if 'direction' in config:
                self.belt_direction = config['direction']
                print(f"Loaded belt direction: {self.belt_direction}")
                
        except Exception as e:
            print(f"Error loading ROI config: {e}")
    
    def create_depth_mask(self, depth_frame):
        if depth_frame is None or self.belt_depth_range is None:
            return None
        
        belt_depth, tolerance = self.belt_depth_range
        depth_mask = np.abs(depth_frame - belt_depth) <= tolerance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_mask = cv2.morphologyEx(depth_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        
        return depth_mask
    
    def extract_roi_region(self, frame, depth_frame=None):
        if self.roi is None:
            return frame, depth_frame
        
        x, y, w, h = self.roi['x'], self.roi['y'], self.roi['width'], self.roi['height']
        
        roi_frame = frame[y:y+h, x:x+w]
        roi_depth = depth_frame[y:y+h, x:x+w] if depth_frame is not None else None
        
        return roi_frame, roi_depth
    
    def detect_features_with_depth(self, gray_frame, depth_mask=None):
        corners = cv2.goodFeaturesToTrack(
            gray_frame, 
            mask=depth_mask,
            **self.feature_params
        )
        
        return corners
    
    def calculate_speed_optical_flow(self, current_frame, previous_frame, 
                                   current_depth=None, dt=1.0):
        """
        Calculate belt speed using optical flow
        Args:
            current_frame: Current RGB frame
            previous_frame: Previous RGB frame  
            current_depth: Current depth frame
            dt: Time difference between frames in seconds
        """
        if previous_frame is None:
            return None
        
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_previous = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        depth_mask = None
        if current_depth is not None:
            depth_mask = self.create_depth_mask(current_depth)
        
        corners = self.detect_features_with_depth(gray_previous, depth_mask)
        
        if corners is None or len(corners) < 5:
            return None
        
        next_corners, status, error = cv2.calcOpticalFlowPyrLK(
            gray_previous, gray_current, corners, None, **self.lk_params
        )
        
        good_old = corners[status == 1]
        good_new = next_corners[status == 1]
        
        if len(good_old) < 3:
            return None
        
        motion_vectors = good_new - good_old
        magnitudes = np.linalg.norm(motion_vectors, axis=1)
        significant_motion = magnitudes > 0.5
        
        if np.sum(significant_motion) < 3:
            return None
        
        filtered_vectors = motion_vectors[significant_motion]
        median_motion = np.median(filtered_vectors, axis=0)
        speed_pixels_per_frame = np.linalg.norm(median_motion)
        speed_mm_per_second = (speed_pixels_per_frame * self.calibration_factor) / dt
        
        return speed_mm_per_second
    
    def smooth_speed(self, raw_speed):
        """Apply smoothing to speed measurements"""
        if raw_speed is not None:
            self.speed_history.append(raw_speed)
        else:
            self.speed_history.append(0.0)
        
        if len(self.speed_history) == 0:
            return 0.0
        
        speeds = list(self.speed_history)
        smoothed_speed = np.median(speeds)
        
        return smoothed_speed
    
    def set_fallback_roi(self, frame_shape):
        """Set a fallback ROI if no configuration is loaded"""
        height, width = frame_shape[:2]
        
        roi_width = width // 2
        roi_height = height // 3
        roi_x = (width - roi_width) // 2
        roi_y = (height - roi_height) // 2
        
        self.roi = {
            'x': roi_x,
            'y': roi_y,
            'width': roi_width,
            'height': roi_height
        }
        
        print(f"Using fallback ROI: x={roi_x}, y={roi_y}, width={roi_width}, height={roi_height}")
        print("Warning: Using fallback ROI. For better results, run auto_roi_detector.py first.")
        
        return True
    
    def visualize_tracking(self, frame, speed=None):
        """Visualize tracking results on frame"""
        vis_frame = frame.copy()
        
        if self.roi:
            cv2.rectangle(vis_frame, 
                         (self.roi['x'], self.roi['y']), 
                         (self.roi['x'] + self.roi['width'], self.roi['y'] + self.roi['height']), 
                         (0, 255, 0), 2)
        
        if speed is not None:
            cv2.putText(vis_frame, f"Speed: {speed:.1f} mm/s", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.belt_direction:
            cv2.putText(vis_frame, f"Direction: {self.belt_direction}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return vis_frame

def main():
    cam = sl.Camera()
    input_type = sl.InputType()
    input_type.set_from_svo_file(os.path.join(PATH, "..", "data", "belt_sample.svo"))
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    status = cam.open(init)
    
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)
    
    camera_parameters = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    print("Camera Parameters: ", camera_parameters.fx, camera_parameters.fy, camera_parameters.cx, camera_parameters.cy)
    
    # Estimate calibration factor based on camera parameters and typical setup
    fx = camera_parameters.fx
    estimated_distance = 1000
    calibration_factor = estimated_distance / fx
    
    print(f"Estimated calibration factor: {calibration_factor:.3f} mm/pixel")
    print("Note: This is an approximation. For accurate results, calibrate with known reference object.")
    
    roi_config_path = os.path.join(PATH, "..", "auto_roi_detection.json")
    speed_detector = BeltSpeedMeasurement(roi_config_path, calibration_factor=calibration_factor)
    
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    depth = sl.Mat()
    timestamp = sl.Timestamp()
    
    results_file = os.path.join(PATH, "..", "results.csv")
    fhand = open(results_file, 'w')
    fhand.write("timestamp_ms,speed_mm_per_s,smoothed_speed_mm_per_s\n")
    
    vis_output_dir = os.path.join(PATH, "..", "visualization_frames")
    if not os.path.exists(vis_output_dir):
        os.makedirs(vis_output_dir)
        print(f"Created visualization output directory: {vis_output_dir}")
    
    print("Starting belt speed measurement...")
    print("Press 'q' to quit, 's' to save current frame, 'r' to reset, 'v' to toggle auto-save visualization")
    
    roi_ready = speed_detector.roi is not None
    if not roi_ready:
        print("No ROI configuration loaded. Will use fallback ROI after first frame.")
    
    save_visualization = True 
    save_every_n_frames = 10
    last_saved_frame = 0
    
    key = ''
    previous_timestamp = None
    frame_count = 0
    
    while key != ord('q'):
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("End of SVO reached")
            break
        elif err != sl.ERROR_CODE.SUCCESS:
            print("Error grabbing frame: ", err)
            break
        
        cam.retrieve_image(image, sl.VIEW.LEFT)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
        timestamp = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        
        rgb_frame = image.get_data()
        depth_frame = depth.get_data()
        
        frame_count += 1
        
        if not roi_ready and frame_count == 1:
            speed_detector.set_fallback_roi(rgb_frame.shape)
            roi_ready = True
            print("Fallback ROI set. Starting speed measurement...")
        
        # Calculate time difference
        dt = 1.0 
        if previous_timestamp is not None:
            dt = max(0.001, (timestamp.get_milliseconds() - previous_timestamp) / 1000.0) 
        
        roi_frame, roi_depth = speed_detector.extract_roi_region(rgb_frame, depth_frame)
        
        # Calculate speed
        raw_speed = None
        if roi_ready and speed_detector.previous_frame is not None:
            raw_speed = speed_detector.calculate_speed_optical_flow(
                roi_frame, speed_detector.previous_frame, 
                roi_depth, dt
            )
        
        # Apply smoothing
        smoothed_speed = speed_detector.smooth_speed(raw_speed)
        
        # Log results
        timestamp_ms = timestamp.get_milliseconds()
        speed_value = raw_speed if raw_speed is not None else 0.0
        
        print(f"Frame {frame_count}: Timestamp: {timestamp_ms}, "
              f"Raw Speed: {speed_value:.2f} mm/s, "
              f"Smoothed Speed: {smoothed_speed:.2f} mm/s")
        
        fhand.write(f"{timestamp_ms},{speed_value:.2f},{smoothed_speed:.2f}\n")
        fhand.flush()
        
        # Store current frame for next iteration
        speed_detector.previous_frame = roi_frame.copy()
        previous_timestamp = timestamp_ms
        
        vis_frame = speed_detector.visualize_tracking(rgb_frame, speed=smoothed_speed)
        
        if save_visualization and (frame_count - last_saved_frame) >= save_every_n_frames:
            vis_save_path = os.path.join(vis_output_dir, f"visualization_{frame_count:05d}.png")
            cv2.imwrite(vis_save_path, vis_frame)
            last_saved_frame = frame_count
            print(f"Auto-saved visualization: {vis_save_path}")
        
        cv2.imshow("Belt Speed Measurement", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            save_path = os.path.join(PATH, "..", f"frame_{frame_count:05d}.png")
            cv2.imwrite(save_path, rgb_frame)
            vis_save_path = os.path.join(vis_output_dir, f"manual_save_{frame_count:05d}.png")
            cv2.imwrite(vis_save_path, vis_frame)
            print(f"Frame saved to: {save_path}")
            print(f"Visualization saved to: {vis_save_path}")
        elif key == ord('r'):
            speed_detector.speed_history.clear()
            print("Speed history reset")
        elif key == ord('v'):
            save_visualization = not save_visualization
            status = "enabled" if save_visualization else "disabled"
            print(f"Auto-save visualization {status}")
            if save_visualization:
                print(f"Will save every {save_every_n_frames} frames to: {vis_output_dir}")
    
    fhand.close()
    cam.close()
    cv2.destroyAllWindows()
    
    print(f"Processing completed. Results saved to: {results_file}")
    print(f"Processed {frame_count} frames")
    
    # Print visualization summary
    vis_files = [f for f in os.listdir(vis_output_dir) if f.endswith('.png')]
    if vis_files:
        print(f"Saved {len(vis_files)} visualization frames to: {vis_output_dir}")
    
    if len(speed_detector.speed_history) > 0:
        speeds = list(speed_detector.speed_history)
        print(f"Final speed statistics:")
        print(f"  Average: {np.mean(speeds):.2f} mm/s")
        print(f"  Median: {np.median(speeds):.2f} mm/s")
        print(f"  Std Dev: {np.std(speeds):.2f} mm/s")
        print(f"  Min: {np.min(speeds):.2f} mm/s")
        print(f"  Max: {np.max(speeds):.2f} mm/s")

if __name__ == "__main__":
    main() 