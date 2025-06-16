import os, sys
import cv2
import numpy as np
import json
from collections import deque
import pyzed.sl as sl

PATH = os.path.dirname(os.path.abspath(__file__))

class BeltSpeedMeasurement:
    def __init__(self, roi_config_path=None, camera_params=None):
        """
        Initialize belt speed measurement system
        
        Args:
            roi_config_path: Path to ROI configuration file
            camera_params: ZED camera calibration parameters for 3D conversion
        """
        if camera_params is None:
            raise ValueError("camera_params is required for 3D speed measurement")
            
        self.roi = None
        self.belt_direction = None
        self.camera_params = camera_params
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
        
        print(f"Belt Speed Measurement initialized with 3D tracking")
        print(f"Camera params: fx={self.camera_params.fx:.1f}, fy={self.camera_params.fy:.1f}")
    
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
    
    def pixel_to_3d(self, u, v, depth):
        """
        Convert 2D pixel coordinates to 3D world coordinates
        
        Args:
            u, v: Pixel coordinates
            depth: Depth value at that pixel (in mm)
            
        Returns:
            3D coordinates [X, Y, Z] in mm
        """
        if self.camera_params is None or depth <= 0:
            return None
            
        fx = self.camera_params.fx
        fy = self.camera_params.fy
        cx = self.camera_params.cx
        cy = self.camera_params.cy
        
        # Convert to 3D coordinates
        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        
        return np.array([X, Y, Z])
    
    def get_depth_at_points(self, points, depth_frame, roi_offset=(0, 0)):
        """
        Get depth values at specified 2D points
        
        Args:
            points: Array of 2D points
            depth_frame: Depth frame data
            roi_offset: Offset to convert ROI coordinates to full frame coordinates
        """
        if depth_frame is None:
            return None
            
        depths = []
        roi_x, roi_y = roi_offset
        
        for point in points:
            u, v = int(point[0] + roi_x), int(point[1] + roi_y)
            
            # Ensure coordinates are within depth frame bounds
            if 0 <= u < depth_frame.shape[1] and 0 <= v < depth_frame.shape[0]:
                depth = depth_frame[v, u]
                # Use median of small neighborhood for robustness
                if depth > 0:
                    neighborhood = depth_frame[max(0, v-1):v+2, max(0, u-1):u+2]
                    valid_depths = neighborhood[neighborhood > 0]
                    if len(valid_depths) > 0:
                        depth = np.median(valid_depths)
                depths.append(depth)
            else:
                depths.append(0)
                
        return np.array(depths)

    def calculate_speed_optical_flow(self, current_frame, previous_frame, 
                          current_depth=None, previous_depth=None, dt=1.0):
        """
        Calculate belt speed using 3D coordinate conversion
        Args:
            current_frame: Current RGB frame
            previous_frame: Previous RGB frame  
            current_depth: Current depth frame
            previous_depth: Previous depth frame (if available)
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
        
        # Get ROI offset for depth coordinate conversion
        roi_offset = (0, 0)
        if self.roi:
            roi_offset = (self.roi['x'], self.roi['y'])
        
        # Get depth values at tracked points
        depth_for_old_points = previous_depth if previous_depth is not None else current_depth
        depths_old = self.get_depth_at_points(good_old, depth_for_old_points, roi_offset)
        depths_new = self.get_depth_at_points(good_new, current_depth, roi_offset)
        
        if depths_old is None or depths_new is None:
            # Fallback to old method if no depth data
            return self.calculate_speed_optical_flow_fallback(good_old, good_new, dt)
        
        # Convert 2D points to 3D coordinates
        points_3d_old = []
        points_3d_new = []
        
        for i in range(len(good_old)):
            if depths_old[i] > 0 and depths_new[i] > 0:
                # Convert ROI coordinates back to full frame coordinates for 3D conversion
                u_old, v_old = good_old[i][0] + roi_offset[0], good_old[i][1] + roi_offset[1]
                u_new, v_new = good_new[i][0] + roi_offset[0], good_new[i][1] + roi_offset[1]
                
                point_3d_old = self.pixel_to_3d(u_old, v_old, depths_old[i])
                point_3d_new = self.pixel_to_3d(u_new, v_new, depths_new[i])
                
                if point_3d_old is not None and point_3d_new is not None:
                    points_3d_old.append(point_3d_old)
                    points_3d_new.append(point_3d_new)
        
        if len(points_3d_old) < 3:
            return self.calculate_speed_optical_flow_fallback(good_old, good_new, dt)
        
        points_3d_old = np.array(points_3d_old)
        points_3d_new = np.array(points_3d_new)
        
        # Calculate 3D motion vectors
        motion_vectors_3d = points_3d_new - points_3d_old
        
        # Calculate 3D distances
        distances_3d = np.linalg.norm(motion_vectors_3d, axis=1)
        
        # Filter out very small motions/noise
        significant_motion = distances_3d > 0.5  # 0.5mm threshold
        
        if np.sum(significant_motion) < 3:
            return None
        
        filtered_distances = distances_3d[significant_motion]
        median_distance_3d = np.median(filtered_distances)
        
        speed_mm_per_second = median_distance_3d / dt
        
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
        
        # Show 3D tracking indicator
        cv2.putText(vis_frame, "3D Tracking", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
    
    roi_config_path = os.path.join(PATH, "..", "auto_roi_detection.json")
    speed_detector = BeltSpeedMeasurement(roi_config_path, camera_params=camera_parameters)
    
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
                roi_depth, speed_detector.previous_depth, dt
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
        
        # Store current frame and depth for next iteration
        speed_detector.previous_frame = roi_frame.copy()
        speed_detector.previous_depth = roi_depth.copy() if roi_depth is not None else None
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