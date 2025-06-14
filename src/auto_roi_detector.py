import cv2
import numpy as np
import json
import os
import sys
import pyzed.sl as sl
from scipy.signal import find_peaks

class AutoROIDetector:
    def __init__(self):
        self.belt_depth_range = None

    def preprocess_depth_frame(self, depth_frame):
        """Preprocess depth frame to handle invalid values"""
        if depth_frame is None:
            return None
        
        depth_clean = np.copy(depth_frame)
        depth_clean = np.nan_to_num(depth_clean, nan=0.0, posinf=0.0, neginf=0.0)
        
        depth_clean[depth_clean > 10000] = 0
        depth_clean[depth_clean < 100] = 0
        
        return depth_clean

    def analyze_depth_distribution(self, depth_frames):
        """Analyze depth distribution to find belt surface"""
        if not depth_frames:
            return None, None
        
        # Combine multiple depth frames
        valid_depth_frames = [d for d in depth_frames if d is not None]
        if len(valid_depth_frames) < 3:
            print(f"Warning: Only {len(valid_depth_frames)} valid depth frames available")
        
        combined_depth = np.stack(valid_depth_frames)
        median_depth = np.median(combined_depth, axis=0)
        std_depth = np.std(combined_depth, axis=0)
        
        stable_mask = std_depth < 100
        valid_mask = (median_depth > 0) & (median_depth < 10000) & np.isfinite(median_depth) & stable_mask
        valid_depths = median_depth[valid_mask]
        
        if len(valid_depths) == 0:
            print("Warning: No valid depth data found")
            return None, None
        
        print(f"Valid depth range: {np.min(valid_depths):.1f} - {np.max(valid_depths):.1f} mm")
        
        # Find peaks in the histogram as potential surfaces
        depth_min, depth_max = np.min(valid_depths), np.max(valid_depths)
        if not np.isfinite(depth_min) or not np.isfinite(depth_max) or depth_min >= depth_max:
            print("Warning: Invalid depth range detected")
            return None, None
            
        hist, bins = np.histogram(valid_depths, bins=50, range=(depth_min, depth_max), density=True)
        peaks, properties = find_peaks(hist, height=0.01, distance=5)
        
        if len(peaks) == 0:
            belt_depth = np.median(valid_depths)
            depth_tolerance = np.std(valid_depths) * 0.5
        else:
            peak_idx = peaks[np.argmax(hist[peaks])]
            belt_depth = bins[peak_idx]
            depth_tolerance = (bins[1] - bins[0]) * 3
        
        print(f"Detected belt depth: {belt_depth:.1f}mm ±{depth_tolerance:.1f}mm")
        
        return belt_depth, depth_tolerance

    def create_depth_mask(self, depth_frame, belt_depth, tolerance):
        """Create mask for pixels at belt depth"""
        if depth_frame is None or belt_depth is None:
            return None
        depth_mask = np.abs(depth_frame - belt_depth) <= tolerance
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        depth_mask = cv2.morphologyEx(depth_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        depth_mask = cv2.morphologyEx(depth_mask, cv2.MORPH_OPEN, kernel)
        
        return depth_mask

    def detect_belt_region_with_depth(self, image, depth_mask):
        """Improved belt region detection using depth information"""
        height, width = image.shape[:2]
        
        if depth_mask is not None:
            # Contour detection for more precise ROI
            contours, _ = cv2.findContours(depth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                min_area = (width * height) * 0.05
                max_area = (width * height) * 0.8
                
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if aspect_ratio > 1.2:
                        mask_rows = np.sum(depth_mask, axis=1)
                        threshold = np.max(mask_rows) * 0.3
                        valid_rows = np.where(mask_rows > threshold)[0]
                        
                        if len(valid_rows) > 0:
                            belt_top = valid_rows[0]
                            belt_bottom = valid_rows[-1]
                            belt_height = belt_bottom - belt_top + 1
                            
                            belt_region = depth_mask[belt_top:belt_bottom+1, :]
                            mask_cols = np.sum(belt_region, axis=0)
                            
                            col_threshold = np.max(mask_cols) * 0.2
                            valid_cols = np.where(mask_cols > col_threshold)[0]
                            
                            if len(valid_cols) > 0:
                                belt_left = valid_cols[0]
                                belt_right = valid_cols[-1]
                                belt_width = belt_right - belt_left + 1
                                
                                roi = {
                                    'x': belt_left,
                                    'y': belt_top,
                                    'width': belt_width,
                                    'height': belt_height
                                }
                                
                                roi_aspect_ratio = belt_width / belt_height if belt_height > 0 else 0
                                if roi_aspect_ratio > 1.2:
                                    return roi, 0.9
                        
                        roi = {'x': x, 'y': y, 'width': w, 'height': h}
                        return roi, 0.7
        
        height, width = image.shape[:2]
        roi = {
            'x': width // 6,
            'y': height // 3,
            'width': width * 2 // 3,
            'height': height // 3
        }
        return roi, 0.3

    def simple_motion_analysis(self, frame1, frame2, roi=None):
        """Simple and reliable motion analysis"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        if roi:
            x, y, w, h = roi['x'], roi['y'], roi['width'], roi['height']
            gray1 = gray1[y:y+h, x:x+w]
            gray2 = gray2[y:y+h, x:x+w]
        
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        threshold = np.percentile(magnitude[magnitude > 0], 75)
        
        significant_mask = magnitude > threshold
        if np.sum(significant_mask) < 50:
            return None, 0
        
        avg_flow_x = np.mean(flow_x[significant_mask])
        avg_flow_y = np.mean(flow_y[significant_mask])
        avg_magnitude = np.mean(magnitude[significant_mask])
        
        angle = np.arctan2(avg_flow_y, avg_flow_x) * 180 / np.pi
        if angle < 0:
            angle += 360
        
        return angle, avg_magnitude

    def detect_direction_interactive(self, svo_path, roi):
        """Interactive direction detection with user feedback"""
        print("\n=== DIRECTION DETECTION ===")
        
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        
        if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            return "unknown"
        
        image_zed = sl.Mat()
        frames = []
        
        for i in range(3):
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                frame = image_zed.get_data()[:, :, :3]
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                frames.append(frame)
        
        zed.close()
        
        if len(frames) < 2:
            return "unknown"
        
        # Analyze motion
        angles = []
        for i in range(len(frames) - 1):
            angle, strength = self.simple_motion_analysis(frames[i], frames[i+1], roi)
            if angle is not None and strength > 1.0:
                angles.append(angle)
                print(f"Frame {i}-{i+1}: {angle:.1f}° (strength: {strength:.2f})")
        
        if not angles:
            print("No significant motion detected")
            return "unknown"
        
        # Average angle
        avg_angle = np.mean(angles)
        print(f"Average angle: {avg_angle:.1f}°")
        
        # Convert to direction
        if 315 <= avg_angle or avg_angle < 45:
            detected_direction = "right"
        elif 45 <= avg_angle < 135:
            detected_direction = "down"
        elif 135 <= avg_angle < 225:
            detected_direction = "left"
        else:
            detected_direction = "up"
        
        print(f"Detected direction: {detected_direction}")
        
        return detected_direction

    def auto_detect_from_svo(self, svo_path, num_frames=10, save_path=None):
        """
        Enhanced auto-detection directly from SVO file
        Returns: (roi, direction, first_frame, depth_frame, depth_mask) for visualization
        """
        
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_path)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error opening SVO file: {err}")
            return None, None, None, None, None
        
        print(f"Reading {num_frames} frames from SVO file for auto-detection...")
        
        rgb_frames = []
        depth_frames = []
        image_zed = sl.Mat()
        depth_zed = sl.Mat()
        
        try:
            frame_count = 0
            skip_frames = 0
            
            while frame_count < num_frames:
                if zed.grab() == sl.ERROR_CODE.SUCCESS:
                    if skip_frames < 3:
                        skip_frames += 1
                        continue
                    
                    skip_frames = 0
                    
                    zed.retrieve_image(image_zed, sl.VIEW.LEFT)
                    rgb_frame = image_zed.get_data()[:, :, :3]
                    rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGBA2BGR)
                    
                    zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
                    depth_frame_raw = depth_zed.get_data()
                    depth_frame = self.preprocess_depth_frame(depth_frame_raw)
                    
                    rgb_frames.append(rgb_frame)
                    depth_frames.append(depth_frame)
                    
                    frame_count += 1
                    print(f"Loaded frame {frame_count}/{num_frames}")
                else:
                    print("Failed to grab frame")
                    break
            
            zed.close()
            
            if len(rgb_frames) < 2:
                print("Failed to load sufficient frames from SVO")
                return None, None, None, None, None
            
            belt_depth, depth_tolerance = self.analyze_depth_distribution(depth_frames)
            self.belt_depth_range = (belt_depth, depth_tolerance)
            
            depth_mask = None
            if depth_frames[0] is not None and belt_depth is not None:
                depth_mask = self.create_depth_mask(depth_frames[0], belt_depth, depth_tolerance)
            
            roi, confidence = self.detect_belt_region_with_depth(rgb_frames[0], depth_mask)
            
            print(f"Detected ROI: x={roi['x']}, y={roi['y']}, width={roi['width']}, height={roi['height']}")
            print(f"ROI Confidence: {confidence:.2f}")
            
            if roi:
                belt_direction = self.detect_direction_interactive(svo_path, roi)
                
                if save_path:
                    result = {
                        'roi': {
                            'x': int(roi['x']),
                            'y': int(roi['y']),
                            'width': int(roi['width']),
                            'height': int(roi['height'])
                        },
                        'direction': belt_direction,
                        'confidence': 0.8,  # Fixed confidence for manual verification
                        'detection_method': 'interactive'
                    }
                    
                    with open(save_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"Results saved to: {save_path}")
                
                return roi, belt_direction, rgb_frames[0], depth_frames[0], depth_mask
            else:
                return None, None, None, None, None
            
        except Exception as e:
            print(f"Error during SVO processing: {e}")
            zed.close()
            return None, None, None, None, None



    def visualize_detection(self, image, roi, direction, depth_frame=None, depth_mask=None):
        vis_image = image.copy()
        cv2.rectangle(vis_image, 
                     (roi['x'], roi['y']), 
                     (roi['x'] + roi['width'], roi['y'] + roi['height']), 
                     (0, 255, 0), 3)
        center_x = roi['x'] + roi['width'] // 2
        center_y = roi['y'] + roi['height'] // 2
        
        arrow_length = min(roi['width'], roi['height']) // 4
        if direction == "right":
            end_point = (center_x + arrow_length, center_y)
        elif direction == "left":
            end_point = (center_x - arrow_length, center_y)
        elif direction == "down":
            end_point = (center_x, center_y + arrow_length)
        elif direction == "up":
            end_point = (center_x, center_y - arrow_length)
        else:
            end_point = (center_x, center_y)
        
        cv2.arrowedLine(vis_image, (center_x, center_y), end_point, 
                       (255, 0, 0), 3, tipLength=0.3)
        
        cv2.putText(vis_image, f"Direction: {direction}", 
                   (roi['x'], roi['y'] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Overlay depth mask if available
        if depth_mask is not None:
            mask_overlay = np.zeros_like(vis_image)
            mask_overlay[depth_mask > 0] = [0, 255, 255]  # Yellow for belt surface
            vis_image = cv2.addWeighted(vis_image, 0.8, mask_overlay, 0.2, 0)
        
        return vis_image

def set_manual_direction(json_path, direction):
    """
    Manually set the belt direction in the JSON config file
    
    Args:
        json_path: Path to the auto_roi_detection.json file
        direction: 'left', 'right', 'up', or 'down'
    """
    valid_directions = ['left', 'right', 'up', 'down']
    if direction not in valid_directions:
        print(f"Error: Direction must be one of {valid_directions}")
        return False
    
    try:
        with open(json_path, 'r') as f:
            config = json.load(f)
        old_direction = config.get('direction', 'unknown')
        config['direction'] = direction
        config['detection_method'] = 'manual_override'
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Direction updated: {old_direction} → {direction}")
        print(f"Updated config saved to: {json_path}")
        return True
        
    except Exception as e:
        print(f"Error updating direction: {e}")
        return False

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, ".."))
    
    if len(sys.argv) == 3 and sys.argv[1] == "--set-direction":
        direction = sys.argv[2].lower()
        json_path = os.path.join(root_dir, "auto_roi_detection.json")
        if set_manual_direction(json_path, direction):
            print("Direction manually set successfully!")
        return
    
    svo_path = os.path.join(root_dir, "data", "belt_sample.svo")
    data_dir = os.path.join(root_dir, "data")

    if not svo_path:
        print("ERROR: No SVO file found!")
        print("Please place an .svo file in the data/ directory")
        print(f"Searched for SVO files in: {data_dir}")
        return
    
    print("Found SVO file, running auto-detection directly from SVO...")
    
    # Enhanced auto-detect ROI and direction from SVO
    detector = AutoROIDetector()
    save_path = os.path.join(root_dir, "auto_roi_detection.json")
    
    roi, direction, first_frame, depth_frame, depth_mask = detector.auto_detect_from_svo(
        svo_path, num_frames=10, save_path=save_path)
    
    if roi and direction:
        print("Auto-detection completed successfully!")
        print(f"Results saved to: {save_path}")
        print(f"ROI: {roi}")
        print(f"Direction: {direction}")
        
        # Create visualization using already loaded data (no need to re-open SVO)
        # print("Creating visualization...")
        try:
            vis_image = detector.visualize_detection(first_frame, roi, direction, depth_frame, depth_mask)
            
            vis_save_path = os.path.join(root_dir, "roi_detection_visualization.png")
            cv2.imwrite(vis_save_path, vis_image)
            print(f"Enhanced visualization saved to: {vis_save_path}")
            
            cv2.imshow('Enhanced Auto ROI Detection', vis_image)
            print("Press any key to close the visualization...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
                
        except Exception as e:
            print(f"Error creating visualization: {e}")
    else:
        print("Auto-detection failed from SVO file.")

if __name__ == "__main__":
    main() 