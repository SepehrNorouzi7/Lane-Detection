import numpy as np
import cv2
from utils import *
import time
import argparse

# Configuration constants
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
NO_ARRAY_VALUES = 10

class LaneDetector:
    def __init__(self, args):
        self.args = args
        self.setup_camera_or_video()
        self.setup_object_detection()
        self.initialize_tracking_variables()
        self.setup_output_video()
        
    def setup_camera_or_video(self):
        """Initialize camera or video capture"""
        self.camera_feed = not bool(self.args.video)
        
        if self.camera_feed:
            self.cap = cv2.VideoCapture(self.args.src)
            self.cap.set(3, FRAME_WIDTH)
            self.cap.set(4, FRAME_HEIGHT)
            self.initial_trackbar_vals = [24, 55, 12, 100]
        else:
            self.cap = cv2.VideoCapture(self.args.video)
            self.initial_trackbar_vals = [42, 63, 14, 87]
            
    def setup_object_detection(self):
        """Initialize YOLO object detection"""
        if self.args.model_weights and self.args.model_cfg:
            self.net = cv2.dnn.readNet(self.args.model_weights, self.args.model_cfg)
            with open("coco.names", "r") as f:
                self.classes = [line.strip() for line in f.readlines()]
            
            layers_names = self.net.getLayerNames()
            self.output_layers = [layers_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            self.font = cv2.FONT_HERSHEY_PLAIN
            self.object_detection_enabled = True
        else:
            self.object_detection_enabled = False
            
    def initialize_tracking_variables(self):
        """Initialize tracking variables"""
        self.frame_id = 0
        self.array_counter = 0
        self.array_curve = np.zeros([NO_ARRAY_VALUES])
        initialize_trackbars(self.initial_trackbar_vals)
        
    def setup_output_video(self):
        """Setup video output writer"""
        self.video_writer = cv2.VideoWriter(
            'output_lane_detection.avi', 
            cv2.VideoWriter_fourcc(*'XVID'), 
            self.cap.get(cv2.CAP_PROP_FPS), 
            (2 * FRAME_WIDTH, FRAME_HEIGHT)
        )
        
    def detect_objects(self, frame):
        """Perform object detection on frame"""
        if not self.object_detection_enabled:
            return frame
            
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        class_ids, confidences, boxes = [], [], []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = f"{self.classes[class_ids[i]]}: {confidences[i]*100:.2f}%"
                color = self.colors[i]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, y+10), self.font, 2, color, 2)
        
        return frame
        
    def process_lane_detection(self, img):
            """Process lane detection pipeline"""
            img_undistorted = undistort(img)
            img_thresholded, img_canny, img_color = thresholding(img_undistorted)
            
            src = val_trackbars()
            img_warped = perspective_warp(img_thresholded, dst_size=(FRAME_WIDTH, FRAME_HEIGHT), src=src)
            img_warp_points = draw_points(img.copy(), src)
            img_sliding, curves, lanes, ploty = sliding_window(img_warped, draw_windows=True)
            
            try:
                curverad = get_curve(img, curves[0], curves[1])
                lane_curve = np.mean([curverad[0], curverad[1]])
                img_final = draw_lanes(img, curves[0], curves[1], FRAME_WIDTH, FRAME_HEIGHT, src=src)
                
                # Calculate average curve
                current_curve = lane_curve // 50
                if int(np.sum(self.array_curve)) == 0:
                    average_curve = current_curve
                else:
                    average_curve = np.sum(self.array_curve) // self.array_curve.shape[0]
                    
                if abs(average_curve - current_curve) > 200:
                    self.array_curve[self.array_counter] = average_curve
                else:
                    self.array_curve[self.array_counter] = current_curve
                    
                self.array_counter += 1
                if self.array_counter >= NO_ARRAY_VALUES:
                    self.array_counter = 0
                    
                cv2.putText(img_final, str(int(average_curve)), 
                        (FRAME_WIDTH//2-70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)
                
                img_final = draw_lines(img_final, lane_curve)
                
            except:
                lane_curve = 0
                img_final = img.copy()
                
            return img_final, img_undistorted, img_color, img_canny, img_warped, img_sliding
        
    def run(self):
        """Main processing loop"""
        starting_time = time.time()
        
        print('----- Lane Detection Started -----')
        print(f'[INFO] Config file: {self.args.model_cfg}')
        print(f'[INFO] Model weights: {self.args.model_weights}')
        print(f'[INFO] Video path: {self.args.video}')
        
        while True:
            success, img = self.cap.read()
            if not success:
                print('[INFO] Processing completed!')
                break
                
            if not self.camera_feed:
                img = cv2.resize(img, (FRAME_WIDTH, FRAME_HEIGHT))
                
            # Lane detection processing
            img_final, img_undistorted, img_color, img_canny, img_warped, img_sliding = self.process_lane_detection(img)
            
            # Object detection
            frame_with_objects = self.detect_objects(img.copy())
            
            # Calculate FPS
            self.frame_id += 1
            elapsed_time = time.time() - starting_time
            fps = self.frame_id / elapsed_time
            cv2.putText(frame_with_objects, f"FPS: {fps:.1f}", (10, 30), self.font, 2, (0, 0, 0), 1)
            
            # Stack images for visualization
            img_stacked = stack_images(0.7, ([img_undistorted, frame_with_objects],
                                          [img_color, img_canny],
                                          [img_warped, img_sliding]))
            
            # Display results
            cv2.imshow("Object Detection", frame_with_objects)
            cv2.imshow("Pipeline", img_stacked)
            cv2.imshow("Lane Detection Result", img_final)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print('[INFO] All resources cleaned up!')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Lane Detection with Object Detection')
    parser.add_argument('--model_cfg', type=str, default='', help='Path to YOLO config file')
    parser.add_argument('--model_weights', type=str, default='', help='Path to YOLO weights file')
    parser.add_argument('--video', type=str, default='', help='Path to video file')
    parser.add_argument('--src', type=int, default=0, help='Camera source index')
    parser.add_argument('--output_dir', type=str, default='', help='Output directory path')
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_arguments()
    detector = LaneDetector(args)
    detector.run()

if __name__ == "__main__":
    main()