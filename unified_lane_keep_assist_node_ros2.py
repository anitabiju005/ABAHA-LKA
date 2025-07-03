import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Float32
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import math
from collections import deque

class UnifiedLaneKeepAssist(Node):
    def __init__(self):
        super().__init__('lane_keep_assist')
        self.bridge = CvBridge()

        video_path = '/home/anita/ros2_ws/src/lane_keeping_assist/test_videos/Sample_3.mp4'
        self.cap = cv.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open video")
            rclpy.shutdown()
            return

        self.timer = self.create_timer(0.03, self.timer_callback)

        self.image_pub = self.create_publisher(Image, '/lka/lines_image', 10)
        self.lane_pub = self.create_publisher(Float32MultiArray, '/lka/lane_lines', 10)
        self.angle_pub = self.create_publisher(Float32, '/lka/steering_angle', 10)

        # New publishers for left and right wheel angles
        self.left_wheel_angle_pub = self.create_publisher(Float32, '/lka/left_wheel_angle', 10)
        self.right_wheel_angle_pub = self.create_publisher(Float32, '/lka/right_wheel_angle', 10)

        self.kalman_left = self.init_kalman_filter()
        self.kalman_right = self.init_kalman_filter()

        self.angle_buffer = deque(maxlen=5)
        self.prev_angle = 0.0

        self.get_logger().info("Unified Lane Keep Assist Node started")

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info("Video ended or cannot read frame. Shutting down node.")
            self.cap.release()
            self.destroy_timer(self.timer)
            cv.destroyAllWindows()
            rclpy.shutdown()
            return

        frame = self.resize(frame, 90)
        edges = self.canny(frame)
        roi = self.region_of_interest(edges)
        combined, left_line, right_line = self.line_detection(roi, frame)

        roi_color = cv.cvtColor(roi, cv.COLOR_GRAY2BGR)
        if left_line:
            cv.line(roi_color, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), 4)
        if right_line:
            cv.line(roi_color, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 4)
            
        cv.imshow("ROI with Lanes (Red)", roi_color)
        cv.imshow("Lane Detection Output", combined)
        
        angle = self.compute_steering_angle(left_line, right_line, self.prev_angle, frame.shape[1])
        self.prev_angle = self.smoothen_angle_display(angle, self.prev_angle)
        smoothed_angle = self.prev_angle

        # Compute separate left and right wheel angles
        left_wheel_angle, right_wheel_angle = self.compute_ackermann_wheel_angles(smoothed_angle)

        # Kalman filter smoothing for left wheel
        self.kalman_left.predict()
        measurement_left = np.array([[np.float32(left_wheel_angle)]])
        est_left = self.kalman_left.correct(measurement_left)
        smoothed_left = float(est_left[0][0])

        # Kalman filter smoothing for right wheel
        self.kalman_right.predict()
        measurement_right = np.array([[np.float32(right_wheel_angle)]])
        est_right = self.kalman_right.correct(measurement_right)
        smoothed_right = float(est_right[0][0])

        # Publish steering angle
        self.angle_pub.publish(Float32(data=smoothed_angle))
        self.get_logger().info(f"Steering angle: {smoothed_angle:.2f} degrees")

        # Publish left and right wheel angles
        self.left_wheel_angle_pub.publish(Float32(data=smoothed_left))
        self.get_logger().info(f"Left wheel angle: {smoothed_left:.2f} degrees")

        self.right_wheel_angle_pub.publish(Float32(data=smoothed_right))
        self.get_logger().info(f"Right wheel angle: {smoothed_right:.2f} degrees")
        print()

        lane_msg = Float32MultiArray()
        lane_msg.data = list(map(float, left_line if left_line else [float('nan')] * 4)) + \
                        list(map(float, right_line if right_line else [float('nan')] * 4))
        self.lane_pub.publish(lane_msg)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(combined, encoding='bgr8'))
        
        cv.waitKey(1)

    def resize(self, image, scale=90):
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

    def canny(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
        return cv.Canny(blur, 30, 120)

    def region_of_interest(self, image):
        height, width = image.shape[:2]
        mask = np.zeros_like(image)
        polygon = np.array([[(
            int(0.001 * width), height),
            (int(0.25 * width), int(0.55 * height)),
            (int(0.75 * width), int(0.55 * height)),
            (width, height)
        ]], np.int32)
        color = 255 if len(image.shape) == 2 else (255, 255, 255)
        cv.fillPoly(mask, polygon, color)
        return cv.bitwise_and(image, mask)

    def line_detection(self, edges, original):
        left_lines, right_lines = [], []
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=20, maxLineGap=50)
        line_img = np.zeros_like(original)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:
                    continue
                (left_lines if slope < 0 else right_lines).append(line[0])
                cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        def avg_line(lines):
            if not lines:
                return None
            x1s, y1s, x2s, y2s = zip(*lines)
            return [int(np.mean(x1s)), int(np.mean(y1s)), int(np.mean(x2s)), int(np.mean(y2s))]

        left_avg = avg_line(left_lines)
        right_avg = avg_line(right_lines)

        if left_avg:
            cv.line(line_img, (left_avg[0], left_avg[1]), (left_avg[2], left_avg[3]), (0, 255, 0), 4)
        if right_avg:
            cv.line(line_img, (right_avg[0], right_avg[1]), (right_avg[2], right_avg[3]), (0, 255, 0), 4)

        combined = cv.addWeighted(original, 0.8, line_img, 1, 1)
        return combined, left_avg, right_avg

    def bottom_point(self, line):
        if line is None or any(np.isnan(line)):
            return None
        x1, y1, x2, y2 = line
        return (x1, y1) if y1 > y2 else (x2, y2)

    def compute_steering_angle(self, left_line, right_line, prev_angle, image_width=None):
        if image_width is None:
            width = 1152
        else:
            width = image_width
        cx = width // 2
        estimated_lane_width_pixels = 400
        pixels_per_meter = estimated_lane_width_pixels / 3.5
        max_deviation_meters = 0.3
        safe_offset_px = int(pixels_per_meter * max_deviation_meters)

        left_pt = self.bottom_point(left_line)
        right_pt = self.bottom_point(right_line)

        if left_pt and right_pt:
            mid_x = (left_pt[0] + right_pt[0]) // 2
            lane_width = abs(right_pt[0] - left_pt[0])
            if lane_width < 100 or lane_width > width * 0.8:
                return prev_angle
        elif right_pt:
            mid_x = right_pt[0] - safe_offset_px
        elif left_pt:
            mid_x = left_pt[0] + safe_offset_px
        else:
            return prev_angle

        dx = mid_x - cx
        norm_dx = dx / (width * 0.25)
        angle_rad = math.atan2(norm_dx, 1.0)
        angle_deg = math.degrees(angle_rad)
        scale = 0.35 if abs(norm_dx) <= 0.5 else 0.25
        scaled_angle = angle_deg * scale
        return max(min(scaled_angle, 25), -25)

    def compute_ackermann_wheel_angles(self, steering_angle_deg, wheelbase=2.5, track_width=1.5):
        """
        Compute left and right front wheel angles using Ackermann steering geometry.
        Returns tuple (left_wheel_angle_deg, right_wheel_angle_deg).
        """
        if abs(steering_angle_deg) < 1e-3:
            return 0.0, 0.0

        delta = math.radians(steering_angle_deg)
        L = wheelbase
        W = track_width

        R = L / math.tan(abs(delta))  # turning radius

        R_in = R - (W / 2)
        R_out = R + (W / 2)

        delta_in = math.atan(L / R_in)
        delta_out = math.atan(L / R_out)

        delta_in_deg = math.degrees(delta_in)
        delta_out_deg = math.degrees(delta_out)

        if steering_angle_deg < 0:  # turning left
            left_wheel = delta_in_deg
            right_wheel = delta_out_deg
        else:  # turning right or zero
            left_wheel = delta_out_deg
            right_wheel = delta_in_deg

        sign = 1 if steering_angle_deg > 0 else (-1 if steering_angle_deg < 0 else 0)

        left_wheel *= sign
        right_wheel *= sign

        return left_wheel, right_wheel

    def init_kalman_filter(self):
        kf = cv.KalmanFilter(2, 1)
        kf.measurementMatrix = np.array([[1, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kf.processNoiseCov = np.array([[1e-3, 0], [0, 1e-3]], np.float32)
        kf.measurementNoiseCov = np.array([[1e-2]], np.float32)
        kf.errorCovPost = np.eye(2, dtype=np.float32)
        kf.statePost = np.zeros((2, 1), np.float32)
        return kf

    def smoothen_angle_display(self, new_angle, prev_angle):
        alpha = 0.2
        smoothed = alpha * new_angle + (1 - alpha) * prev_angle
        self.angle_buffer.append(smoothed)
        return sum(self.angle_buffer) / len(self.angle_buffer)

def main(args=None):
    try:
        rclpy.init(args=args)
        node = UnifiedLaneKeepAssist()

        if not hasattr(node, 'cap') or not node.cap.isOpened():
            node.get_logger().error("Failed to open video. Exiting.")
            node.destroy_node()
            rclpy.shutdown()
            return

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as e:
        print(f"Exception caught in main: {e}")

    finally:
        if hasattr(node, 'cap') and node.cap.isOpened():
            node.cap.release()
        if 'node' in locals():
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv.destroyAllWindows()
        cv.waitKey(1)

if __name__ == '__main__':
    main()
