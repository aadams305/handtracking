#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class StereoSplitter(Node):
    def __init__(self):
        super().__init__('stereo_splitter')
        self.bridge = CvBridge()

        # Hard-coded flip settings - change these as needed
        self.flip_horizontal = True
        self.flip_vertical = True
        self.flip_both = False

        # Subscription to the raw stereo image topic
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Make sure this is your camera's raw topic
            self.listener_callback,
            10
        )

        # Publishers for the left and right camera images
        self.pub_left = self.create_publisher(Image, '/cam0/image_raw', 10)
        self.pub_right = self.create_publisher(Image, '/cam1/image_raw', 10)
        self.pub_combined = self.create_publisher(Image, '/stereo_combined/image_raw', 10)
        
        self.get_logger().info('Stereo splitter with flip functionality started')

    def apply_flip(self, image):
        """Apply flipping based on hard-coded settings"""
        flipped_image = image.copy()
        if self.flip_both:
            flipped_image = cv2.flip(flipped_image, -1)
        else:
            if self.flip_horizontal:
                flipped_image = cv2.flip(flipped_image, 1)
            if self.flip_vertical:
                flipped_image = cv2.flip(flipped_image, 0)
        return flipped_image

    def listener_callback(self, msg):
        try:
            # Convert to mono8 for consistent output
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

            # Apply flipping to the entire stereo image
            cv_image_flipped = self.apply_flip(cv_image)

            height, width = cv_image_flipped.shape[:2]
            half_width = width // 2

            # Split stereo image
            left_image = cv_image_flipped[:, :half_width]
            right_image = cv_image_flipped[:, half_width:]

            # Convert back to ROS Image messages
            left_msg = self.bridge.cv2_to_imgmsg(left_image, encoding='mono8')
            right_msg = self.bridge.cv2_to_imgmsg(right_image, encoding='mono8')
            combined_msg = self.bridge.cv2_to_imgmsg(cv_image_flipped, encoding='mono8')

            # CRITICAL: Preserve the EXACT original timestamp and frame_id
            # This ensures camera timestamps match between topics for Kalibr
            current_stamp = msg.header.stamp
            left_msg.header.stamp = current_stamp
            right_msg.header.stamp = current_stamp
            combined_msg.header.stamp = current_stamp
            left_msg.header.frame_id = "cam0_link" # Or your left camera frame
            right_msg.header.frame_id = "cam1_link" # Or your right camera frame
            combined_msg.header.frame_id = msg.header.frame_id # Keep original frame_id for combined

            # Publish immediately to minimize additional latency
            self.pub_left.publish(left_msg)
            self.pub_right.publish(right_msg)
            self.pub_combined.publish(combined_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    splitter = StereoSplitter()
    rclpy.spin(splitter)
    splitter.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
