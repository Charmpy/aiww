import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

MOD = 10

class ImageSubscriber(Node):
    def __init__(self):
        print(cv2.__version__)
        super().__init__('image_subscriber')
        self.cam_sub = self.create_subscription(Image,'/camera/image', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image,'/depth_camera/image', self.depth_callback, 10)
        self.bridge = CvBridge()
        self.get_logger().info('Image subscriber started')
        self.k = 0
        if not os.path.exists("/aiww/data/camera"):
            os.makedirs("/aiww/data/camera")
        if not os.path.exists("/aiww/data/depth_camera"):
            os.makedirs("/aiww/data/depth_camera")

    def image_callback(self, msg):
        try:
            self.rgb_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def depth_callback(self, msg):
        global MOD
        try:
            depth_img = self.bridge.imgmsg_to_cv2(msg, "32FC1")
            depth_img = depth_img/10
            if self.k % MOD == 0:
                cv2.imwrite("/aiww/data/camera/img" + str(self.k//MOD) + ".png", self.rgb_img)
                cv2.imwrite("/aiww/data/depth_camera/img" + str(self.k//MOD) + ".png", depth_img*200)
            self.k += 1
            cv2.imshow("Camera Image", self.rgb_img)
            cv2.imshow("Depth Image", depth_img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()