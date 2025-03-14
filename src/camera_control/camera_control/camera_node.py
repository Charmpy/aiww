import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os


class ImageSubscriber(Node):
    def __init__(self):
        print(cv2.__version__)
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image',  
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.get_logger().info('Image subscriber started')
        self.k = 0
        if not os.path.exists("/aiww/data/camera"):
            os.makedirs("/aiww/data/camera")

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            if self.k % 3 == 0:
                cv2.imwrite("/aiww/data/camera/img" + str(self.k//3) + ".png", image)
            self.k += 1
            cv2.imshow("Camera Image", image)
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