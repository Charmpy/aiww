import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from .navi import Navi
import time
import math


from geometry_msgs.msg import PoseStamped # Pose with ref frame and timestamp
from rclpy.duration import Duration # Handles time for ROS 2
import rclpy # Python client library for ROS 2
 
from .robot_navigator import BasicNavigator # Helper module
from .util import Servo, Gripper, ServoControl
from std_msgs.msg import Int16
from std_msgs.msg import Int64
from geometry_msgs.msg import Twist 
# from req_res_str_service.srv import ReqRes 

from .navi import RobotUtil

def main(args=None):
    rclpy.init(args=args)



    navigator = BasicNavigator()

    navi = Navi()    
    navi.publish(1.0, -0.3, 0.111) 
    navigator.waitUntilNav2Active() # почему-то робот появлялся в точке (0,0,0). Так что задаю initial_pose через nav2_params

    
    # goal_pose = Navi.set_goal_pose(x, y, old_rot, time_)
    # navigator.goToPose(goal_pose)
    # while not navigator.isNavComplete():
    #     pass
    # print("done".join(RE.get_str()))
    
    rclpy.shutdown()

    

if __name__ == '__main__':
    main()