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
    navi.publish(0.0, 0.0, 0.0) 

    # navigator.setInitialPose(navi.publish(0.0, 0.0, 0.0) )
    navigator.waitUntilNav2Active() # почему-то робот появлялся в точке (0,0,0). Так что задаю initial_pose через nav2_params

    time_ = navigator.get_clock().now().to_msg()
    goal_pose = Navi.set_goal_pose(2.0, 0.0, -1.57, time_)
    navigator.goToPose(goal_pose)

    while not navigator.isNavComplete():
        pass
    print("done")
    time_ = navigator.get_clock().now().to_msg()

    goal_pose = Navi.set_goal_pose(2.0, -4.0, -1.57, time_)
    navigator.goToPose(goal_pose)
    while not navigator.isNavComplete():
        pass
    print("done")


    time_ = navigator.get_clock().now().to_msg()
    goal_pose = Navi.set_goal_pose(0.0, -4.0, -3.14, time_)
    navigator.goToPose(goal_pose)
    while not navigator.isNavComplete():
        pass
    print("done")

    time_ = navigator.get_clock().now().to_msg()
    goal_pose = Navi.set_goal_pose(-5.0, -5.0, -3.14, time_)
    navigator.goToPose(goal_pose)
    while not navigator.isNavComplete():
        pass
    print("done")



    time_ = navigator.get_clock().now().to_msg()
    goal_pose = Navi.set_goal_pose(-7.0, -5.0, -3.14, time_)
    navigator.goToPose(goal_pose)
    while not navigator.isNavComplete():
        pass
    print("done")


    time_ = navigator.get_clock().now().to_msg()
    goal_pose = Navi.set_goal_pose(-7.0, -10.0, -1.57, time_)
    navigator.goToPose(goal_pose)
    while not navigator.isNavComplete():
        pass
    print("done")

    time_ = navigator.get_clock().now().to_msg()
    goal_pose = Navi.set_goal_pose(-7.0, -15.0, -1.57, time_)
    navigator.goToPose(goal_pose)
    while not navigator.isNavComplete():
        pass
    print("done")
    rclpy.shutdown()

if __name__ == '__main__':
    main()