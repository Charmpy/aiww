import os

from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node



def generate_launch_description():


    # Include the robot_state_publisher launch file, provided by our own package. Force sim time to be enabled
    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!

    package_name='gz_main' #<--- CHANGE ME


    build_map = LaunchConfiguration('build_map')
    is_localization = LaunchConfiguration('is_localization')    
    map_file_path = LaunchConfiguration('map')    

    declare_build_map_cmd = DeclareLaunchArgument(
        'build_map', default_value='false', description='build map'
    )

    declare_localization_cmd = DeclareLaunchArgument(
        'is_localization', default_value='false', description='localization'
    )

    default_map = os.path.join(
            get_package_share_directory(package_name),
            'maps',
            'map.yaml'
            )     
    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map', default_value=default_map, description='Full path to map yaml file to load'
    )



    rsp = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','rsp.launch.py'
                )]), launch_arguments={'use_sim_time': 'true'}.items()
    )

    route_control = Node(
        package="route_controller",
        executable="driver",
        arguments=[]
    )

    default_world = os.path.join(
            get_package_share_directory(package_name),
            'worlds/',
            'obstacles.sdf'
            )    
    

    
    world = LaunchConfiguration('world')
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=default_world,
        description='World to load'
        )

    # Include the Gazebo launch file, provided by the ros_gz_sim package
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
                    launch_arguments={'gz_args': ['-r  -v4 ', world], 'on_exit_shutdown': 'true'}.items() # --render-engine ogre
             )

    # Run the spawner node from the ros_gz_sim package. The entity name doesn't really matter if you only have a single robot.
    spawn_entity = Node(package='ros_gz_sim', executable='create',
                        arguments=['-topic', 'robot_description',
                                   '-name', 'my_bot', "-z", '0.5' , "-y", '12.0', "-x", '7.0' ],
                        output='screen')

    # mover = Node(
    #         package='incredible_mover',
    #         namespace='',
    #         executable='odom_emu',

    #     )



    # Code for delaying a node (I haven't tested how effective it is)
    # 
    # First add the below lines to imports
    # from launch.actions import RegisterEventHandler
    # from launch.event_handlers import OnProcessExit
    #
    # Then add the following below the current diff_drive_spawner
    # delayed_diff_drive_spawner = RegisterEventHandler(
    #     event_handler=OnProcessExit(
    #         target_action=spawn_entity,
    #         on_exit=[diff_drive_spawner],
    #     )
    # )
    #
    # Replace the diff_drive_spawner in the final return with delayed_diff_drive_spawner

    # bridge_params = os.path.join(get_package_share_directory(package_name),'config','gz_bridge.yaml')
    # ros_gz_bridge = Node(
    #     package="ros_gz_bridge",
    #     executable="parameter_bridge",
    #     arguments=[
    #         '--ros-args',
    #         '-p',
    #         f'config_file:={bridge_params}',
    #     ]
    # )'


    # ros_gz_image_bridge = Node(
    #     package="ros_gz_image",
    #     executable="image_bridge",
    #     arguments=["/camera/image_raw"]
    # )




    bridge_params = os.path.join(get_package_share_directory(package_name),'config','gz_bridge.yaml')
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_params}',
        ]
    )

    # move_control = Node(
    #     package="cpp_omnidrive_gazebo_controller",
    #     executable="omni_gz_con",
    #     arguments=[
 
    #     ]
    # )

        # # to do map 
    # slam_params = os.path.join(get_package_share_directory(package_name),'config','mapper_params_online_async.yaml')
    # slam_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([os.path.join(
    #                 get_package_share_directory(package_name),'launch','online_async_launch.py'
    #             )]),
    #     # condition=IfCondition( build_map ),        
    #     launch_arguments={
    #         'use_sim_time': 'true',
    #         'params_file': slam_params,
    #     }.items()
    # )

    nav_params = os.path.join(get_package_share_directory(package_name),'config','nav2_params.yaml')
    start_localization = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','localization_launch.py'
                )]), 
                # condition=IfCondition( is_localization ), 
                launch_arguments={'map': map_file_path, 'use_sim_time': 'true', 'params_file': nav_params}.items()
    )
    
    start_navigation = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory(package_name),'launch','navigation_launch.py'
                )]), 
                # condition=IfCondition( is_navigation ), 
                launch_arguments={'use_sim_time': 'true', 'map_subscribe_transient_local': 'true', 'params_file': nav_params}.items()
    )


    # Launch them all!
    return LaunchDescription([
        rsp,
        world_arg,
        gazebo,
        ros_gz_bridge,
        spawn_entity,
        # move_control,

        # start_localization,
        # start_navigation,
        # route_control,
        # slam_launch,    

    ])
