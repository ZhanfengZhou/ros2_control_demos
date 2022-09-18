# Copyright 2021 Stogl Robotics Consulting UG (haftungsbeschränkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from glob import glob

from setuptools import setup

package_name = "ros2_control_test_nodes"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, glob("launch/*.launch.py")),
        ("share/" + package_name + "/configs", glob("configs/*.*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    author="Denis Štogl",
    author_email="denis@stogl.de",
    maintainer="Denis Štogl",
    maintainer_email="denis@stogl.de",
    keywords=["ROS"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    description="Demo nodes for showing and testing functionalities of ros2_control framework.",
    long_description="""\
Demo nodes for showing and testing functionalities of the ros2_control framework.""",
    license="Apache License, Version 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "publisher_forward_position_controller = \
                ros2_control_test_nodes.publisher_forward_position_controller:main",
            "publisher_joint_trajectory_controller = \
                ros2_control_test_nodes.publisher_joint_trajectory_controller:main",
            "publisher_inverse_kinematics_controller = \
                ros2_control_test_nodes.publisher_inverse_kinematics_controller:main",
            "publisher_end_effector_cartesian_move_controller = \
                ros2_control_test_nodes.publisher_end_effector_cartesian_move_controller:main",
            "publisher_grasp_original_pose_move = \
                ros2_control_test_nodes.publisher_grasp_original_pose_move:main",
            "publisher_touch_sensor_ur5_IKcontroller = \
                ros2_control_test_nodes.publisher_touch_sensor_ur5_IKcontroller:main",
            "publisher_camera_calibration_ik_controller = \
                ros2_control_test_nodes.publisher_camera_calibration_ik_controller:main",
            "publisher_vision_guided_grasp_controller = \
                ros2_control_test_nodes.publisher_vision_guided_grasp_test_controller:main",
            "publisher_vision_based_grasp_controller = \
                ros2_control_test_nodes.publisher_vision_based_grasp_preset_controller:main",
            "publisher_grasp_from_human_5direction_controller = \
                ros2_control_test_nodes.publisher_grasp_from_human_5direction_controller:main",
            "node_vision_based_grasp_marker_from_human = \
                ros2_control_test_nodes.node_vision_based_grasp_marker_from_human:main",    
            "node_human_robot_interactive_grasp_with_marker = \
                ros2_control_test_nodes.node_human_robot_interactive_grasp_with_marker:main",  
            "node_human_robot_interactive_grasp_yolo = \
                ros2_control_test_nodes.node_human_robot_interactive_grasp_yolo:main", 
            "node_human_robot_interactive_grasp_yolo_server = \
                ros2_control_test_nodes.node_human_robot_interactive_grasp_yolo_server:main",  
        ],
    },
)
