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

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

import math
import numpy as np
import ikfastpy
import time


class PublisherJointTrajectory(Node):
    def __init__(self):
        super().__init__("publisher_position_trajectory_controller")
        # Declare all parameters
        self.declare_parameter("controller_name", "position_trajectory_controller")
        self.declare_parameter("wait_sec_between_publish", 8)
        self.declare_parameter("moving_duration_time", 6)
        self.declare_parameter("goal_names", ["pos1", "pos2"])
        self.declare_parameter("final_goal1")
        self.declare_parameter("final_goal2")
        self.declare_parameter("joints")
        self.declare_parameter("objects_num", 1)
        self.declare_parameter("check_starting_point", False)
        self.declare_parameter("starting_point_limits")

        # Read parameters
        controller_name = self.get_parameter("controller_name").value
        wait_sec_between_publish = self.get_parameter("wait_sec_between_publish").value
        self.moving_duration_time = self.get_parameter("moving_duration_time").value
        goal_names = self.get_parameter("goal_names").value
        objects_num = self.get_parameter("objects_num").value
        self.final_goal1 = self.get_parameter("final_goal1").value
        self.final_goal2 = self.get_parameter("final_goal2").value
        self.final_pos1 = self.final_goal1[3:6]
        self.final_pos2 = self.final_goal2[3:6]
        self.joints = self.get_parameter("joints").value
        self.check_starting_point = self.get_parameter("check_starting_point").value
        self.starting_point = {}  #a dictionary
        
        # Read the object transformation matrix and center point from Runze code (copy it from .npy)
        self.objects_trans_all = []
        self.objects_center_point_all = []
        
        for i in range(objects_num):
            
            object_trans_filename = f'/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Vision_based_grasping_data/bounding_box_transform_matrix_{i}.npy'
            object_trans = np.load(object_trans_filename)
        
            if object_trans[2,1] < 0:
                object_trans[2] = object_trans[2] * (-1)
                object_trans[0] = object_trans[0] * (-1)
            object_trans = object_trans.transpose()
            
            self.objects_trans_all.append(object_trans)
        
            object_contour_point_filename = f'/home/zhanfeng/camera_ws/src/Realsense_python/camera_calibration/Vision_based_grasping_data/bounding_box_8pts_{i}.npy'
            object_contour_point = np.load(object_contour_point_filename)
        
            object_point_sum = np.array([0, 0, 0])
            for i in range(8):
                object_point_sum = object_point_sum + object_contour_point[i]
            object_center_point = object_point_sum / 8
            object_center_point[0] = object_center_point[0] - 0.01
            object_center_point[1] = object_center_point[1] - 0.05
            
            self.objects_center_point_all.append(object_center_point)
        

        if self.joints is None or len(self.joints) == 0:
            raise Exception('"joints" parameter is not set!')

        # starting point stuff
        if self.check_starting_point:
            # declare nested params
            for name in self.joints:
                param_name_tmp = "starting_point_limits" + "." + name
                self.declare_parameter(param_name_tmp, [-2 * 3.14159, 2 * 3.14159])
                self.starting_point[name] = self.get_parameter(param_name_tmp).value

            for name in self.joints:
                if len(self.starting_point[name]) != 2:
                    raise Exception('"starting_point" parameter is not set correctly!')
            self.joint_state_sub = self.create_subscription(
                JointState, "joint_states", self.joint_state_callback, 10
            )    
            #joint_state_callback can be used to get current position of joints
            
        # initialize starting point status
        if not self.check_starting_point:
            self.starting_point_ok = True
        else:
            self.starting_point_ok = False

        self.joint_state_msg_received = False

        #Read all end effector euler angle and position from parameters
        self.goals = []
        for name in goal_names:
            self.declare_parameter(name)
            goal = self.get_parameter(name).value
            if goal is None or len(goal) == 0:
                raise Exception(f'Values for goal "{name}" not set!')
            float_goal = []
            for value in goal:
                float_goal.append(float(value))
            self.goals.append(float_goal)
        
        self.get_logger().info('center point x1: {} '.format(self.objects_center_point_all[1]))
        
        self.joints_goals = []
        for i in range(objects_num):
            joints_goals_value = self.inverse_kinematics_cylinder(self.goals[i], self.objects_trans_all[i], self.objects_center_point_all[i])
            self.joints_goals.append(joints_goals_value)
            final_joints_goals_value1 = self.inverse_kinematics_cylinder(self.final_goal1, self.objects_trans_all[i], self.final_pos1)
            self.joints_goals.append(final_joints_goals_value1)
            final_joints_goals_value2 = self.inverse_kinematics_cylinder(self.final_goal2, self.objects_trans_all[i], self.final_pos2)
            self.joints_goals.append(final_joints_goals_value2)
            self.joints_goals.append(final_joints_goals_value1)
        
        
        publish_topic = "/" + controller_name + "/" + "joint_trajectory"

        self.get_logger().info(
            'Publishing {} joints goals on topic "{}" every {} s'.format(
                objects_num, publish_topic, wait_sec_between_publish
            )
        )

        self.publisher_ = self.create_publisher(JointTrajectory, publish_topic, 1)
        time.sleep(2)
        self.timer = self.create_timer(wait_sec_between_publish, self.timer_callback)
        self.i = 0
        
        
        
    def inverse_kinematics_cylinder(self, goals_value, object_trans, object_center_point):
    
        self.get_logger().info('Calculate inverse kinematics for goal value: {}'.format(object_center_point))
        ur5_kinematics = ikfastpy.PyKinematics()
        n_joints = ur5_kinematics.getDOF()
        
        #change ZYZ Euler angle to Trans Matrix for end effector (ee)
        phi = math.radians(goals_value[0])
        theta = math.radians(goals_value[1])
        psi = math.radians(goals_value[2])
        x = object_center_point[0]
        y = object_center_point[1]
        z = object_center_point[2]
        
        sp = math.sin(phi)
        cp = math.cos(phi)
        st = math.sin(theta)
        ct = math.cos(theta)
        ss = math.sin(psi)
        cs = math.cos(psi)
        
        nx = cp * ct * cs - sp * ss
        ny = sp * ct * cs + cp * ss
        nz = - st * cs
        ox = - cp * ct * ss - sp * cs
        oy = - sp * ct * ss + cp * cs
        oz = st * ss
        ax = cp * st
        ay = sp * st
        az = ct
        
        nx_r = object_trans[0,2]
        ny_r = object_trans[1,2]
        nz_r = object_trans[2,2]
        ox_r = object_trans[0,0]
        oy_r = object_trans[1,0]
        oz_r = object_trans[2,0]
        ax_r = object_trans[0,1]
        ay_r = object_trans[1,1]
        az_r = object_trans[2,1]
        
        T_ee = [[nx, ox, ax, x],[ny, oy, ay, y],[nz, oz, az, z]]
        
        #self.get_logger().info(f"SoftHand grasp center pose: \n {T_ee}")
        
        #check if input goals is okay, the z axis of input must face forward!
        if (az <= 0.1) :
            input_goals_ok = True
        else:
            input_goals_ok = False
            raise Exception('The input goals is incorrect. Facing upward!')
            
        if (z < -0.08):
            input_goals_ok = False
            raise Exception('The soft hand grasp position is too low!')
        
        if input_goals_ok :
        
            #change from end effector Trans to the 6 joint Trans
            ze_6 = -0.255    #grasp center: z: 255mm
            T6_0 = [[nx, ox, ax, x+ax*ze_6],[ny, oy, ay, y+ay*ze_6],[nz, oz, az, z+az*ze_6]]
        
            self.Trans = np.array(T6_0)
        
            joints_configs = ur5_kinematics.inverse(self.Trans.reshape(-1).tolist())
            n_solutions = int(len(joints_configs)/n_joints)
        
            joints_configs = np.reshape(joints_configs, (n_solutions,n_joints))
        
            # find the best joints_goals solution
            # the best solution is chosen to be closest to desired grasp position
            desired_joints_configs = [float(angle) for angle in [0, -90, -60, -100, 90, 180]]
            desired_joints_configs = [math.radians(angle) for angle in desired_joints_configs]
        
            # First, the joints solution should satisfy joint limits!
            joints_limits = {}
            joints_limits['shoulder_pan_joint'] = [math.radians(r) for r in [float(angle) for angle in [-90, 90+1]] ] 
            joints_limits['shoulder_lift_joint'] = [math.radians(r) for r in [float(angle) for angle in [-150, -10]] ]  # !!!!!!change -30 to -10
            joints_limits['elbow_joint'] = [math.radians(r) for r in [float(angle) for angle in [-150,15+1]] ] 
            joints_limits['wrist_1_joint'] = [math.radians(r) for r in [float(angle) for angle in [-230, 10]] ]     # lateral grasp
            #joints_limits['wrist_1_joint'] = [math.radians(r) for r in [float(angle) for angle in [-130, 80]] ]     # vertical grasp
            joints_limits['wrist_2_joint'] = [math.radians(r) for r in [float(angle) for angle in [-150, 170+1]] ]     # !!!!!!change 80 to 145
            joints_limits['wrist_3_joint'] = [math.radians(r) for r in [float(angle) for angle in [30, 300]] ] 
            
            valid_sols = []
            for sol in joints_configs:
                test_sol = np.ones(6) * 9999.0
                for i in range(6):
                    for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                        test_ang = sol[i] + add_ang
                        if (abs(test_ang) <= 2. * np.pi and abs(test_ang - desired_joints_configs[i] ) < abs(test_sol[i] - desired_joints_configs[i]) and test_ang > joints_limits[self.joints[i]][0] and test_ang < joints_limits[self.joints[i]][1] ):
                            test_sol[i] = test_ang
                if np.all(test_sol != 9999.):
                    valid_sols.append(test_sol)  # the element in the list is of array type.
            
            if not valid_sols:
                raise Exception('No valid solutions for input grasp goal position!')
                return None
            else:
                joints_configs_distance = np.sum((valid_sols - np.array(desired_joints_configs))**2, 1) 
                best_joints_config_indice = np.argmin(joints_configs_distance)
                best_joints_config = valid_sols[best_joints_config_indice].tolist()  #an array to list
                return best_joints_config


    def timer_callback(self):

        if self.starting_point_ok:

            traj = JointTrajectory()
            traj.joint_names = self.joints
            point = JointTrajectoryPoint()
            point.positions = self.joints_goals[self.i]  #the inverse kinematics results
            point.time_from_start = Duration(sec=self.moving_duration_time)

            traj.points.append(point)
            self.publisher_.publish(traj)

            self.i += 1
            self.i %= len(self.joints_goals)

        elif self.check_starting_point and not self.joint_state_msg_received:
            self.get_logger().warn(
                'Start configuration could not be checked! Check "joint_state" topic!'
            )
        else:
            self.get_logger().warn("Start configuration is not within configured limits!")

    def joint_state_callback(self, msg):

        if not self.joint_state_msg_received:

            # check start state
            limit_exceeded = [False] * len(msg.name)
            for idx, enum in enumerate(msg.name):
                if (msg.position[idx] < self.starting_point[enum][0]) or (
                    msg.position[idx] > self.starting_point[enum][1]
                ):
                    self.get_logger().warn(f"Starting point limits exceeded for joint {enum} !")
                    limit_exceeded[idx] = True

            if any(limit_exceeded):
                self.starting_point_ok = False
            else:
                self.starting_point_ok = True

            self.joint_state_msg_received = True
        else:
            return


def main(args=None):
    rclpy.init(args=args)

    publisher_joint_trajectory = PublisherJointTrajectory()

    rclpy.spin(publisher_joint_trajectory)
    publisher_joint_trajectory.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
