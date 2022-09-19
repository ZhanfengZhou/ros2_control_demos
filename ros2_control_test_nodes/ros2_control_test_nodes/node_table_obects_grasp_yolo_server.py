# Use paramters in yaml file to control the robotic arm.
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

import rclpy
from rclpy.node import Node
from rclpy.exceptions import ParameterNotDeclaredException
from rclpy.executors import MultiThreadedExecutor

from threading import Thread

from builtin_interfaces.msg import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

from example_interfaces.srv import SetBool
from ur_msgs.srv import YOLOOutput, Task

import math
import time
import numpy as np
import ikfastpy
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi

import termcolor


class Node_vision_based_grasp_object_from_table(Node):
    def __init__(self):
        super().__init__("Node_vision_based_grasp_object_from_table")

        ## Declare all parameters
        self.declare_parameter("controller_name", "position_trajectory_controller")
        self.declare_parameter("to_sleep_traj_duration", 6)
        self.declare_parameter("to_start_traj_duration", 6)
        self.declare_parameter("to_grasp_traj_duration", 6)
        self.declare_parameter("to_final_traj_duration", 6)
        self.declare_parameter("sleep_goal")
        self.declare_parameter("start_goal")
        self.declare_parameter("grasp_goal")
        self.declare_parameter("grasp_goal_top")
        self.declare_parameter("grasp_goal_bottom")
        self.declare_parameter("final_goal")
        self.declare_parameter("joints")

        ## Read parameters
        controller_name = self.get_parameter("controller_name").value
        self.sleep_traj_duration = self.get_parameter("to_sleep_traj_duration").value
        self.start_traj_duration = self.get_parameter("to_start_traj_duration").value
        self.grasp_traj_duration = self.get_parameter("to_grasp_traj_duration").value
        self.final_traj_duration = self.get_parameter("to_final_traj_duration").value
        self.joints = self.get_parameter("joints").value
        self.sleep_goal = self.get_parameter("sleep_goal").value
        self.start_goal = self.get_parameter("start_goal").value
        self.grasp_goal = self.get_parameter("grasp_goal").value
        self.grasp_goal_top = self.get_parameter("grasp_goal_top").value
        self.grasp_goal_bottom = self.get_parameter("grasp_goal_bottom").value
        self.final_goal = self.get_parameter("final_goal").value

        if self.joints is None or len(self.joints) == 0:
            raise Exception('"joints" parameter is not set!')

        ## Read all end effector goal and transform to joint goal of robotic arm
        self.get_logger().info('Calculating inverse kinematics to get joint value')
        self.sleep_joints_goals = self.inverse_kinematics(self.sleep_goal)
        self.start_joints_goals = self.inverse_kinematics(self.start_goal)
        self.grasp_joints_goals = []
        self.grasp_joints_goals.append(self.inverse_kinematics(self.grasp_goal))
        self.final_joints_goals = self.inverse_kinematics(self.final_goal)


        ## Create a publisher to publish joint goal to robotic arm with a timer
        publish_topic = "/" + controller_name + "/" + "joint_trajectory"

        self.get_logger().info(f'Publishing joints goals on topic "{publish_topic}"')
        self.publisher_ = self.create_publisher(JointTrajectory, publish_topic, 1, callback_group=MutuallyExclusiveCallbackGroup())
        
        self.arrived_start = False
        self.softhand_grasp = False



        ## Create a subsriber for reading joint state
        self.joint_state_sub = self.create_subscription(
            JointState, 
            "joint_states", 
            self.joint_state_callback, 
            10, callback_group=MutuallyExclusiveCallbackGroup()
        )
        # joint_state_callback can be used to get current position of joints
        self.joint_state_msg_received = False


        ## Create a service for reveiving object center info from yolo detect client: "yolo_xyz"
        self.yoloservice = self.create_service(
            YOLOOutput,
            'yolo_xyz',
            self.yoloservice_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )       


        ## Create a service for reveiving task number, "task"
        self.taskservice = self.create_service(
            Task,
            'task',
            self.taskservice_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )       
        self.task_num = 0


        ## Create a client to send request to soft robotic hand to grasp object
        self.client_softhand = self.create_client(
            SetBool, 
            'Grasp_object', 
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.request_softhand = SetBool.Request()

        self.get_logger().info('Node initialization finished.')
    
    
    
    def taskservice_callback(self, request, response):
        self.get_logger().info(termcolor.colored('taskservice_callback is called', 'cyan'))

        self.task_num = request.task

        self.get_logger().info(termcolor.colored(
            f'task request received, task number: {self.task_num}', 'cyan'))
        
        response.success = True
        return response


    def yoloservice_callback(self, request, response):
        self.get_logger().info(termcolor.colored('yoloservice_callback is called', 'cyan'))
        object_x = request.object_center_x
        object_y = request.object_center_y
        object_z = request.object_center_z
        grasp_dir = request.grasp_dir

        self.get_logger().info(termcolor.colored(
            f'yolo request received, object center: [{object_x}, {object_y}, {object_z}, {grasp_dir}]', 'cyan'))

        if self.joint_state_msg_received:
            self.object_pos = [object_x, object_y, object_z]
            Trans_camera2base = self.camera2base_transform()
            self.object_pos_array = np.array(
                [self.object_pos[0], self.object_pos[1], self.object_pos[2], 1])
            self.object_pos_array = np.matrix(self.object_pos_array).transpose()
            self.object_pos_array_global = np.matmul(Trans_camera2base, self.object_pos_array)

            if int(grasp_dir) == 1:
                self.get_logger().info(termcolor.colored(
                    f'Grasp object from forward direction', 'cyan'))
                self.new_grasp_goal = [
                    self.grasp_goal[0], 
                    self.grasp_goal[1], 
                    self.grasp_goal[2], 
                    self.object_pos_array_global[0,0], 
                    self.object_pos_array_global[1,0], 
                    self.object_pos_array_global[2,0]
                ]
                new_grasp_joints_goals_value = self.inverse_kinematics(self.new_grasp_goal)
                self.grasp_joints_goals[0] = new_grasp_joints_goals_value

            elif int(grasp_dir) == 2:
                self.grasp_goal = self.grasp_goal_top
                self.get_logger().info(termcolor.colored(
                    f'Grasp object from top direction', 'cyan'))

                self.new_grasp_goal = [
                    self.grasp_goal[0], 
                    self.grasp_goal[1], 
                    self.grasp_goal[2], 
                    self.object_pos_array_global[0,0], 
                    self.object_pos_array_global[1,0], 
                    self.object_pos_array_global[2,0]+0.01
                ]
                new_grasp_joints_goals_value = self.inverse_kinematics(self.new_grasp_goal)
                self.grasp_joints_goals[0] = new_grasp_joints_goals_value
                
                self.new_grasp_goal = [
                    self.grasp_goal[0], 
                    self.grasp_goal[1], 
                    self.grasp_goal[2], 
                    self.object_pos_array_global[0,0], 
                    self.object_pos_array_global[1,0], 
                    self.object_pos_array_global[2,0]
                ]
                new_grasp_joints_goals_value = self.inverse_kinematics(self.new_grasp_goal)
                self.grasp_joints_goals.append(new_grasp_joints_goals_value)

            elif int(grasp_dir) == 3:
                self.grasp_goal = self.grasp_goal_bottom
                self.get_logger().info(termcolor.colored(
                    f'Grasp object from bottom direction', 'cyan'))
                
                self.new_grasp_goal = [
                    self.grasp_goal[0], 
                    self.grasp_goal[1], 
                    self.grasp_goal[2], 
                    self.object_pos_array_global[0,0], 
                    self.object_pos_array_global[1,0], 
                    self.object_pos_array_global[2,0]-0.01
                ]
                new_grasp_joints_goals_value = self.inverse_kinematics(self.new_grasp_goal)
                self.grasp_joints_goals[0] = new_grasp_joints_goals_value
                
                self.new_grasp_goal = [
                    self.grasp_goal[0], 
                    self.grasp_goal[1], 
                    self.grasp_goal[2], 
                    self.object_pos_array_global[0,0], 
                    self.object_pos_array_global[1,0], 
                    self.object_pos_array_global[2,0]
                ]
                new_grasp_joints_goals_value = self.inverse_kinematics(self.new_grasp_goal)
                self.grasp_joints_goals.append(new_grasp_joints_goals_value)

            else:
                self.get_logger().info(termcolor.colored(
                    f'Wrong direction input from yolo detect', 'cyan'))

            self.get_logger().info(termcolor.colored(
                f'Object grasp target sent to the robotic arm successfully', 'cyan'))
        # else:
        #     self.get_logger().warn('Start configuration could not be checked! Check "joint_state" topic!')

        response.success = True
        return response

        

    def camera2base_transform(self):

        # hand-eye calibration matrix load numpy array
        Trans_camera2end = np.loadtxt(
            "/home/zhanfeng/camera_ws/src/Realsense_python/camera_l515_test/3D_reconstruction_images/hand_eye_calibration_matrix.txt")
        Trans_camera2end[:, 3] = Trans_camera2end[:, 3]/1000
        Trans_camera2end[0, 3] = Trans_camera2end[0, 3] - 0.08
        Trans_camera2end[3, 3] = 1.0
        # get the robotic arm transformation matrix from end effector to base
        if self.joint_state_msg_received:

            # coefficients
            d = np.matrix([0.089159, 0, 0, 0.10915, 0.09465, 0.0823])
            a = np.matrix([0, -0.425, -0.39225, 0, 0, 0])
            alph = np.matrix([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0])

            A_1 = self.joint2joint_transform(
                1, self.joint_angles_feedback, d, a, alph)
            A_2 = self.joint2joint_transform(
                2, self.joint_angles_feedback, d, a, alph)
            A_3 = self.joint2joint_transform(
                3, self.joint_angles_feedback, d, a, alph)
            A_4 = self.joint2joint_transform(
                4, self.joint_angles_feedback, d, a, alph)
            A_5 = self.joint2joint_transform(
                5, self.joint_angles_feedback, d, a, alph)
            A_6 = self.joint2joint_transform(
                6, self.joint_angles_feedback, d, a, alph)

            T_06 = A_1*A_2*A_3*A_4*A_5*A_6

            #self.get_logger().info(f'input: {T_06}, camera2end: {Trans_camera2end}')

            Trans_camera2base = T_06 * Trans_camera2end

            return Trans_camera2base
        else:

            raise Exception('No joint_state_msg_received!')


    def joint2joint_transform(self, n, th, d, a, alph):

        T_a = np.matrix(np.identity(4), copy=False)
        T_a[0, 3] = a[0, n-1]
        T_d = np.matrix(np.identity(4), copy=False)
        T_d[2, 3] = d[0, n-1]

        Rzt = np.matrix([[cos(th[n-1]), -sin(th[n-1]), 0, 0],
                         [sin(th[n-1]),  cos(th[n-1]), 0, 0],
                         [0,               0,              1, 0],
                         [0,               0,              0, 1]], copy=False)

        Rxa = np.matrix([[1, 0,                 0,                  0],
                         [0, cos(alph[0, n-1]), -sin(alph[0, n-1]),   0],
                         [0, sin(alph[0, n-1]),  cos(alph[0, n-1]),   0],
                         [0, 0,                 0,                  1]], copy=False)

        A_i = T_d * Rzt * T_a * Rxa

        return A_i

    def inverse_kinematics(self, goals_value):
        #Calculating the ur5 inverse kinematics for goal value

        self.get_logger().info(
            'Calculate inverse kinematics for goal value: {}'.format(goals_value))
        ur5_kinematics = ikfastpy.PyKinematics()
        n_joints = ur5_kinematics.getDOF()

        # get desired euler angle
        phi = math.radians(goals_value[0])
        theta = math.radians(goals_value[1])
        psi = math.radians(goals_value[2])

        # get center grasping point of object
        x = goals_value[3]
        y = goals_value[4]
        z = goals_value[5]

        # change ZYZ Euler angle to Trans Matrix for end effector (ee)
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

        T_ee = [[nx, ox, ax, x], [ny, oy, ay, y], [nz, oz, az, z]]

        #self.get_logger().info(f"SoftHand grasp center pose: \n {T_ee}")

        # check if input goals is okay, the z axis of input must face forward!
#        if (ax >= -0.1) :
#            input_goals_ok = True
#        else:
#            input_goals_ok = False
#            raise Exception('The input goals is incorrect. Facing backward!')
        input_goals_ok = True

        if ((az <= 0.0) and (z < -0.04)) or ((az > 0.5) and (z < 0.35)):
            input_goals_ok = False
            raise Exception('The soft hand grasp position is too low!')

        if input_goals_ok:

            # change from end effector Trans to the 6 joint Trans
            # zcamera_6 = -0.175    #camera: z: 175mm
            z_center = -0.18  # grasp center: z: 230mm

            T6_0 = [[nx, ox, ax, x+ax*z_center], [ny, oy, ay, y+ay*z_center], [nz, oz, az, z+az*z_center]]

            self.Trans = np.array(T6_0)

            joints_configs = ur5_kinematics.inverse(
                self.Trans.reshape(-1).tolist())
            n_solutions = int(len(joints_configs)/n_joints)

            joints_configs = np.reshape(
                joints_configs, (n_solutions, n_joints))

            # find the best joints_goals solution
            # the best solution is chosen to be closest to desired grasp position
            desired_joints_configs = [float(angle)
                                      for angle in [0, -90, -90, 0, 90, 270]]
            desired_joints_configs = [math.radians(
                angle) for angle in desired_joints_configs]

            # First, the joints solution should satisfy joint limits!
            joints_limits = {}
            joints_limits['shoulder_pan_joint'] = [math.radians(
                r) for r in [float(angle) for angle in [-100, 90+1]]]
            joints_limits['shoulder_lift_joint'] = [math.radians(r) for r in [float(
                angle) for angle in [-150, -30]]] 
            joints_limits['elbow_joint'] = [math.radians(
                r) for r in [float(angle) for angle in [-150, 0+1]]]
            joints_limits['wrist_1_joint'] = [math.radians(r) for r in [float(
                angle) for angle in [-180, 100]]] 
            joints_limits['wrist_2_joint'] = [math.radians(r) for r in [float(
                angle) for angle in [-150, 145+1]]]  
            joints_limits['wrist_3_joint'] = [math.radians(
                r) for r in [float(angle) for angle in [90, 350]]]

            valid_sols = []
            for sol in joints_configs:
                test_sol = np.ones(6) * 9999.0
                for i in range(6):
                    for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                        test_ang = sol[i] + add_ang
                        if (abs(test_ang) <= 2. * np.pi and abs(test_ang - desired_joints_configs[i]) < abs(test_sol[i] - desired_joints_configs[i]) and test_ang > joints_limits[self.joints[i]][0] and test_ang < joints_limits[self.joints[i]][1]):
                            test_sol[i] = test_ang
                #self.get_logger().info('test solution: {}'.format(test_sol))

                if np.all(test_sol != 9999.):
                    # the element in the list is of array type.
                    valid_sols.append(test_sol)

            if not valid_sols:
                raise Exception(
                    'No valid solutions for input grasp goal position!')
                
            else:
                joints_configs_distance = np.sum(
                    (valid_sols - np.array(desired_joints_configs))**2, 1)
                best_joints_config_indice = np.argmin(joints_configs_distance)
                # an array to list
                best_joints_config = valid_sols[best_joints_config_indice].tolist(
                )
                return best_joints_config

    
    def run(self):
        self.get_logger().info(termcolor.colored('Robotic arm initializing', 'yellow'))

        # First move to sleep position
        self.pub_joint_traj(self.sleep_joints_goals, self.sleep_traj_duration)
        

        while True:
            time.sleep(0.1)

            if (self.task_num == 1):  # task num 1: move to sleep position
                self.get_logger().info(termcolor.colored('Robotic arm moving to sleep position', 'yellow'))
                self.pub_joint_traj(self.sleep_joints_goals, self.sleep_traj_duration)
                self.get_logger().info(termcolor.colored('Robotic arm sleeping', 'yellow'))
                self.arrived_start = False
                self.task_num = 0
            elif (self.task_num == 2) and (not self.arrived_start):  # task num 2: move to sleep position
                self.get_logger().info(termcolor.colored('Robotic arm moving to start position', 'yellow'))
                self.pub_joint_traj(self.start_joints_goals, self.start_traj_duration)
                self.get_logger().info(termcolor.colored('Robotic arm started: ready to grasp', 'yellow'))
                self.arrived_start = True

            elif (self.task_num == 3) and (self.arrived_start):  # task num 3: grasp object and place to final position
                time.sleep(1)
                self.get_logger().info(termcolor.colored('Robotic arm approaching the object', 'yellow'))
                for i in range(len(self.grasp_joints_goals)):
                    self.pub_joint_traj(self.grasp_joints_goals[i], self.grasp_traj_duration)  # move to grasp
                self.get_logger().info(termcolor.colored('Robotic arm arrived object position', 'yellow'))
                #request soft hand grasp object
                self.softhand_grasp = True
                time.sleep(1)
                self.send_request_softhand()
                time.sleep(1)
                
                self.get_logger().info(termcolor.colored('Robotic arm moving to place object', 'yellow'))
                self.pub_joint_traj(self.final_joints_goals, self.final_traj_duration)  # move to release
                self.get_logger().info(termcolor.colored('Robotic arm arrived collection position', 'yellow'))
                #request soft hand release object
                self.softhand_grasp = False
                time.sleep(1)
                self.send_request_softhand()
                time.sleep(1)

                self.get_logger().info(termcolor.colored('Robotic arm moving back to start position', 'yellow'))
                self.pub_joint_traj(self.start_joints_goals, self.start_traj_duration)  # move back to start
                self.get_logger().info(termcolor.colored('Robotic arm started: ready to grasp', 'yellow'))
                self.arrived_start = True
                self.task_num = 0
            else:
                continue
                

            # if not self.joint_state_msg_received:
            #     self.get_logger().warn(
            #         'Start configuration could not be checked! Check "joint_state" topic!'
            #     )
    

    def pub_joint_traj(self, joints_goals, traj_duration):
        traj = JointTrajectory()
        traj.joint_names = self.joints
        point = JointTrajectoryPoint()
        point.positions = joints_goals   # list
        point.time_from_start = Duration(sec=traj_duration)
        traj.points.append(point)
        self.publisher_.publish(traj)
        self.get_logger().info(termcolor.colored('Publishing trajectory to robotic arm joints', 'yellow'))
    
        time.sleep(traj_duration)
        time.sleep(1)


    def joint_state_callback(self, msg):
        # get joint angle feedback from the robotic arm motor.

        #self.get_logger().info(f'Joint state name: {msg.name}')
        #self.get_logger().info(f'Joint state position: {msg.position}')  # in radius; msg.position is a numpy array
        shoulder_pan_joint_angle = msg.position[2]
        shoulder_lift_joint_angle = msg.position[1]
        elbow_joint_angle = msg.position[0]
        wrist_1_joint_angle = msg.position[3]
        wrist_2_joint_angle = msg.position[4]
        wrist_3_joint_angle = msg.position[5]
        self.joint_angles_feedback = np.array([shoulder_pan_joint_angle, shoulder_lift_joint_angle,
                                                elbow_joint_angle, wrist_1_joint_angle, wrist_2_joint_angle, wrist_3_joint_angle])
        self.joint_state_msg_received = True



    def send_request_softhand(self):
        # send request (a bool variable) to robotic hand server to grasp or release objects
        if not self.client_softhand.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(termcolor.colored('softhand service not available, wait and send request again...', 'green'))
        else:
            if self.softhand_grasp:
                self.request.data = True
                future = self.client_softhand.call_async(self.request)
                self.get_logger().info(termcolor.colored("Requests Sent to SoftHand: Grasp object", 'green'))
                
                while not future.done():
                    pass
                self.get_logger().info(termcolor.colored(f'Service finished: Object Grasped', 'green'))

            else:
                self.request.data = False
                future = self.client_softhand.call_async(self.request)
                self.get_logger().info(termcolor.colored("Requests Sent to SoftHand: Release object", 'green'))
                
                while not future.done():
                    pass
                self.get_logger().info(termcolor.colored(f'Service finished: Object Released', 'green'))



def main(args=None):
    rclpy.init(args=args)

    Node_vision_based_grasp_object_from_table = Node_vision_based_grasp_object_from_table()

    # This creates a parallel thread of execution that will execute the `send_request` method of the client node. 
    # This is because I want the send request to run concurrently with the callbacks of the node.
    thread = Thread(target=Node_vision_based_grasp_object_from_table.run)
    thread.start()

    # I am using a MultiThreadedExecutor here as I want all the callbacks to run on a different thread each 
    executor = MultiThreadedExecutor()
    try:
        executor.add_node(Node_vision_based_grasp_object_from_table)
        executor.spin()
    except KeyboardInterrupt:
        pass


    Node_vision_based_grasp_object_from_table.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
