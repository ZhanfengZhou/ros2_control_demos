import math
import numpy as np
import ikfastpy
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Duration

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

# Read all end effector euler angle and position from parameters
#goals = [[-90.0, 90.0, 90.0, 0.6, -0.109, 0.3],[-90.0, 90.0, 90.0, 0.4, -0.109, 0.6],[-90.0, 90.0, 90.0, 0.4, -0.3, 0.6],[-90.0, 90.0, 90.0, 0.4, -0.3, 0.3]]
goals = [[-89.9, -179.9, 89.9, 0.4, -0.109, 0.3],]   #okay....should be have 0.1 degree input error to make the ikfastpy code working!!!!!! 



joints_goals = []
for goals_value in goals:   
        
    ur5_kinematics = ikfastpy.PyKinematics()
    n_joints = ur5_kinematics.getDOF()
        
    #change ZYZ Euler angle to Trans Matrix for end effector (ee)
    phi = math.radians(goals_value[0])
    theta = math.radians(goals_value[1])
    psi = math.radians(goals_value[2])
    #phi = round(math.radians(goals_value[0]),5)
    #theta = round(math.radians(goals_value[1]),5)
    #psi = round(math.radians(goals_value[2]),5)
    print(phi,theta,psi)
    x = goals_value[3]
    y = goals_value[4]
    z = goals_value[5]
        
    sp = math.sin(phi)
    cp = math.cos(phi)
    st = math.sin(theta)
    ct = math.cos(theta)
    ss = math.sin(psi)
    cs = math.cos(psi)
    print(sp,cp,st,ct,ss,cs)
    
    nx = cp * ct * cs - sp * ss
    ny = sp * ct * cs + cp * ss
    nz = - st * cs
    print(nx,ny,nz)
    ox = - cp * ct * ss - sp * cs
    oy = - sp * ct * ss + cp * cs
    oz = st * ss
    ax = cp * st
    ay = sp * st
    az = ct
    
    #nx = round(cp * ct * cs - sp * ss,5)
    #ny = round(sp * ct * cs + cp * ss,5)
    #nz = round(- st * cs,5)
    #print(nx,ny,nz)
    #ox = round(- cp * ct * ss - sp * cs,5)
    #oy = round(- sp * ct * ss + cp * cs,5)
    #oz = round(st * ss,5)
    #ax = round(cp * st,5)
    #ay = round(sp * st,5)
    #az = round(ct,5)
        
    T_ee = [[nx, ox, ax, x],[ny, oy, ay, y],[nz, oz, az, z]]
      
    print(f"End effector pose: \n {T_ee}")
    
    #check if input goals is okay, the z axis of input must face forward!
    if (ax >= -0.1) :
        input_goals_ok = True
    else:
        input_goals_ok = False
        raise Exception('The input goals is incorrect. Facing backward!')
        
    if ((az <= 0.0) and (z < 0)) or ((az > 0.5) and (z < 0.35)) :
        input_goals_ok = False
        raise Exception('The soft hand grasp position is too low!')
        
    #change from end effector Trans to the 6 joint Trans
    ze_6 = -0.175    #camera: z: 175mm
    zcamera_6 = -0.255    #grasp center: z: 255mm
        
    T6_0 = [[nx, ox, ax, x+ax*ze_6],[ny, oy, ay, y+ay*ze_6],[nz, oz, az, z+az*ze_6]]
    print(f"T6_0 pose: \n {T6_0}")
        
    Trans = np.array(T6_0)
    print(f"Trans: \n {Trans}")
        
    joints_configs = ur5_kinematics.inverse(Trans.reshape(-1).tolist())
    n_solutions = int(len(joints_configs)/n_joints)
        
    joints_configs = np.reshape(joints_configs, (n_solutions,n_joints))
    print(f"Joints config: \n {joints_configs/3.14159*180}")
       
    # find the best joints_goals solution
    desired_joints_configs = [float(angle) for angle in [0, -90, -60, -120, 90, 0]]
    desired_joints_configs = [math.radians(angle) for angle in desired_joints_configs]

    joints_limits = {}
    joints_limits['shoulder_pan_joint'] = [math.radians(r) for r in [float(angle) for angle in [-90, 90+1]] ] 
    joints_limits['shoulder_lift_joint'] =[math.radians(r) for r in [float(angle) for angle in [-150, -30]] ] 
    joints_limits['elbow_joint'] = [math.radians(r) for r in [float(angle) for angle in [-150,150+1]] ] 
    joints_limits['wrist_1_joint'] =[math.radians(r) for r in [float(angle) for angle in [-300, 100]] ] 
    joints_limits['wrist_2_joint'] =[math.radians(r) for r in [float(angle) for angle in [-150, 80+1]] ] 
    joints_limits['wrist_3_joint'] = [math.radians(r) for r in [float(angle) for angle in [-181, 180]] ] 
    print(f'joints limits: {joints_limits}')
    
    joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
    
    valid_sols = []
    for sol in joints_configs:
        test_sol = np.ones(6) * 9999.0
        for i in range(6):
            for add_ang in [-2. * np.pi, 0, 2. * np.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2. * np.pi and abs(test_ang - desired_joints_configs[i] ) < abs(test_sol[i] - desired_joints_configs[i]) and test_ang > joints_limits[joints[i]][0] and test_ang < joints_limits[joints[i]][1] ):
                    test_sol[i] = test_ang
                    print(test_ang/3.14*180)
        print(test_sol)
        if np.all(test_sol != 9999.):
            valid_sols.append(test_sol)  # the element in the list is of array type.
    
    if not valid_sols:
        print('No valid solutions: {valid_sols}')
    else:
        print(f'all valid solutions: {valid_sols}')
    
    joints_configs_distance = np.sum((valid_sols - np.array(desired_joints_configs))**2, 1) 
    best_joints_config_indice = np.argmin(joints_configs_distance)
    best_joints_config = valid_sols[best_joints_config_indice]/math.pi*180
    
    print(f'best joint config is {best_joints_config}')    
    
    joints_goals.append(best_joints_config)
    
print(f'all best joint config: {joints_goals}')
    
    
    
    
