import argparse
import os
import sys
import time
import math

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd,'lib'))
sys.path.append(os.path.join(cwd,'utilities'))

import numpy as np
import vrep_utils as vu
import locobot_joint_ctrl
import PRM

def createRotationMatrix(orient):
        roll,pitch,yaw = orient

        R_x = np.array([[1,                      0,               0],
                        [0,         math.cos(roll), -math.sin(roll)],
                        [0,         math.sin(roll),  math.cos(roll)]]) 
                
        R_y = np.array([[math.cos(pitch),    0,      math.sin(pitch)],
                        [0,                  1,                    0],
                        [-math.sin(pitch),   0,      math.cos(pitch)]])
                
        R_z = np.array([[math.cos(yaw),    -math.sin(yaw),    0],
                        [math.sin(yaw),     math.cos(yaw),    0],
                        [0,                             0,    1]])
                                
        return np.matmul(R_z, np.matmul(R_y, R_x))

def getCollisionCuboid(clientID, var):

    if var == 'arm':    
        cuboidHandles = vu.get_arm_cuboid_handles(clientID)
    elif var == 'obstacle':
        cuboidHandles = vu.get_obstacle_cuboid_handles(clientID)
    collision_cuboids = np.zeros((len(cuboidHandles),3,3))

    for i in range(len(cuboidHandles)):
        collision_cuboids[i,0,:] = vu.get_object_position(clientID, cuboidHandles[i])
        collision_cuboids[i,1,:] = vu.get_object_orientation(clientID, cuboidHandles[i])
        min_pos, max_pos = vu.get_object_bounding_box(clientID, cuboidHandles[i])
        collision_cuboids[i,2,:] = np.asarray(max_pos) - np.asarray(min_pos) 
    return collision_cuboids

def getJoint(clientID):
    jointHandles = vu.get_arm_joint_handles(clientID)
    joints = np.zeros((len(jointHandles),2,3))

    for i in range(len(jointHandles)):
        joints[i,0,:] = vu.get_object_position(clientID, jointHandles[i])
        joints[i,1,:] = vu.get_object_orientation(clientID, jointHandles[i])
    return joints

# def Save_Joints(clientID):
#     joints = getJoint(clientID)
#     links = np.zeros((5,3))
#     links[0,:] = joints[0,0,:]
#     for i in range(1,5):
#         links[i,:] = joints[i,0] - joints[i-1,0]
#     #np.save('links.npy', links)

# def Save_Cuboids(clientID):
#     arm_cuboid = getCollisionCuboid(clientID,'arm')
#     joints = getJoint(clientID)
#     arm_cuboids = np.ones((7,4))
#     for i in range(0,5):
#         arm_cuboids[i,0:3] = arm_cuboid[i,0,:] - joints[i,0]
#     arm_cuboids[5,0:3] = arm_cuboid[5,0,:] - joints[4,0]    #The connection is between 4 and 5
#     arm_cuboids[6,0:3] = arm_cuboid[6,0,:] - joints[4,0]    #The connection is betweeb 4 and 6
#     dimension = getArmCuboidDimension(arm_cuboid[:,1,:], arm_cuboid[:,2,:]) 
#     np.savez('arm_cuboids.npz', origin = arm_cuboids, dimension = dimension)

def getArmCuboidDimension(orientation, dimension):
        volume = []
        for i in range(orientation.shape[0]):
                R = createRotationMatrix(orientation[i,:])
                volume.append(np.matmul(R.transpose(),dimension[i,:].transpose()).transpose())
        volume = np.stack(volume)
        return(volume)

if __name__ == "__main__":
    # Connect to V-REP
    print ('Connecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    vu.reset_sim(clientID)
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))
    vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))

    deg_to_rad = np.pi/180. 
    
    controller = locobot_joint_ctrl.ArmController()

    obstacles = np.load('obstacles.npy')
    prm = PRM.PRM_Map(obstacles)
    start = np.array([[-80, 0, 0, 0, 0]])
    target = np.array([[0, 60, -75, -75, 0]])
    path = prm.Planner(start, target)
    
    vu.step_sim(clientID)
    for joints in path:
        joints = (joints*deg_to_rad).tolist()
        joints.append(-0.03)
        joints.append(0.03)
        print(joints)
        controller.control(clientID, joints)
    
    # Stop the simulation
    vu.stop_sim(clientID)

    # Use the below code to get the new obstacle, not recommended; Ensure to move all the joints to neutral position before saving

    # obstacles = getCollisionCuboid(clientID,'obstacle')
    # np.save('obstacles.npy', obstacles)
    # Save_Joints(clientID)
    # Save_Cuboids(clientID)

