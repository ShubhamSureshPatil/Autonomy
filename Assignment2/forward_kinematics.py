#This file contains all the classes and functions needed for forward kinemtics

import numpy as np
import math 

def EulerAngles(Rot):
        
        #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        #Refered the above site for learning

        test = math.sqrt(Rot[0,0] * Rot[0,0] +  Rot[1,0] * Rot[1,0])
        singular = test < 1e-6
        
        if  not singular:
                x = math.atan2(Rot[2,1] , Rot[2,2])
                y = math.atan2(-Rot[2,0], test)
                z = math.atan2(Rot[1,0], Rot[0,0])
        else :
                x = math.atan2(-Rot[1,2], Rot[1,1])
                y = math.atan2(-Rot[2,0], test)
                z = 0
        
        return np.array([x, y, z])

def createTransformationMatrix(orient,trans):
        H = np.identity(4)
        H[0:3,0:3] = createRotationMatrix(orient)
        H[0:3,3] = trans
        return H

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

class forwardKinematics:
        def __init__(self):
                self.position = np.load('links.npy')
                self.axis = np.array([[0,0,1],[0,1,0],[0,1,0],[0,1,0],[0,0,1],[0,1,0],[0,1,0]])
                arm_cuboids = np.load('arm_cuboids.npz')
                self.arm_cuboid_origin = arm_cuboids[arm_cuboids.files[0]]
                self.dimension = arm_cuboids[arm_cuboids.files[1]]


        def getJointForwardKinematics(self, joint_angles):
                temp = np.identity(4)
                H = []
                for i in range(joint_angles.shape[1]):
                        next_H = createTransformationMatrix(self.axis[i,:]*joint_angles[0,i],self.position[i,:])
                        temp = np.matmul(temp, next_H)
                        H.append(temp)
                return np.asarray(H)

        def Rotate_Cuboid(self, joint_angles):
                H = self.getJointForwardKinematics(joint_angles)
                cuboid = np.zeros((7,3,3))                     

                for i in range(H.shape[0]):
                        cuboid[i,0,:] = np.matmul(H[i,:,:],self.arm_cuboid_origin[i,:].T)[0:3]
                        cuboid[i,1,:] = EulerAngles(H[i,0:3,0:3])
                        cuboid[i,2,:] = self.dimension[i,:]

                cuboid[5,0,:] = np.matmul(H[4,:,:],self.arm_cuboid_origin[5,:].T)[0:3]
                cuboid[5,1,:] = EulerAngles(H[4,0:3,0:3])
                cuboid[5,2,:] = self.dimension[5,:]
                cuboid[6,0,:] = np.matmul(H[4,:,:],self.arm_cuboid_origin[6,:].T)[0:3]
                cuboid[6,1,:] = EulerAngles(H[4,0:3,0:3])
                cuboid[6,2,:] = self.dimension[6,:]
                return(cuboid)
 


