import numpy as np
import math 
from sklearn.preprocessing import normalize

def createRotationMatrix(orientation):
        roll,pitch,yaw = orientation

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

class Cuboid:
    def __init__(self, cube):
        self.origin = cube[0,:]
        self.orientation = cube[1,:]
        self.dimension = cube[2,:]
        self.corners = np.zeros((8,3))
        self.axes = np.zeros((3,3))
        self.R = np.zeros((3,3))
        self.createCuboid()


    def createCuboid(self):
        self.R = createRotationMatrix(self.orientation)
        self.findCorners()
        self.findAxes()        


    def findCorners(self):
        span = np.array([[ 0.5, 0.5, 0.5],[-0.5, 0.5, 0.5], [ 0.5,-0.5, 0.5],[-0.5,-0.5, 0.5], \
                        [ 0.5, 0.5,-0.5],[-0.5, 0.5,-0.5], [ 0.5,-0.5,-0.5],[-0.5,-0.5,-0.5]])
        vertices = np.tile(self.dimension[np.newaxis,:], (8, 1))
        span = span*vertices
        orient = np.tile(self.origin[np.newaxis,:], (8,1))
        self.corners = np.matmul(self.R, np.transpose(span)).T + orient

    def findAxes(self):
        direction = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.axes = np.matmul(self.R, direction).T
        self.axes = normalize(self.axes, axis=1)
    
    def getAxes(self):
        return self.axes

    def getCorners(self):
        return self.corners

    def project(self, axis):
        return np.matmul(self.corners, axis)

def CollisionAxes(ax1, ax2):  
    axes = np.vstack((ax1, ax2))

    for i in range(ax1.shape[0]):
        for j in range(ax2.shape[0]):
            axis = normalize( np.cross(ax1[i, :], ax2[j, :]).reshape(1,-1) )
            axes = np.vstack((axes, axis))
    return axes

def checkProjection(cube1, cube2, axes):
    for axis in axes:
        proj1 = cube1.project(axis)      
        proj2 = cube2.project(axis)
        p1_min = proj1.min()
        p1_max = proj1.max()
        p2_min = proj2.min()
        p2_max = proj2.max()
    
        if (p1_max < p2_min and p1_min < p2_min) or (p2_max < p1_min and p2_min < p1_min):
                return False
    return True

def check_collision(A, B):
    box1 = Cuboid(A)
    box2 = Cuboid(B)
    ax1 = box1.getAxes()
    ax2 = box2.getAxes()
    axes = CollisionAxes(ax1, ax2)
    collision = checkProjection(box1, box2, axes)
    return collision

if __name__ == "__main__":
    reference = np.array([[0, 0, 0], [0, 0, 0], [3, 1, 2]])

    # Uncomment the test case for which you want to evaluate
    test = np.array([[ 0.0, 1.0, 0.0], [ 0.0, 0.0, 0.0], [ 0.8, 0.8, 0.8]]) 
    #test = np.array([[ 1.5,-1.5, 0.0], [ 1.0, 0.0, 1.5], [ 1.0, 3.0, 3.0]]) 
    #test = np.array([[ 0.0, 0.0,-1.0], [ 0.0, 0.0, 0.0], [ 2.0, 3.0, 1.0]]) 
    #test = np.array([[ 3.0, 0.0, 0.0], [ 0.0, 0.0, 0.0], [ 3.0, 1.0, 1.0]]) 
    #test = np.array([[-1.0, 0.0,-2.0], [ 0.5, 0.0, 0.4], [ 2.0, 0.7, 2.0]]) 
    #test = np.array([[ 1.8, 0.5, 1.5], [-0.2, 0.5, 0.0], [ 1.0, 3.0, 1.0]]) 
    #test = np.array([[ 0.0,-1.2, 0.4], [0.0,0.785,0.785],[ 1.0, 1.0, 1.0]]) 
    #test = np.array([[-0.8, 0.0,-0.5], [ 0.0, 0.0, 0.2], [ 1.0, 0.5, 0.5]]) 
             
    print('The fact that the two cuboids are in collision is ', check_collision(reference, test))
