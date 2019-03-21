import numpy as np
import math 
import forward_kinematics as forward
from collections import defaultdict
import time
import pickle
import heapq
import question1 as q1


class Node:
    def __init__(self, state, id):
        self.state = state
        self.id = id

class Graph:
    def __init__(self):
        self.adjacent_nodes = defaultdict(dict)
        self.nodes = defaultdict(np.float)
        self.ID = 0

    def addNode(self, sample):
        node = Node(sample, self.ID)
        self.nodes[self.ID] = sample
        self.ID += 1
        return node

    def deleteNode(self, node):
        connections = self.adjacent_nodes[node.id]
        del self.adjacent_nodes[node.id]
        for c in connections:
            del self.adjacent_nodes[c][node.id]

    def addEdge(self, new, nearest_node):
        self.adjacent_nodes[int(new)][nearest_node[0]] = nearest_node[1]
        self.adjacent_nodes[nearest_node[0]][int(new)] = nearest_node[1]

    def getVertices(self):
        pass

    def getEdges(self):
        pass

def get_distance(node_A, node_B):
    dist = np.zeros((1,node_A.shape[0]))
    for i in range(node_A.shape[0]):
        angle = np.linalg.norm(node_B[i] - node_A[i])
        dist[0,i] = min(angle, 360 - angle)
    return np.linalg.norm(dist)

class PRM_Map:
    def __init__(self, obstacles_cuboids):
        self.constraint = np.tile(np.array([-179,179])[:,np.newaxis],(1,5))

        self.obstacles_cuboids = obstacles_cuboids
        ground_plane = np.array([[[0, 0, 0],[0, 0, 0],[2.0, 2.0, 0.01]]])
        self.obstacles_cuboids = np.append(self.obstacles_cuboids, ground_plane, axis=0)

        self.map = Graph()
        self.forward_kinematics = forward.forwardKinematics()

    def sampleNode(self):
        samples = np.zeros((1,5))

        for i in range(samples.shape[1]):
            samples[:,i] = np.random.randint(self.constraint[0,i],self.constraint[1,i], size=1)

        collision = self.checkPointCollision(samples)
        if collision == True:
            return None
        else: 
            return self.map.addNode(samples)

    def KNearest(self, new_node, K): 
        dist = np.zeros((len(self.map.nodes),2))

        for i in range(len(self.map.nodes)):
            dist[i,0] = i
            dist[i,1] = get_distance(new_node.state,self.map.nodes[i])

        dist = dist[dist[:,1].argsort()]

        if len(self.map.nodes) > K:
            return dist[1:K,:]
        else:
            return dist[1:,:]

    def checkPointCollision(self, sample):
        deg_to_rad = np.pi/180.0
        arm_cuboids = self.forward_kinematics.Rotate_Cuboid(deg_to_rad*sample)

        for i in range(arm_cuboids.shape[0]):
            for j in range(self.obstacles_cuboids.shape[0]):
                collision = q1.check_collision(arm_cuboids[i,:,:], self.obstacles_cuboids[j,:,:]) 
                if collision is True:
                    return True
        return False

    def Line_Collision(self, new_node, nearest_node):
        edge = False

        for i in range(0, nearest_node.shape[0]):
            delta = self.map.nodes[nearest_node[i,0]] - new_node.state
            steps = math.ceil(abs(delta).max()/7)   #considering 7 as a factor

            for j in range(0,int(steps)+1):
                sample = new_node.state + j*delta/steps

                collision = self.checkPointCollision(sample)
                if collision is True:
                    break
            if collision is False:
                self.map.addEdge(new_node.id, nearest_node[i,:])
                edge = True
        return edge  

    def CreateGraph(self, sim_time):
        K = 10

        start_time = time.time()
        while time.time() - start_time < sim_time:
            new_node = self.sampleNode()

            if new_node is not None and len(self.map.nodes) > 1:
                nearest_node = self.KNearest(new_node, K)
                self.Line_Collision(new_node, nearest_node)

        #self.SaveGraph()

    # def SaveGraph(self):
    #     with open('graph.pickle', 'wb') as outfile:
    #         pickle.dump(self.map.adjacent_nodes, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    #     with open('node.pickle', 'wb') as outfile:
    #         pickle.dump(self.map.nodes,outfile, protocol = pickle.HIGHEST_PROTOCOL)

    def loadGraph(self):
        with open('graph.pickle', 'rb') as infile:
            self.map.adjacent_nodes = pickle.load(infile)
    
        with open('node.pickle', 'rb') as infile:
            self.map.nodes = pickle.load(infile)
        self.map.ID = len(self.map.nodes)

    def Planner(self, start, target):
        K = 10
        self.loadGraph()

        start_node = self.map.addNode(start)
        nearest_node = self.KNearest(start_node, K)
        assert self.Line_Collision(start_node, nearest_node), 'No Connection with Start node; Retry'

        target_node = self.map.addNode(target)
        nearest_node = self.KNearest(target_node, K)
        assert self.Line_Collision(target_node, nearest_node), 'No Connection with Target node; Retry'

        path = self.shortest_path(start_node, target_node)
        return path

    def shortest_path(self, start_node, target_node):
        priority_list = []

        past_node = defaultdict(int)
        penalty = defaultdict(int)
        past_node[start_node.id] = None
        penalty[start_node.id] = 0
        heapq.heappush(priority_list, (get_distance(target_node.state, start_node.state), start_node.id))
        flag = 0

        while priority_list:
            current_node = heapq.heappop(priority_list) #Get the maximum value out

            if current_node[1] == target_node.id:
                flag = 1
                break

            for next_node in self.map.adjacent_nodes[current_node[1]]:
                new_cost = self.map.adjacent_nodes[current_node[1]][next_node] + penalty[current_node[1]]

                if next_node not in penalty or penalty[next_node] > new_cost:
                    penalty[next_node] = new_cost
                    priority_cost = new_cost + get_distance(self.map.nodes[next_node], target_node.state)
                    heapq.heappush(priority_list, (priority_cost, next_node))
                    past_node[next_node] = current_node[1]
        
        next_id = target_node.id
        path = [self.map.nodes[next_id]]

        while flag:
            if next_id == start_node.id:
                path = np.flipud(np.squeeze(np.stack(path)))
                return path
            next_id = past_node[next_id]
            path.append(self.map.nodes[next_id])
        return None

if __name__ == "__main__":
    obstacles = np.load('obstacles.npy')
    prm = PRM_Map(obstacles)
    prm.CreateGraph(120.0)
