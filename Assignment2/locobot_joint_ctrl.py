# Import system libraries
import argparse
import os
import sys
import time
import pdb

# Modify the following lines if you have problems importing the V-REP utilities
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd,'lib'))
sys.path.append(os.path.join(cwd,'utilities'))

# Import application libraries
import numpy as np
import vrep_utils as vu

# Import any other libraries you might want to use ############################
import matplotlib.pyplot as plt
###############################################################################

class ArmController:

    def __init__(self):
        # Fill out this method ##################################
        # Define any variables you may need here for feedback control
        # ...
        #########################################################
        # Do not modify the following variables
        self.convergence_window = 10
        self.threshold = [0.1,0.1,0.1,0.1,0.1,0.1,0.1]
        self.Kp = np.array([2 ,2 ,2 ,2 ,2 ,2 ,2])
        self.Kd = np.array([0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1 ,0.1])
        self.Ki = np.array([0.00001 ,0.00001 ,0.00001 ,0.00001 ,0.00001 ,0.00001 ,0.00001])
        self.last_joint_positions = np.zeros(7)
        self.last_timestamp = 0
        self.error_sum = np.zeros(7)
        self.history = {'timestamp': [],
                        'joint_feedback': [],
                        'joint_target': [],
                        'ctrl_commands': []}
        self._target_joint_positions = None

    def set_target_joint_positions(self, target_joint_positions):
        assert len(target_joint_positions) == vu.N_ARM_JOINTS, \
            'Expected target joint positions to be length {}, but it was length {} instead.'.format(len(target_joint_positions), vu.N_ARM_JOINTS)
        self._target_joint_positions = target_joint_positions

    def calculate_commands_from_feedback(self, timestamp, sensed_joint_positions):
        assert self._target_joint_positions, \
            'Expected target joint positions to be set, but it was not.'
        if timestamp - self.last_timestamp == 0:
            return np.zeros((7))
        # Fill out this method ##################################
        # Using the input joint feedback, and the known target joint positions,
        # calculate the joint commands necessary to drive the system towards
        # the target joint positions.
        self.error_sum = self.error_sum + (self._target_joint_positions - np.asarray(sensed_joint_positions))
        ctrl_commands = np.multiply(self.Kp,(self._target_joint_positions - np.asarray(sensed_joint_positions))) + np.multiply(self.Ki, self.error_sum) + \
        np.multiply(self.Kd,( - (sensed_joint_positions - self.last_joint_positions)/(timestamp - self.last_timestamp))) 
        self.last_joint_positions = np.asarray(sensed_joint_positions)
        self.last_timestamp = timestamp
        #########################################################

        # Do not modify the following variables
        # append time history
        self.history['timestamp'].append(timestamp)
        self.history['joint_feedback'].append(sensed_joint_positions)
        self.history['joint_target'].append(self._target_joint_positions)
        self.history['ctrl_commands'].append(ctrl_commands)
        return ctrl_commands

    def has_stably_converged_to_target(self):
        # Fill out this method ##################################
        has_stably_converged_to_target = False
        target_vector = np.tile(self._target_joint_positions,[self.convergence_window,1])
        if len(self.history['joint_feedback']) < self.convergence_window:
        	return False
        if all(np.linalg.norm(target_vector - np.asarray(self.history['joint_feedback'][-self.convergence_window:]), axis = 0) < self.threshold):
            has_stably_converged_to_target = True
            time.sleep(1)
            self.error_sum = np.zeros(7)
        print(np.linalg.norm(target_vector - np.asarray(self.history['joint_feedback'][-self.convergence_window:]))," ")
        # ...
        #########################################################
        return has_stably_converged_to_target

    def control(self, clientID, target):
        self.set_target_joint_positions(target)

        steady_state_reached = False
        while not steady_state_reached:

            timestamp = vu.get_sim_time_seconds(clientID)
            print('Simulation time: {} sec'.format(timestamp))

            # Get current joint positions
            sensed_joint_positions = vu.get_arm_joint_positions(clientID)

            # Calculate commands
            commands = self.calculate_commands_from_feedback(timestamp, sensed_joint_positions)
            # Send commands to V-REP
            vu.set_arm_joint_target_velocities(clientID, commands)

            # Print current joint positions (comment out if you'd like)
            print(sensed_joint_positions)
            vu.step_sim(clientID, 1)

            # Determine if we've met the condition to move on to the next point
            steady_state_reached = self.has_stably_converged_to_target()

def main(args):
    # Connect to V-REP
    print ('Connecting to V-REP...')
    clientID = vu.connect_to_vrep()
    print ('Connected.')

    # Reset simulation in case something was running
    vu.reset_sim(clientID)
    
    # Initial control inputs are zero
    vu.set_arm_joint_target_velocities(clientID, np.zeros(vu.N_ARM_JOINTS))

    # Despite the name, this sets the maximum allowable joint force
    vu.set_arm_joint_forces(clientID, 50.*np.ones(vu.N_ARM_JOINTS))

    # One step to process the above settings
    vu.step_sim(clientID)

    # Joint targets. Specify in radians for revolute joints and meters for prismatic joints.
    # The order of the targets are as follows:
    #   joint_1 / revolute  / arm_base_link <- shoulder_link
    #   joint_2 / revolute  / shoulder_link <- elbow_link
    #   joint_3 / revolute  / elbow_link    <- forearm_link
    #   joint_4 / revolute  / forearm_link  <- wrist_link
    #   joint_5 / revolute  / wrist_link    <- gripper_link
    #   joint_6 / prismatic / gripper_link  <- finger_r
    #   joint_7 / prismatic / gripper_link  <- finger_l
    joint_targets = [[  0.,
                        0.,
                        0.,
                        0.,
                        0.,
                      - 0.00,
                        0.00]]

    # Instantiate controller
    controller = ArmController()

    # Iterate through target joint positions
    for target in joint_targets:
        # Set new target position
        controller.set_target_joint_positions(target)

        steady_state_reached = False
        while not steady_state_reached:

            timestamp = vu.get_sim_time_seconds(clientID)
            print('Simulation time: {} sec'.format(timestamp))

            # Get current joint positions
            sensed_joint_positions = vu.get_arm_joint_positions(clientID)

            # Calculate commands
            commands = controller.calculate_commands_from_feedback(timestamp, sensed_joint_positions)

            # Send commands to V-REP
            vu.set_arm_joint_target_velocities(clientID, commands)

            # Print current joint positions (comment out if you'd like)
            print(sensed_joint_positions)
            vu.step_sim(clientID, 1)

            # Determine if we've met the condition to move on to the next point
            steady_state_reached = controller.has_stably_converged_to_target()

    vu.stop_sim(clientID)

    titles = ['Joint_1','Joint_2','Joint_3','Joint_4','Joint_5','Joint_6','Joint_7']
    for i in range(7):
    	plt.figure(i)
    	#plt.subplot(4,2,i+1)
    	plt.plot(np.asarray(controller.history['timestamp']),np.asarray(controller.history['joint_feedback'])[:,i])
    	plt.plot(np.asarray(controller.history['timestamp']),np.asarray(controller.history['joint_target'])[:,i],'r--')
    	if(i < 5):
    		plt.ylabel('Joint Angle')
    	else:
    		plt.ylabel('Joint Position')
    	plt.xlabel('Time')
    	plt.title(titles[i])
    plt.show()

    # Post simulation cleanup -- save results to a pickle, plot time histories, etc #####
    # Fill this out here (optional) or in your own script 
    # If you use a separate script, don't forget to include it in the deliverables
    # ...
    #####################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
    #print(type(sensed))
    #print(timestamp)
    #print(type(sensed_joint_positions))
    #error = _target_joint_positions - sensed_joint_positions
