#!/usr/bin/env python3

import numpy as np
from world_ground_truth import WorldGroundTruth
from robot_ground_truth import RobotGroundTruth


# This class handles the robot's sensors
#  - In front of door, y/n
#  - Distance to wall
class RobotSensors:
    def __init__(self):

        #quick test:
        #np.random.seed(3)

        # Default sensor probabilities, stored as dictionaries (see probabilities homework)
        #   door is if the robot is in front of the door, return True/False
        #   no_door is if the robot is NOT in front of a door, return True/False
        #   distance_wall - returns a distance (with noise)

        # Bayes filter:
        # Create a dictionary that has two dictionaries in it (one for door, one for no door)
        # The actual dictionaries will be set in set_door_sensor_probabilities
        self.door_dict = {"door": {}, "no_door": {}}

        # Kalman filter:
        #  TODO: Add another dictionary for the distance to the wall sensor noise
        # Note: The actual values in the dictionaries will be set in the calls to set_* below
        # Second note: all variables should be referenced with self. or they will disappear
        self.distance_dict = {"mu": 0, "sigma": 0}

        # In the GUI version, these will be called with values from the GUI after the RobotSensors instance
        #   has been created
        # Actually SET the values for the dictionaries
        self.set_door_sensor_probabilites()
        self.set_distance_wall_sensor_probabilities()

    def set_door_sensor_probabilites(self, in_prob_see_door_if_door=0.8, in_prob_see_door_if_not_door=0.1):
        """ Set the two door probabilities.
        @param in_prob_see_door_if_door - probability of seeing a door if there is one
        @param in_prob_see_door_if_not_door - probability of seeing a door if there is NOT one
        """
        # Bayes assignment
        # Quick Value check:
        # if not ((0 <= in_prob_see_door_if_door <= 1) or (0 <= in_prob_see_door_if_not_door <= 1)):
        #     raise ValueError("Door probabilities suck")
        # Store the input values in TWO dictionaries (one for the door there, one for no door)
        #Robot is in front of the door in the real world
        self.door_dict["door"] = {"see_door_if_door": in_prob_see_door_if_door,
                                  "see_no_door_if_door": 1-in_prob_see_door_if_door}
        #Robot is not in front of the door in the real world
        self.door_dict["no_door"] = {"see_door_if_no_door": in_prob_see_door_if_not_door,
                                     "see_no_door_if_no_door": 1-in_prob_see_door_if_not_door}

    def set_distance_wall_sensor_probabilities(self, sigma=0.1):
        """ Setup the wall sensor probabilities (store them in the dictionary)
        Note: Mean is zero for this assignment
        @param sigma - sigma of noise"""

        # Kalman assignment
        #Store the mean and standard deviation
        self.distance_dict['mu'] = 0
        self.distance_dict['sigma'] = sigma

    def query_door(self, robot_gt, world_gt):
        """ Query the door sensor
        @param robot_gt - ground truth of robot location is in here
        @param world_gt - ground truth of where the doors are
        @return True (thinks open) or False (thinks closed
        """

        # Bayes assignment
        # I've handled the checking if the robot is in front of the door (y/n) part for you
        # This is the ground truth - True if robot is actually in front of the door, False otherwise
        #  Use this to determine which probabilities to use
        # is_in_front_of_door is a Boolean, world_gt has the actual doors, robot_gt has the actual robot location (in 0,1)
        is_in_front_of_door = world_gt.is_location_in_front_of_door(robot_gt.robot_loc)

        dice_roll = np.random.uniform()

        #if we are in front of a door in the real world
        if (is_in_front_of_door == True):
            #check if our dice roll lets us see the door (robot does a perception check)
            if (dice_roll < self.door_dict["door"]["see_door_if_door"]):
                return True
            else: return False
        #if we are not in front of a door in the real world
        else:
            #constituion check or smth to resist door illusion idk
            if (dice_roll < self.door_dict["no_door"]["see_no_door_if_no_door"]):
                return False
            else: return True

    def query_distance_to_wall(self, robot_gt):
        """ Return a distance reading (with correct noise) of the robot's location
        This returns the estimated robot's location by measuring the distance to the left/right walls - i.e., it
        is a direct measure of the robot's location
        @param robot_gt - ground truth of robot location is in here
        @return distance + noise """

        # Kalman assignment
        # Return the distance to the wall (with noise)
        return robot_gt.robot_loc + np.random.normal(self.distance_dict["mu"], self.distance_dict["sigma"])


def test_discrete_sensors(b_print=True):
    """ Test that the door sensor is working correctly
    @param b_print - do print statements, yes/no"""
    np.random.seed(3)

    robot_gt = RobotGroundTruth()
    world_gt = WorldGroundTruth()
    robot_sensor = RobotSensors()

    probs_see_door = (0.7, 0.2)
    robot_sensor.set_door_sensor_probabilites(probs_see_door[0], probs_see_door[1])
    n_samples = 1000

    # Test if the robot is actually in front of the door versus if it is NOT in front of the door
    #   Doing this as a for loop to avoid duplicating the counting code. p will have the probability of returning
    #   True, given the first case (robot is in front of door), versus the second case (not in front of the door)
    for loc, p in zip((world_gt.location_in_front_of_door(), world_gt.location_not_in_front_of_door()), probs_see_door):
        robot_gt.robot_loc = loc

        is_in_front_of_door_gt = world_gt.is_location_in_front_of_door(robot_gt.robot_loc)
        if b_print:
            print(f"Testing case for robot in front of door: {is_in_front_of_door_gt}")

        # Check that we get our probabilities back (mostly)
        count_returned_true = 0
        for i in range(0, n_samples):
            if robot_sensor.query_door(robot_gt=robot_gt, world_gt=world_gt):
                count_returned_true += 1

        prob_count_true = count_returned_true / n_samples
        # If the robot is in front of the door, then we get the first set of probabilities - the robot is in
        #  front of the door and we should return True 70% of the time
        # If the robot is NOT in front of the door, then we get the second set of probabilities - the robot is NOT in
        #  front of the door and we should return True 20% of the time
        if not np.isclose(prob_count_true, p, atol=0.1):
            raise ValueError(f"Probability should be close to {p:0.4f}, is {prob_count_true:0.4f}, robot loc {is_in_front_of_door_gt}")

    if b_print:
        print("Passed tests")
    return True


def test_continuous_sensor(b_print=True):
    """ Test that the distance to wall sensor is working properly
    @param b_print - do print statements, yes/no"""
    if b_print:
        print("Checking query wall with normal distribution probabilities")

    robot_gt = RobotGroundTruth()
    robot_sensor = RobotSensors()

    sigma = 0.01

    # Doing this as a for loop with pre-allocating the data (np.zeros instead of a list with an append) because
    #   we are going to place the robot randomly at each iteration, then measure the distance the sensor returns,
    #   rather than keeping the robot in one place all the time
    # This is the exactly how one would measure sensor noise in the first place, btw
    n_samples = 10000
    dist_measured = np.zeros(n_samples)
    robot_sensor.set_distance_wall_sensor_probabilities(sigma)
    for i in range(0, n_samples):
        # Put the robot in a random location
        robot_gt.place_random()

        # How much is the measurement off by?
        dist_measured[i] = robot_sensor.query_distance_to_wall(robot_gt) - robot_gt.robot_loc

    # What is the mu and sigma of the sensor noise?
    mu_query = np.mean(dist_measured)
    sigma_query = np.std(dist_measured)
    if not np.isclose(mu_query, 0.0, atol=0.01):
        raise ValueError(f"Mean should be close to 0.0, is {mu_query}")
    if not np.isclose(sigma_query, sigma, atol=0.01):
        raise ValueError(f"Mean should be close to {sigma}, is {sigma_query}")

    if b_print:
        print("Passed continuous tests")
    return True


if __name__ == '__main__':
    b_print = True

    # ----------------------- Bayes filter -----------------
    # Syntax check
    robot_gt = RobotGroundTruth()
    world_gt = WorldGroundTruth()
    robot_sensor = RobotSensors()

    probs_see_door = (0.7, 0.2)
    robot_sensor.set_door_sensor_probabilites(probs_see_door[0], probs_see_door[1])
    ret_value = robot_sensor.query_door(robot_gt, world_gt)
    if ret_value is True or ret_value is False:
        print("Passed robot sensor syntax check")

    test_discrete_sensors(b_print)

    # ----------------------- Kalman filter -----------------
    # Syntax check
    robot_gt = RobotGroundTruth()
    robot_sensor = RobotSensors()
    robot_sensor.set_distance_wall_sensor_probabilities(sigma=0.01)
    dist_with_noise = robot_sensor.query_distance_to_wall(robot_gt)
    if 0.0 < dist_with_noise < 1.0:
        print("Dist wall sensor, passed syntax test")

    test_continuous_sensor(b_print)
