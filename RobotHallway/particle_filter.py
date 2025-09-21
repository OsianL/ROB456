#!/usr/bin/env python3

# This is just the code you need to do the particle filter for the door localization

import numpy as np

from world_ground_truth import WorldGroundTruth
from robot_sensors import RobotSensors
from robot_ground_truth import RobotGroundTruth


# State estimation using a particle filter
#   Stores the belief about the robot's location as a set of sample points over the state
class ParticleFilter:
    def __init__(self):

        # Probability representation
        #   Don't change these variable names (used for automatic grading)
        self.particles = []
        self.weights = []

        # Note that in the GUI version, this will be called with the desired number of samples to save
        self.reset_particles()

    def reset_particles(self, n_samples=1000):
        """ Initialize particle filter with uniform samples and weights
        @param n_samples - the number of state samples to keep """

        # TODO
        #create n_samples of the state space, uniformly distributed
        self.particles = np.random.uniform(size=n_samples)
        
        #create n_samples of uniform weights
        self.weights = np.ones(n_samples)/n_samples

    def update_particles_move_continuous(self, robot_ground_truth, amount):
        """ Update state estimation based on sensor reading
        Lec 4.1 particle filters
        Slide https://docs.google.com/presentation/d/1yddr6QwnUNHfW4GqLkC5Ds6tk8ezO56-WI2-B4ninWU/edit#slide=id.p11
        The move - lines 3-4 (sampling x_t^m
        @param robot_ground_truth
        @param dist_reading - distance reading returned by sensor"""

        #First, get the movement std dev from robot ground truth

        sigma = robot_ground_truth.move_probabilities["move_continuous"]["sigma"]

        #Create a p_new array to store moved particles in:
        p_new = np.zeros(np.shape(self.particles))

        #For each particle, move it by the given amount PLUS some noise
        #If you don't add noise, you will quickly have all of the particles at the same location..
        #If it runs into a wall, offset it from the wall by a random amount

        for i,p in enumerate(self.particles):
            
            #Calculate the noisy, moved, particle position
            p_new[i] = p + amount + np.random.normal(0,sigma)
        
            #Check if it's outside the upper bound:
            if (p_new[i] > 1):
                #if it is, have the robot bounce off a random amount
                p_new[i] = 1 - abs(np.random.normal(0,sigma))
            #Similiarly, check if it's outside the lower bound:
            if (p_new[i] < 0):
                #same as above
                p_new[i] = 0 + abs(np.random.normal(0,sigma))
            
        #Finally, update self.particles
        self.particles = p_new

    def calculate_weights_door_sensor_reading(self, world_ground_truth, robot_sensor, sensor_reading):
        """ Update your weights based on the sensor reading being true (door) or false (no door)
        Lec 4.1 particle filters
        Slide https://docs.google.com/presentation/d/1yddr6QwnUNHfW4GqLkC5Ds6tk8ezO56-WI2-B4ninWU/edit#slide=id.p11
        The weight calculation - line 5
        @param world_ground_truth - has where the doors actually are
        @param robot_sensor - has the robot sensor probabilities
        @param sensor_reading - the actual sensor reading - either True or False
        """

        #Prep New Weights
        new_weights = np.zeros(np.shape(self.weights))

        #  You'll need a for loop to loop over the particles
        #  For each particle, calculate an importance weight using p(y|x) p(x) (the numerator of the Bayes' sensor update)
        #     p(x) is the probability of being at the point x for this sample (so... what is this value?)
        #     p(y|x) is the probability of the sensor returning T/F given the location x
        #     The location of each particle... is the value stored in the particle.
        # You might find enumerate useful
        for i, p in enumerate(self.particles):
            #First check if the current particle is in front of a door in the map
            particle_in_front_of_door = world_ground_truth.is_location_in_front_of_door(p)
            #Then, run through the double if statement:
            if particle_in_front_of_door:
                if sensor_reading:
                    new_weights[i] = robot_sensor.door_dict["door"]["see_door_if_door"] * self.weights[i]
                else:
                    new_weights[i] = robot_sensor.door_dict["door"]["see_no_door_if_door"] * self.weights[i]
            else:
                if sensor_reading:
                    new_weights[i] = robot_sensor.door_dict["no_door"]["see_door_if_no_door"] * self.weights[i]
                else:
                    new_weights[i] = robot_sensor.door_dict["no_door"]["see_no_door_if_no_door"] * self.weights[i]

        #normalize and update the weights
        self.weights = new_weights/sum(new_weights)


    def calculate_weights_distance_wall(self, robot_sensors, dist_reading):
        """ Calculate weights based on distance reading
        Lec 4.1 particle filters
        Slide https://docs.google.com/presentation/d/1yddr6QwnUNHfW4GqLkC5Ds6tk8ezO56-WI2-B4ninWU/edit#slide=id.p11
        The weight calculation - line 5
        @param robot_sensors - for mu/sigma of wall sensor
        @param dist_reading - distance reading returned by sensor"""

        # TODO
        #  See calculate_weights above - only this time, set the weight for the particle based on how likely it
        #   was that the distance sensor was correct, given the location in the particle.

        #Prep New Weights:
        new_weights = np.zeros(np.shape(self.weights))

        #get sigma from robot_sensors
        sigma = robot_sensors.distance_dict["sigma"]

        #gaussian function to get the position weights
        # Yes, you can put function definitions in functions.
        def gaussian(x, mu, sigma):
            """Gaussian with given mu, sigma
            @param x - the input x value
            @param mu - the mu
            @param sigma - the standard deviation
            @return y = gauss(x) """
            return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2))

        for i,p in enumerate(self.particles):

            #Assuming the distance sensor has roughly gaussian noise centered on the reading and according to it's std dev
            #Evaluate the height of the gaussian at each particle
            new_weights[i] = gaussian(p,dist_reading,sigma) * self.weights[i]

        #update the weights and normalize
        self.weights = new_weights/sum(new_weights)
        

    def resample_particles(self):
        """ Importance sampling - take the current set of particles and weights and make a new set
        Lec 4.1 particle filters
        Slide https://docs.google.com/presentation/d/1yddr6QwnUNHfW4GqLkC5Ds6tk8ezO56-WI2-B4ninWU/edit#slide=id.p11
        The re-sampling (lines 8-10)
        Also see Lec 4.1 importance sampling
        https://docs.google.com/presentation/d/1E7mYA-3YoRt7FepwB0rN9vEoqSbHRey-AnlfVBaGRD4/edit#slide=id.p6
        """

        #Part 0: Prep an array to store the new list of particles in
        new_particles = np.zeros(np.shape(self.particles))
        n_samples = len(self.particles)

        #Part 1: make a new numpy array that is a running sum of the weights (normalized)
        # This is to speed up the computation
        # Weights have been normalized in the previous functions
        sum_weights = np.zeros(np.shape(self.weights))

        for i in range(1,len(self.weights)):
            sum_weights[i] = sum_weights[i-1] + self.weights[i-1]


        #Part 2: for n_samples (current number of particles) grab one of the particles based on the weights
        # Like discrete_sampling, only with n_particles
        # Generate a value between 0 and 1, use that to pick one of the particles
        # Put that particle in the new list
        # Note that np.where can be used to substantially speed up finding which particle

        for i in range(0,n_samples):

            #Get a random sample
            zero_to_one = np.random.uniform()
            
            #Check it against the running sum weights, finding the corresponding index
            idx = np.max(np.where(zero_to_one > sum_weights))

            #Get the particle corresponding to that index and put it in the array of new particles
            new_particles[i] = self.particles[idx]

        #Part 3: Set the new particles and reset the weights to uniform
        self.particles = new_particles
        self.weights = np.ones(np.shape(self.particles))/n_samples
        

    def one_full_update_door(self, world_ground_truth, robot_ground_truth, robot_sensor, u: float, z: bool):
        """This is the full update loop that takes in one action, followed by a door sensor reading
        Lec 4_1 Particle filter
        Slides: https://docs.google.com/presentation/d/1yddr6QwnUNHfW4GqLkC5Ds6tk8ezO56-WI2-B4ninWU/edit#slide=id.g16c435bbb81_1_0
        Assumes the robot has been moved by the amount u, then a door sensor reading was taken
        ONLY door sensor

        @param world_ground_truth - has where the doors actually are
        @param robot_sensor - has the robot sensor probabilities
        @param robot_ground_truth - robot location, has the probabilities for actually moving left if move_left called
        @param u will be the amount moved
        @param z will be one of True or False (door y/n)
        """
        #  Step 1 Move the particles (with noise added)
        self.update_particles_move_continuous(robot_ground_truth,u)

        #  Step 2 Calculate the weights using the door sensor return value
        self.calculate_weights_door_sensor_reading(world_ground_truth,robot_sensor,z)

        #  Step 3 Resample/importance weight
        self.resample_particles()


    def one_full_update_distance(self, robot_ground_truth, robot_sensor, u: float, z: float):
        """This is the full update loop that takes in one action, followed by a door sensor reading
        Lec 4_1 Particle filter
        Slides: https://docs.google.com/presentation/d/1yddr6QwnUNHfW4GqLkC5Ds6tk8ezO56-WI2-B4ninWU/edit#slide=id.g16c435bbb81_1_0
        Assumes the robot has been moved by the amount u, then a wall distance reading was taken
        ONLY door sensor

        @param robot_sensor - has the robot sensor probabilities
        @param robot_ground_truth - robot location, has the probabilities for actually moving left if move_left called
        @param u will be the amount moved
        @param z will be the distance from the sensor
        """
        #  Step 1 Move the particles (with noise added)
        self.update_particles_move_continuous(robot_ground_truth,u)

        #  Step 2 Calculate the weights using the distance sensor return value
        self.calculate_weights_distance_wall(robot_sensor,z)

        #  Step 3 Resample/importance weight
        self.resample_particles()

    def plot_particles_with_weights(self, axs, world_ground_truth, robot_ground_truth):
        """Plot the particles (scaled by weights) and the doors and where the robot actually is
        @param axs - window to draw in
        @param world_ground_truth - for the doors
        @param robot_ground_truth - for the robot's location"""

        # Plot "walls"
        height = 0.75
        axs.plot([0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, height, height, 0.0], '-k')
        # Plot "doors"
        door_width = 0.95 * world_ground_truth.door_width / 2.0
        for d in world_ground_truth.doors:
            axs.plot([d - door_width, d - door_width, d + door_width, d + door_width], [0.0, 0.5 * height, 0.5 * height, 0.0], '-r')

        min_ws = np.min(self.weights)
        max_ws = np.max(self.weights)
        if np.isclose(max_ws, min_ws):
            max_ws = min_ws + 0.01

        for w, p in zip(self.weights, self.particles):
            h = 0.2 * (w - min_ws) / (max_ws - min_ws) + 0.05
            axs.plot([p, p], [0.01, 0.01 + h], '-g')

        # Robot
        axs. plot(robot_ground_truth.robot_loc, 0.05, 'xb', markersize=10)



def convert_histogram(pf, n_bins):
    """ Convert the particle filter to an (approximate) pmf in order to compare results
    @param pf - the particle filter
    @param n_bins - number of bins
    @returns a numpy array with (normalized) probabilities"""

    bins = np.zeros(n_bins)
    for p in pf.particles:
        bin_index = int(np.floor(p * n_bins))
        try:
            bins[bin_index] += 1.0
        except IndexError:
            if p < 0.0 or p > 1.0:
                raise ValueError(f"Convert histogram: particle location not in zero to 1 {p}")
            bins[-1] += 1.0

    bins /= np.sum(bins)
    return bins




def test_particle_filter_syntax(b_print=True):
    """ Test the sequence of calls for syntax and basic errors
    @param b_print - do print statements, yes/no"""

    # Read in some move sequences and compare your result to the correct answer
    import json
    with open("Data/check_particle_filter.json", "r") as f:
        answers = json.load(f)

    if b_print:
        print("Testing particle filter (syntax)")

    particle_filter = ParticleFilter()
    world_ground_truth = WorldGroundTruth()
    robot_ground_truth = RobotGroundTruth()
    robot_sensor = RobotSensors()

    # Generate some move sequences and compare to the correct answer
    n_doors = answers["n_doors"]
    n_bins = answers["n_bins"]

    seed = 3
    np.random.seed(seed)
    world_ground_truth.random_door_placement(n_doors, n_bins)

    # Set mu/sigmas
    robot_ground_truth.set_move_continuous_probabilities(answers["move_error"]["sigma"])
    robot_sensor.set_distance_wall_sensor_probabilities(answers["sensor_noise"]["sigma"])

    # This SHOULD insure that you get the same answer as the solutions, provided you're only calling uniform within
    #  the door sensor reading, one call to random.normal() for the move, and one call to uniform for each particle
    #  in the importance sampling*
    np.random.seed(3)

    # Try different sequences
    for seq in answers["results"]:
        # Reset to uniform
        particle_filter.reset_particles()
        robot_ground_truth.reset_location()

        for i, s in enumerate(seq["seq"]):
            if s == "Door":
                saw_door = robot_sensor.query_door(robot_ground_truth, world_ground_truth)
                particle_filter.calculate_weights_door_sensor_reading(world_ground_truth, robot_sensor, saw_door)
                particle_filter.resample_particles()
                if saw_door != seq["sensor_reading"][i]:
                    print("Warning: expected {seq['sensor_reading'][i]} got {saw_door}")
            elif s == "Dist":
                dist = robot_sensor.query_distance_to_wall(robot_ground_truth)
                particle_filter.calculate_weights_distance_wall(robot_sensor, dist)
                particle_filter.resample_particles()
                if not np.isclose(dist, seq["sensor_reading"][i]):
                    print("Warning: expected {seq['sensor_reading'][i]} got {dist}")
            elif s == "Move":
                actual_move = robot_ground_truth.move_continuous(seq["move_amount"][i])
                particle_filter.update_particles_move_continuous(robot_ground_truth, seq["move_amount"][i])
            elif s == "Move_dist":
                actual_move = robot_ground_truth.move_continuous(seq["move_amount"][i])
                dist = robot_sensor.query_distance_to_wall(robot_ground_truth)
                particle_filter.one_full_update_distance(robot_ground_truth, robot_sensor, u=seq["move_amount"][i], z=dist)
            elif s == "Move_door":
                actual_move = robot_ground_truth.move_continuous(seq["move_amount"][i])
                saw_door = robot_sensor.query_door(robot_ground_truth, world_ground_truth)
                particle_filter.one_full_update_door(world_ground_truth, robot_ground_truth, robot_sensor, u=seq["move_amount"][i], z=saw_door)
            else:
                raise ValueError(f"Should be one of Move, Sensor, or Both, got {s}")

        h = convert_histogram(particle_filter, n_bins)
        h_expected = np.array(seq["histogram"])
        res = np.isclose(h, h_expected, 0.1)
        if b_print:
            print(f"Should be approximately equal, seq: {seq['seq']}")
            print(f"{res}")
            print(f"Your h: {h}")
            print(f"Approximate h: {h_expected}\n")

    if b_print:
        print("Passed syntax check")
    return True


if __name__ == '__main__':
    b_print = True

    # Syntax checks
    n_doors_syntax = 2
    n_bins_syntax = 10
    n_samples_syntax = 100
    world_ground_truth_syntax = WorldGroundTruth()
    world_ground_truth_syntax.random_door_placement(n_doors_syntax, n_bins_syntax)
    robot_ground_truth_syntax = RobotGroundTruth()
    robot_sensor_syntax = RobotSensors()
    particle_filter_syntax = ParticleFilter()

    # Syntax check 1, reset probabilities
    particle_filter_syntax.reset_particles(n_samples_syntax)

    # Syntax check 2, update move
    particle_filter_syntax.update_particles_move_continuous(robot_ground_truth_syntax, 0.1)

    # Syntax checks 3 and 4 - the two different sensor readings
    particle_filter_syntax.calculate_weights_door_sensor_reading(world_ground_truth_syntax, robot_sensor_syntax, True)
    if np.isclose(np.max(particle_filter_syntax.weights), np.min(particle_filter_syntax.weights)):
        print(f"Possible error: The weights should not all be the same")

    particle_filter_syntax.reset_particles(n_samples_syntax)
    particle_filter_syntax.calculate_weights_distance_wall(robot_sensor_syntax, 0.1)
    if np.isclose(np.max(particle_filter_syntax.weights), np.min(particle_filter_syntax.weights)):
        print(f"Possible error: The weights should not all be the same")

    # Syntax check 5 - importance sampling
    particle_filter_syntax.resample_particles()
    if not np.isclose(np.max(particle_filter_syntax.weights), np.min(particle_filter_syntax.weights)):
        print(f"Possible error: The weights should be set back to all the same")
    if np.unique(particle_filter_syntax.particles, return_counts=True) == n_samples_syntax:
        print(f"Possible error: There probably should be duplicate particles {np.unique(particle_filter_syntax.particles, return_counts=True)} {n_samples_syntax}")

    # Syntax checks 6 and 7 - the two full updates
    particle_filter_syntax.one_full_update_door(world_ground_truth_syntax, robot_ground_truth_syntax, robot_sensor_syntax, u=0.1, z=True)
    particle_filter_syntax.one_full_update_distance(robot_ground_truth_syntax, robot_sensor_syntax, u=0.1, z=0.6)


    # The syntax tests/approximate histogram tests
    test_particle_filter_syntax(b_print)

    print("Done")
