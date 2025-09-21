#!/usr/bin/env python3

# This assignment lets you both define a strategy for picking the next point to explore and determine how you
#  want to chop up a full path into way points. You'll need path_planning.py as well (for calculating the paths)
#
# Note that there isn't a "right" answer for either of these. This is (mostly) a light-weight way to check
#  your code for obvious problems before trying it in ROS. It's set up to make it easy to download a map and
#  try some robot starting/ending points
#
# Given to you:
#   Image handling
#   plotting
#   Some structure for keeping/changing waypoints and converting to/from the map to the robot's coordinate space
#
# Slides

# The ever-present numpy
import numpy as np

# Your path planning code
import path_planning as path_planning
# Our priority queue
import heapq

# Using imageio to read in the image
import imageio


# -------------- Showing start and end and path ---------------
def plot_with_explore_points(im_threshhold, zoom=1.0, robot_loc=None, explore_points=None, best_pt=None):
    """Show the map plus, optionally, the robot location and points marked as ones to explore/use as end-points
    @param im - the image of the SLAM map
    @param im_threshhold - the image of the SLAM map
    @param robot_loc - the location of the robot in pixel coordinates
    @param best_pt - The best explore point (tuple, i,j)
    @param explore_points - the proposed places to explore, as a list"""

    # Putting this in here to avoid messing up ROS
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[0].set_title("original image")
    axs[1].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[1].set_title("threshold image")
    """
    # Used to double check that the is_xxx routines work correctly
    for i in range(0, im_threshhold.shape[1]-1, 10):
        for j in range(0, im_threshhold.shape[0]-1, 2):
            if is_reachable(im_thresh, (i, j)):
                axs[1].plot(i, j, '.b')
    """

    # Show original and thresholded image
    if explore_points is not None:
        for p in explore_points:
            axs[1].plot(p[0], p[1], '.b', markersize=2)

    for i in range(0, 2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if best_pt is not None:
            axs[i].plot(best_pt[0], best_pt[1], '*y', markersize=10)
        axs[i].axis('equal')

    for i in range(0, 2):
        # Implements a zoom - set zoom to 1.0 if no zoom
        width = im_threshhold.shape[1]
        height = im_threshhold.shape[0]

        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)


# -------------- For converting to the map and back ---------------
def convert_pix_to_x_y(im_size, pix, size_pix):
    """Convert a pixel location [0..W-1, 0..H-1] to a map location (see slides)
    Note: Checks if pix is valid (in map)
    @param im_size - width, height of image
    @param pix - tuple with i, j in [0..W-1, 0..H-1]
    @param size_pix - size of pixel in meters
    @return x,y """
    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Pixel {pix} not in image, image size {im_size}")

    return [size_pix * pix[i] / im_size[1-i] for i in range(0, 2)]


def convert_x_y_to_pix(im_size, x_y, size_pix):
    """Convert a map location to a pixel location [0..W-1, 0..H-1] in the image/map
    Note: Checks if x_y is valid (in map)
    @param im_size - width, height of image
    @param x_y - tuple with x,y in meters
    @param size_pix - size of pixel in meters
    @return i, j (integers) """
    pix = [int(x_y[i] * im_size[1-i] / size_pix) for i in range(0, 2)]

    if not (0 <= pix[0] <= im_size[1]) or not (0 <= pix[1] <= im_size[0]):
        raise ValueError(f"Loc {x_y} not in image, image size {im_size}")
    return pix


def is_underexplored(im, pix):
    """ Is the pixel reachable, i.e., has a neighbor that is free?
    Used for
    @param im - the image
    @param pix - the pixel i,j"""

    #Note, this has a potential edgecase failure at the edges of the image, where it 'wraps around' and could give an invalid result.
    #For now I will ignore this problem...

    #Hardcoding these to avoid for loops ig
    values_to_check = np.array([im[pix[1]-1,pix[0]-1],im[pix[1]-1,pix[0]],im[pix[1]-1,pix[0]+1],im[pix[1],pix[0]-1],
                       im[pix[1],pix[0]+1],im[pix[1]+1,pix[0]-1],im[pix[1]+1,pix[0]],im[pix[1]+1,pix[0]+1]])

    #Return true if any are true
    return np.any(values_to_check == 128)



def find_all_possible_goals(im):
    """ Find all of the places where you have a pixel that is unseen next to a pixel that is free
    It is probably easier to do this, THEN cull it down to some reasonable places to try
    This is because of noise in the map - there may be some isolated pixels
    @param im - thresholded image
    @return dictionary or list or binary image of possible pixels"""

    #Setup a list we can append points to
    possible_goals = []

    #First, Find all the image locations marked as open
    open_locs = np.argwhere(im == 255)

    #Then, check each of these for locations with unexplored neighbors.
    for i in range(1,len(open_locs)):

        point = open_locs[i]

        #check for and ignore points that are on the absolute edge of the map, 
        #if we really need to go there we can assume the map will expand/resize
        if point[0]+1 == im.shape[0] or point[1]+1 == im.shape[1]:
            continue

        pix = [point[1],point[0]]

        #check if the point has unexplored neighbors and append it to the list if so
        if is_underexplored(im,pix):
            possible_goals.append(pix)

    return possible_goals

    #The list we want is the combination of these two (unexplored AND free neighbors)
    #I feel like there is a way to use np.argwhere to get a list of just these points. 
    #The hard point is getting an image which highlights just the points with free neighbors. A problem for the future.


def find_best_point(im, possible_points, robot_loc):
    """ Pick one of the unseen points to go to
    @param im - thresholded image
    @param possible_points - possible points to chose from
    @param robot_loc - location of the robot (in case you want to factor that in)
    """

    #Becuase this is written as a function and not part of a class, any blacklisting/filtering of points will need to be done elsewhere
    #e.g. if we need to prevent the same unreachable point from being selected over and over, this will need to be handled externally.
    
    #Biggest unexplored area feels like the optimal choice for exploration, but seems hard to implement
    #Closest to the robot seems good for minimizing driving time/energy, but means it may focus on unimportant details over the bigger map...
    #Some external filtering will be required in the future to avoid this pitfall.

    #having a minimum distance lets us tune this to not pick a point basically at the robot
    #presumably, if there is an unexplored point directly at the robot, we can't get there or see it
    min_dist = 50

    #assume an absurdly large distance to start
    closest_distance = 1e30
    closest_pix = []
    #just going to iterate over all possible points provided to find the closest one
    #surely there is a more efficient way to do this but it's not worth thinking about
    for i,pix in enumerate(possible_points):
        #calculate the hypotenuse
        dist = np.sqrt(((pix[0] - robot_loc[0])**2) + ((pix[1] - robot_loc[1])**2))
        #continually update to find the closest pixel
        if min_dist < dist < closest_distance:
            closest_distance = dist
            closest_pix = pix
    
    #return the best (closest) pixel:
    #convert to tuple so it is immutable for feeding into path planning
    #probably kinda hacky
    return tuple(closest_pix)


def find_waypoints(im, path):
    """ Place waypoints along the path
    @param im - the thresholded image
    @param path - the initial path
    @ return - a new path"""

    #pick waypoints at 'corners'
    #Initial idea: To detect a corner (change in direction) in the path, do a rough second derivative test,
    #the idea here is that the path 'accelerates' only at corners/changes in direction
    #Problem - path aliasing

    #Better solution: - Create two vectors leading up to and away from each point
    #Use the dot product definition to estimate the angle
    
    start_point = path[0]
    end_point = path[-1]

    waypoints = []
    waypoints.append(start_point)

    for i in range(4,len(path)-4):
        #Vectors to compare
        v_prev = np.array(path[i]) - np.array(path[i-4])
        v_next = np.array(path[i+4]) - np.array(path[i])

        #Dot product definition
        v_angle = np.arccos(np.dot(v_prev,v_next)/(np.linalg.norm(v_prev)*np.linalg.norm(v_next)))
        
        #Also check the distance between this and the previous sampled waypoint
        #This prevents us from having high point density at corners.
        prev_dist = np.linalg.norm(np.array(waypoints[-1])-np.array(path[i]))

        #lets give a 0.1 rad margin for angle error, and a min 10 pixel spacing:
        epsilon = 0.3
        min_dist = 5

        if v_angle > epsilon and prev_dist > min_dist:
            waypoints.append(path[i])

        #Central differencing for 2nd derivative
        #dd_dxdx = path[i+1][0]-2*path[i][0]+path[i-1][0]
        #dd_dydy = path[i+1][1]-2*path[i][1]+path[i-1][1]

        #check if 2nd derivative == 0
        #The points in path should be integers by default, so we can use == instead of .isclose():
        #if ((dd_dxdx == 0) and (dd_dydy == 0)):
        #    pass
        #else:
        #    waypoints.append(path[i])
    
    #append the last waypoint:
    waypoints.append(end_point)

    #return the list of waypoints:
    return waypoints
         

    

if __name__ == '__main__':
    # Doing this here because it is a different yaml than JN
    import yaml_1 as yaml

    im, im_thresh = path_planning.open_image("map.pgm")

    robot_start_loc = (1940, 1953)

    all_unseen = find_all_possible_goals(im_thresh)
    best_unseen = find_best_point(im_thresh, all_unseen, robot_loc=robot_start_loc)

    plot_with_explore_points(im_thresh, zoom=0.1, robot_loc=robot_start_loc, explore_points=all_unseen, best_pt=best_unseen)

    path = path_planning.dijkstra(im_thresh, robot_start_loc, best_unseen)
    waypoints = find_waypoints(im_thresh, path)
    path_planning.plot_with_path(im, im_thresh, zoom=0.1, robot_loc=robot_start_loc, goal_loc=best_unseen, path=waypoints)

    # Depending on if your mac, windows, linux, and if interactive is true, you may need to call this to get the plt
    # windows to show
    # plt.show()

    print("Done")
