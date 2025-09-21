#!/usr/bin/env python3


import sys
import rospy
import signal
import time

from controller import RobotController
import exploring as explore
import path_planning as pathplan

class StudentController(RobotController):
	'''
	This class allows you to set waypoints that the robot will follow.  These robots should be in the map
	coordinate frame, and will be automatially sent to the code that actually moves the robot, contained in
	StudentDriver.
	'''
	def __init__(self):
		super().__init__()

		#Setup some variables for our timeout system
		self.goal_start_time = time.time()
		self.making_progress = True
		self.goal_timeout = 90


	def distance_update(self, distance):
		'''
		This function is called every time the robot moves towards a goal.  If you want to make sure that
		the robot is making progress towards it's goal, then you might want to check that the distance to
		the goal is generally going down.  If you want to change where the robot is heading to, you can
		make a call to set_waypoints here.  This call will override the current set of waypoints, and the
		robot will start to drive towards the first waypoint in the new list.

		Parameters:
			distance:	The distance to the current goal.
		'''
		rospy.loginfo(f'Distance: {distance}')

		'''

		-use this function to check that robot is making progress to goal, implement way to send new waypoints if
		reasonable amount of progress isnt made in some timeframe (idk how to access time here tho)


		note:besides above use case i generally dont think we want to set waypoints here unless we arent getting map updates,
		there should be some way to guarantee we can find new goals even if we dont get map update?
		-when this function runs force map_update to run (if statement to make it not actually run if we are making progress?)

		'''
		
		#units are in seconds
		#there might be an error for start_travelling_time on first program loop?
		travelling_time = time.time() - self.goal_start_time

		#gives robot ~60 seconds to reach goal location
		if travelling_time > self.goal_timeout:
			self.making_progress = False
		else:
			self.making_progress = True


	def map_update(self, point, map, map_data):
		'''
		This function is called every time a new map update is available from the SLAM system.  If you want
		to change where the robot is driving, you can do it in this function.  If you generate a path for
		the robot to follow, you can pass it to the driver code using set_waypoints().  Again, this will
		override any current set of waypoints that you might have previously sent.

		Parameters:
			point:		A PointStamped containing the position of the robot, in the map coordinate frame.
			map:		An OccupancyGrid containing the current version of the map.
			map_data:	A MapMetaData containing the current map meta data.
		'''




		rospy.loginfo('Got a map update.')

		# It's possible that the position passed to this function is None.  This try-except block will deal
		# with that.  Trying to unpack the position will fail if it's None, and this will raise an exception.
		# We could also explicitly check to see if the point is None.
		try:
			# The (x, y) position of the robot can be retrieved like this.
			robot_position = (point.point.x, point.point.y)

			rospy.loginfo(f'Robot is at {robot_position} {point.header.frame_id}')
		except:
			rospy.loginfo('No odometry information')

        #deal with what to do if we dont get robot position (either dont run rest of code or use last known robot location?)

		#run explore code to find best goal (modify to be furthest away) (dont think we need pathfinding bcuz define waypoints at corners?)
		
		#Map updates are configured to always run in the launch file (even when the robot is still)
		
		#Check if we need to generate new waypoints:
		if len(self._waypoints) == 0 or self.making_progress == False:

			#Step 0:
			#Convert the map into a more useful form:
			im = np.array(map.data).reshape(map.info.height,map.info.width)
			map_dims = (map.info.width,map.info.height)
			
			#log map info:
			rospy.loginfo(f'Map Width: {map.info.width}, Height: {map.info.height}, Res: {map.info.resolution}')

			#Convert robot location to map pixel location
			robot_map_loc = explore.convert_x_y_to_pix(map_dims,robot_position,map.info.resolution)

			#Step 1: Find all possible goals
			all_goals = explore.find_all_possible_goals(im)
			#TODO: hold all possible goals in a queue
			#Compare possible goals with goals we already attempted to reach (spatially: w/i radius)

			#Step 2: Find best goal of possible goals
			best_goal = explore.find_best_point(im,all_goals,robot_map_loc)

			#Step 3: Get waypoints
			#Path plan
			path = pathplan.dijkstra(im,robot_map_loc,best_goal)
			#Get corner waypoints
			map_waypoints = explore.find_waypoints(im,path)
		
			#Step 4:
			#Convert outputs back into world units from map pixels
			waypoints = []
			for point in map_waypoints:
				waypoints.append(explore.convert_pix_to_x_y(map_dims,point,map.info.resolution))
			
			#Log the new waypoints real quick:
			rospy.loginfo(f'New Waypoints: {waypoints}')

			#Set waypoints
			self.set_waypoints(waypoints)

			#reset self.goal_start_time and self.making_progress
			self.goal_start_time = time.time()
			self.making_progress = True

		#defines what remaining counts as completing the mapping 
		#completion_percent = 0.01

		#possible goals defined in explore code (all goals we might want to send robot to)
		#if possible_goals_amount < completion_percent*explored_points_amount:
			#rospy.loginfo('Mapping complete, program ended.')
			#break #dont know how to end program
	


if __name__ == '__main__':
	# Initialize the node.
	rospy.init_node('student_controller', argv=sys.argv)



	# Start the controller.
	controller = StudentController()

	# This will move the robot to a set of fixed waypoints.  You should not do this, since you don't know
	# if you can get to all of these points without building a map first.  This is just to demonstrate how
	# to call the function, and make the robot move as an example.
	# controller.set_waypoints(((-4, -3), (-4, 0), (5, 0)))

	# Once you call this function, control is given over to the controller, and the robot will start to
	# move.  This function will never return, so any code below it in the file will not be executed.
	controller.send_points()
