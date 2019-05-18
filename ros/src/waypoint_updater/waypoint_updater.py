#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
import numpy as np
import math


'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

'''

# Implementation skeleton is based on Udacity 6. Waypoint Updater Partial Walkthrough

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL=0.5

class WaypointUpdater(object):

    def __init__(self):
        # Initializing the node
        rospy.init_node('waypoint_updater')

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Initializing auxiliary variables
        self.base_lane = None
        self.stopline_wp_idx = -1
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None

        self.stop_temp = False

        rate=rospy.Rate(10)
        # The rate is decreased to enhance the performance of the simulator
        # while not rospy.core.is_shutdown():
        while not rospy.is_shutdown():  
            #rospy.logerr(">> Active - stopline  :%s",self.stop_temp)
            if self.pose and self.base_lane and self.waypoints_2d:
                self.publish_waypoints()
            #rospy.rostime.wallsleep(1)
            rate.sleep()
    def get_closest_waypoint_id(self):
        # Retrieving x,y coordinates from the message
        x =  self.pose.pose.position.x
        y =  self.pose.pose.position.y

       
        if self.waypoints_2d:
            # Capturing the closest point with the function closest() 
            closest_point = self.closest(self.waypoints_2d,[x,y])
            # Using the hyperplane approach presented in Udacity 6. Waypoint Updater Partial Walkthrough
            # Verifying if point is ahead of the vehicle
            # Equation for hyperplane through closest coords
            closest_xy = np.array(self.waypoints_2d[closest_point])
            prev_xy = np.array(self.waypoints_2d[closest_point-1])
            current_xy = np.array([x,y])
            dotproduct = np.dot(closest_xy-prev_xy,current_xy-closest_xy)
            if dotproduct > 0:
                closest_point = (closest_point+1)%len(self.waypoints_2d)
            return closest_point

    def publish_waypoints(self):
        final_lane=self.generate_lane()
        self.final_waypoints_pub.publish(final_lane)

    def generate_lane(self):
        lane=Lane()
        # Capturing the closest waypoint
        closest_idx=self.get_closest_waypoint_id()
        #rospy.logerr(">> Closest idx :%s",closest_idx)


        farthest_idx=closest_idx+LOOKAHEAD_WPS
        base_waypoints=self.base_lane.waypoints[closest_idx:farthest_idx]

        if self.stop_temp == False:
            lane.waypoints=self.initial_stop(base_waypoints,closest_idx)


        #elif base_waypoints and (self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx) or self.stopline_wp_idx < (closest_idx+1)):
        elif base_waypoints and (self.stopline_wp_idx == -1  or (self.stopline_wp_idx >= farthest_idx)): 
            lane.waypoints=base_waypoints
            #lane.waypoints = self.extrapolate_acceleration(closest_idx, farthest_idx)
        elif base_waypoints and (self.stopline_wp_idx == -2):
            lane.waypoints=base_waypoints
            lane.waypoints=self.slowdown(base_waypoints,closest_idx)
        elif base_waypoints and (self.stopline_wp_idx == -3):
            #lane.waypoints=base_waypoints
            lane.waypoints=self.faststart(base_waypoints,closest_idx)
        else:
            lane.waypoints=self.decelerate_waypoints(base_waypoints,closest_idx)
        return lane
    def decelerate_waypoints(self,waypoints,closest_idx):
        temp=[]
        for i,wp in enumerate(waypoints):
            p=Waypoint()
            p.pose=wp.pose
            stop_idx=max(self.stopline_wp_idx-closest_idx-3,0) # 4 is a distance buffer to make the car stop behind the line
            dist = self.distance(waypoints,i,stop_idx)
            vel=math.sqrt(2*MAX_DECEL*dist)
            if vel<1.:
                vel=0.
            p.twist.twist.linear.x=min(vel,wp.twist.twist.linear.x)
            temp.append(p)
        return temp
    def slowdown(self,waypoints,closest_idx):
        temp=[]
        for i,wp in enumerate(waypoints):
            p=Waypoint()
            p.pose=wp.pose
            #stop_idx=max(self.stopline_wp_idx-closest_idx-3,0) # 4 is a distance buffer to make the car stop behind the line
            #dist = self.distance(waypoints,i,stop_idx)
            #vel=math.sqrt(4*MAX_DECEL*dist)
            #if vel<1.:
            #    vel=0.
            #p.twist.twist.linear.x=min(vel,wp.twist.twist.linear.x)
            p.twist.twist.linear.x=wp.twist.twist.linear.x/2.0
            temp.append(p)
        return temp  
    def faststart(self,waypoints,closest_idx):
        temp=[]
        for i,wp in enumerate(waypoints):
            p=Waypoint()
            p.pose=wp.pose
            #stop_idx=max(self.stopline_wp_idx-closest_idx-3,0) # 4 is a distance buffer to make the car stop behind the line
            #dist = self.distance(waypoints,i,stop_idx)
            #vel=math.sqrt(4*MAX_DECEL*dist)
            #if vel<1.:
            #    vel=0.
            #p.twist.twist.linear.x=min(vel,wp.twist.twist.linear.x)
            #rospy.logerr(">>>> Speed :%s",wp.twist.twist.linear.x)
            p.twist.twist.linear.x=2.0*wp.twist.twist.linear.x
            #p.twist.twist.linear.x=60
            temp.append(p)
        return temp      
    def initial_stop(self,waypoints,closest_idx):
        temp=[]
        for i,wp in enumerate(waypoints):
            p=Waypoint()
            p.pose=wp.pose
            stop_idx=max(self.stopline_wp_idx-closest_idx-4,0) # 4 is a distance buffer to make the car stop behind the line
            dist = self.distance(waypoints,i,stop_idx)
            vel=math.sqrt(2*MAX_DECEL*dist)
            p.twist.twist.linear.x=0.0
            temp.append(p)
        return temp
    def extrapolate_acceleration(self,idx1,idx2):
        temp = []
        # Exponential sequence for acceleration extrapolation
        temp_seq = np.logspace(1,13,num=20,base=1.5,dtype='int')
        temp_seq = np.unique(temp_seq)
        temp_seq = temp_seq.tolist()
        #print(temp_seq)  # For debuggin purposes only
        for i in temp_seq:
            idx = idx1 + i
            if idx < idx2:
                wp = self.base_waypoints.waypoints[idx]
                temp.append(wp)
        return temp

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        self.base_lane=waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # Done
        self.stopline_wp_idx = msg.data

        # Auxiliary flag to indicate that the TL Detector instance was fully loaded and publishes
        self.stop_temp = True


        # New
        #rospy.logerr(">>Stopline idx :%s",self.stopline_wp_idx)



    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        # Not used in this implementation
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    # Closest point implementation
    def closest(self, wpts, wp):
        waypoint_array = np.asarray(wpts)
        delt = waypoint_array - wp
        distn = np.einsum('ij,ij->i',delt,delt)
        return np.argmin(distn)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
