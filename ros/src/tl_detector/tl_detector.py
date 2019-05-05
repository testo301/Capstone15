#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import os
import uuid
import tf
import cv2
import yaml

# Constants
# Threshold for counting the states of the traffic lights
STATE_COUNT_THRESHOLD = 3
# The lookahead distance to the next traffic lights that would trigger detection
TL_DIST = 200

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')
        #rate=rospy.Rate(50)

        self.pose = None
        self.load_status = False


        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.pub_light=None
        self.lights_2d=None
        self.waypoints_2d = None
        # Counter for processing every 5th image from camera
        self.counter_processing = 1
        self.inside_state = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        # NEW
        #if self.load_status == False:
        #    self.upcoming_red_light_pub.publish(Int32(288))
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN


        #self.bridge = CvBridge()
        #self.light_classifier = TLClassifier()
        #self.listener = tf.TransformListener()


        self.last_wp = -1
        self.state_count = 0

        # Closest vehicle waypoint
        self.closest_id_temp = None

        self.light_waypoint_id_temp = None

        # Closest light waypoint
        self.light_wp_temp = None

        # Real light state - for testing correctness of the model only
        self.temp_light_state  = None

        self.lights_2d = None
        self.temp_line = None


        # To be deleted
        self.flag_enter = None
        self.intheloop = None
        self.flag_image_cb  = None
        self.closest_light_temp = None

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.load_status = self.light_classifier.load_status_tl


        rospy.spin()

        #self.loop()

    # Loop for debugging only
    def loop(self):
        rate=rospy.Rate(30)
        while not rospy.is_shutdown():

            #rospy.logerr("Light 2D :%s",self.intheloop )
            #rospy.logerr("Light 2D :%s",self.flag_enter)

            #if self.light_wp_temp and self.closest_id_temp  and ( self.light_wp_temp  - self.closest_id_temp < 70):
            #    rospy.logerr("Close light")

            if self.pose and self.pub_light:
                self.publish_light()
                #rospy.logerr("Light :%s",self.pub_light)
                self.flag_image_cb = 1

            rate.sleep()

    # For debugging purposes only
    def publish_light(self):
        self.upcoming_red_light_pub.publish(self.pub_light)

    def pose_cb(self, msg):
        self.pose = msg

        #rospy.logerr("Load status :%s",self.load_status )

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d =  [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]

    def traffic_cb(self, msg):
        self.lights = msg.lights
        self.lights_2d  = [[lght.pose.pose.position.x, lght.pose.pose.position.y] for lght in msg.lights]

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.flag_image_cb = 1

        self.has_image = True
        self.camera_image = msg
        self.counter_processing += 1

        # Processing every second image
        # Nigdy nie wchodzi do tego warunku
        #if self.load_status == False:
        #    self.upcoming_red_light_pub.publish(Int32(285))
            #rospy.logerr(">> Publishing inside False :%s",285)

        # If the model is not entirely loaded, publishing temporary stop point

        if self.load_status == False and self.pose and self.lights_2d and self.waypoints:    
            temp_idx=self.get_closest_waypoint(self.pose.pose.position.x,self.pose.pose.position.y)
            #rospy.logerr(">> Temp idx :%s",temp_idx)
            self.upcoming_red_light_pub.publish(Int32(temp_idx))
            self.pub_light=Int32(temp_idx)



        #if (self.counter_processing % 2 == 0 and self.load_status == True):
        elif (self.counter_processing % 2 == 0):


            light_wp, state = self.process_traffic_lights()
            self.light_wp_temp = light_wp
        
            # For debugging purposes only
            #rospy.logerr(">> Inside True :%s",light_wp)
            #rospy.logerr(">> Image_CB state :%s",state)
            #rospy.logerr("Self state :%s",self.state)
            #rospy.logerr("Counter processing :%s",self.counter_processing)
            #rospy.logerr("Light wp: %s",light_wp)


            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
            elif self.state_count >= STATE_COUNT_THRESHOLD:
                self.last_state = self.state
                # Conservative approach - slowing down on both YELLOW and RED lights


                light_wp = light_wp if state == TrafficLight.YELLOW or state == TrafficLight.RED or self.load_status == False else -1

                self.last_wp = light_wp
                self.upcoming_red_light_pub.publish(Int32(light_wp))
                self.pub_light=Int32(light_wp)
            else:
                self.upcoming_red_light_pub.publish(Int32(self.last_wp))
                self.pub_light=Int32(self.last_wp)
            self.state_count += 1


    def get_closest_waypoint(self, x,y):   
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """

        if self.waypoints_2d:

            #closest_point=self.waypoint_tree.query([x,y],1)[1]

            closest_point = self.closest(self.waypoints_2d,[x,y])

            # Trick for finding the coordinates ahead, as described in the Partial Walkthrough
            closest_coordinates = np.array(self.waypoints_2d[closest_point])
            prev_xy = np.array(self.waypoints_2d[closest_point-1])
            current_xy = np.array([x,y])
            dotproduct = np.dot(closest_coordinates-prev_xy,current_xy-closest_coordinates)
            if dotproduct > 0:
                closest_point = (closest_point+1)%len(self.waypoints_2d)
            return closest_point

    # Closest point implementation
    def closest(self, wpts, wp):
        waypoint_array = np.asarray(wpts)
        delt = waypoint_array - wp
        distn = np.einsum('ij,ij->i',delt,delt)
        return np.argmin(distn)
    def get_light_state99(self, light):
        return light.state
    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # UNCOMMENT LATER!


        if self.load_status == False:
            return TrafficLight.RED

        if self.load_status == True:
            if(not self.has_image):
                self.prev_light_loc = None
                return False

            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")


            # Saving captured images for labeling
            # For labeling purposes only
            #full_path = os.path.join("saved_images", "{}.png".format(str(uuid.uuid4())))
            #cv2.imwrite(full_path, cv_image)

            # Capturing ground truth for model debugging
            self.temp_light_state = light.state

            # Getting classification of the input image
            return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # Auxiliary flag for debugging
        self.flag_enter = 1
        closest_light=None
        light_waypoint_id=None
        # Lists of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions=self.config['stop_line_positions']



        #rospy.logerr(">> Stop line positions :%s",stop_line_positions)




        if(self.pose and  self.lights_2d and self.waypoints):    
            # Auxiliary flag for debugging
            self.intheloop=1
            car_wp_id=self.get_closest_waypoint(self.pose.pose.position.x,self.pose.pose.position.y)
            self.closest_id_temp = car_wp_id

            # TODO find the closest visible traffic light (if one exists)
            # Done using get_closest_waypoint function
            diff=len(self.waypoints.waypoints)

            for i, light in enumerate(self.lights):
                # Getting the light waypoint index
                self.ith = i
                line=stop_line_positions[i]
                self.temp_line = stop_line_positions[i]
                self.temp_var4 = line
                temp_wp_id=self.get_closest_waypoint(line[0],line[1])
                # Finding the closest stop line waypoint index
                d=temp_wp_id-car_wp_id
                if d>=0 and d<diff:
                    diff=d
                    closest_light = light
                    light_waypoint_id=temp_wp_id
                    self.light_waypoint_id_temp = light_waypoint_id
                    self.closest_light_temp = closest_light
        # Checking conditions for the distance to the traffic light less than k and processing every nth traffic light state
        if self.load_status == False:
            return light_waypoint_id, TrafficLight.RED




        if closest_light and self.light_waypoint_id_temp and self.closest_id_temp  and ( self.light_waypoint_id_temp  - self.closest_id_temp < TL_DIST):
            state=self.get_light_state(closest_light)
            self.inside_state = state

            # For debugging purposes only
            #rospy.logerr(">> Model light :%s",self.inside_state)
            #rospy.logerr(">> True light :%s",self.temp_light_state )
            

            return light_waypoint_id, state
        return -1, TrafficLight.UNKNOWN



if __name__ == '__main__':
    try:
        TLDetector()
        #rospy.logerr("Self lights main :%s",self.lights)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
