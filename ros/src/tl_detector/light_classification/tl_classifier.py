from styx_msgs.msg import TrafficLight
import numpy as np
import os
import sys
import tensorflow as tf
import time
import rospy
from collections import defaultdict
from io import StringIO
from PIL import Image
from glob import glob
import yaml

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

class TLClassifier(object):
    def __init__(self):
        # TODO load classifier
        # Done
        self.light = TrafficLight.UNKNOWN
        simpath = './light_classification/model/frozen_inference_graph.pb'
        lotpath = './light_classification/model/frozen_41.pb'


        
        # Capturing is_site flag for model choice
        conf = rospy.get_param("/traffic_light_config")
        config = yaml.load(conf)
        ssd_inception_v2 = lotpath if config['is_site'] else simpath
        rospy.logwarn("Path to model: %s",ssd_inception_v2)
        rospy.logwarn("Is Site: %s",config['is_site'])

        # Configuration for detection
        self.config = tf.ConfigProto(log_device_placement=False)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8  
        self.config.operation_timeout_in_ms = 5000000 


        NUM_CLASSES = 4
        self.load_status_tl = False

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
    
            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(ssd_inception_v2, 'rb') as fid:
            
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            self.sess = tf.Session(graph=self.detection_graph) 

        # Input and output tesnors
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                
        # Detection box where the individual traffic light is detected
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                
        # Score is confidence level and is subject to cutoff threshold
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        print("Entire model loaded")
        self.load_status_tl = True

    def load_image_into_numpy_array(self,image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        # Done


        out_class = TrafficLight.UNKNOWN

        image_np_expanded = np.expand_dims(image, axis=0)

        # Light class detection
        with self.detection_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections], feed_dict={self.image_tensor: image_np_expanded})
        # Cleaning dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
    
        # 50% threshold for significance of detection
        min_score_thresh = 0.3

        # List for appending significant classes
        class_list = []
                    
        for i in range(boxes.shape[0]):
            if scores[i] > min_score_thresh:
                class_list.append(classes[i])

        class_list_mode = None
        if class_list:
            class_list_mode = (max(set(class_list), key=class_list.count))
            # If red light is detected in any of the boxes
            if 2 in class_list:
                self.light = TrafficLight.RED
                out_class = TrafficLight.RED
            elif (class_list_mode == 1):
                self.light = TrafficLight.GREEN
                out_class = TrafficLight.GREEN
            elif (class_list_mode == 3):
                self.light = TrafficLight.YELLOW
                out_class = TrafficLight.YELLOW
            else:
                self.light = TrafficLight.UNKNOWN
                out_class = TrafficLight.UNKNOWN
        
        
        # For debugging
        #return [classes, scores]

        return out_class
