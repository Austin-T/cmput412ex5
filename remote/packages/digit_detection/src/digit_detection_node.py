#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from model import CNN


class DigitDetectionNode(DTROS):

    def __init__(self, node_name):
        super(DigitDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        # Services
        self.service = rospy.Service(f'/{self.veh}/digit_detection_node/digit_detection_service', CompressedImage, self.detect_digit)
        
        # image processing tools
        self.bridge = CvBridge()
        
        # MLP model
        self.INPUT_H = 28
        self.INPUT_W = 28
        self.INPUT_DIM = self.INPUT_H * self.INPUT_W
        self.OUTPUT_DIM = 10

        self.model = CNN(self.INPUT_DIM, self.OUTPUT_DIM)
        
    def detect_digit(self, img_msg):

        # convert image into cv2 type
        cv_image = None
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(img_msg)
        except CvBridgeError as e:
            self.log(e)
            return []

        # reformat the image to the appropriate 28 * 28 size
        cv_image = cv2.resize(cv_image, (self.INPUT_H, self.INPUT_W))
        
        # normalize the image
        cv_image = (cv_image - np.mean(cv_image)) / np.std(cv_image)
        
        # TODO: flatten the image
        
        # TODO: predict the digit
        
        # TODO: return the result
        
    def hook(self):
        print("SHUTTING DOWN")


if __name__ == "__main__":
    node = DigitDetectionNode("digit_detection_node")
    node.run()
