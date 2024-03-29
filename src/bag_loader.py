#!/usr/bin/python3
# -*- coding: utf-8 -*- 

import rosbag
import rospy
from people_msgs.msg import People, Person
from visualization_msgs.msg import MarkerArray, Marker

import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan





class ScanBagLoader():

    def __init__(self, bag_path):

        self.path = bag_path
        self.bag = rosbag.Bag(bag_path)

        self.data = []
        self.timeStamps = []
        self.theta = None
        self.length = None
        self.range_max = None
        self.true_range_max = None
        self.range_min = None
        self.angle_min = None
        self.angle_max = None
        self.angle_increment = None
        self.time_increment = None
        self.scan_time = None
        self.tf_scan = 'laserscan'
        
    
    def getScanTopics(self):
        topics = []
        #self.bag = rosbag.Bag(self.bag_path)
        types = self.bag.get_type_and_topic_info()
        #print(types)
        #print("")
        for i in types.topics:
            print(i)
            if types.topics[i].msg_type == "sensor_msgs/LaserScan":
                topics.append(i)

        if len(topics) == 0:
            print("No laser scan message found!")
            exit()
        #elif len(topics) == 1:
        #    self.loadData(topics[0])
        return topics          


    def loadData(self, topic):
        self.bag = rosbag.Bag(self.path)
        if len(self.data) > 0:
            self.data = []
            self.theta = None
        for topic,msg,t in self.bag.read_messages(topics=[topic]):
            if self.theta is None:
                self.true_range_max = msg.range_max
                if msg.range_max > 10.0:
                    self.range_max = 10.0
                else:
                    self.range_max = msg.range_max
                self.theta = np.arange(msg.angle_min, msg.angle_max+msg.angle_increment, msg.angle_increment)
                if len(self.theta)>len(msg.ranges):
                    self.theta = self.theta[0:len(msg.ranges)]
                self.angle_min = msg.angle_min
                self.angle_max = msg.angle_max
                self.angle_increment = msg.angle_increment
                self.range_min = msg.range_min
                self.time_increment = msg.time_increment
                self.scan_time = msg.scan_time
                
            self.data.append(np.array(msg.ranges))
            self.timeStamps.append(t)
        self.bag.close()
        self.length = len(self.data)
        print("Loaded %d laser scan messages" % self.length)
        #print("theta: ")
        #print(self.theta)
        #print("timestamps:")
        #print(self.timeStamps)


    def get_cartesian(self, index):
        """
        Return the scaled cartesian on a scale of 0-1,0-1 where .5,.5c is 0,0r
        """
        output = []
        for i in range(len(self.theta)):
            x = math.cos(self.theta[i])*self.data[index][i]/self.range_max/2+0.5
            y = math.sin(self.theta[i])*self.data[index][i]/self.range_max/2+0.5
            output.append([x,y])
        return output



    def save_bag(self, people_dict, init_index, end_index, path):

        people_topic = '/scan/people'
        marker_topic = '/scan/people/markers'

        scan_msg = LaserScan()
        scan_msg.angle_increment = self.angle_increment
        scan_msg.angle_max = self.angle_max
        scan_msg.angle_min = self.angle_min
        scan_msg.time_increment = self.angle_increment
        scan_msg.scan_time = self.scan_time
        scan_msg.range_min = self.range_min
        scan_msg.range_max = self.true_range_max
        scan_msg.intensities = []

        index = init_index
        
        with rosbag.Bag(path, 'w') as outbag:

            for people in people_dict['people']:
                
                scan_msg.ranges = self.data[index]
                ts = self.timeStamps[index]
                scan_msg.header.frame_id = self.tf_scan
                scan_msg.header.stamp = ts #rospy.Time.from_sec(people['timestamp']) 
                #sensor_msgs/LaserScan message
                outbag.write(self.tf_scan, scan_msg, ts)
                print('people detected scan ', index, ': ', len(people['circles']))
                people_msg, marker_msg = self.build_messages(people['circles'], ts) #people['timestamp']
                #people_msgs/People message
                outbag.write(people_topic, people_msg, ts)
                #visualization_makers/MarkerArray
                outbag.write(marker_topic, marker_msg, ts)

                index+=1
            # for topic, msg, t in rosbag.Bag(self.path).read_messages():

            #     if t.to_sec() >= init_t and t.to_sec() < last_t:
                
            #         if topic == 'tf' or topic == '/tf':
            #             outbag.write(topic, msg, t)

            #         if topic == scan_topic or topic == ('/'+scan_topic):
            #             outbag.write(topic, msg, t)

            #             if index < len(people_dict['people']):
            #                 scan_msg, people_msg, marker_msg = self.build_messages(people_dict, index, t, msg.header)

            #                 #people_msgs/People message
            #                 outbag.write(people_topic, people_msg, t)

            #                 #visualization_makers/MarkerArray
            #                 outbag.write(marker_topic, marker_msg, t)

            #             index += 1
            outbag.close()



    def build_messages(self, circles, time):

        p_msg = People()
        p_msg.header.stamp = time #rospy.Time.from_sec(time)
        p_msg.header.frame_id = self.tf_scan

        m_msg = MarkerArray()

        if len(circles) > 0:
            for p in circles:

                # Person
                person = Person()
                person.name = str(p['idp'])
                person.tagnames.append(str(p['type']))
                person.reliability = 1.0
                person.position.x = p['x']
                person.position.y = p['y']
                person.position.z = 0.0
                p_msg.people.append(person)
                
                # Marker
                marker = Marker()
                marker.header.frame_id = self.tf_scan
                marker.header.stamp = time #rospy.Time.from_sec(time)
                marker.id = p['idp']
                #marker.ns = ''
                marker.type = 3 #cylinder
                marker.action = 0 #ADD
                marker.lifetime = rospy.Time(0.1)
                marker.pose.position.x = p['x']
                marker.pose.position.y = p['y']
                marker.pose.position.z = 1.6/2.5
                marker.pose.orientation.w = 1.0
                marker.scale.x = p['r']*2.0
                marker.scale.y = p['r']*2.0
                marker.scale.z = 1.6
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 0.6
                if(p['type'] == 1):
                    marker.color.g = 1.0
                elif(p['type'] == 2):
                    marker.color.b = 1.0
                elif(p['type'] == 3):
                    marker.color.r = 1.0
                    marker.color.g = 1.0

                m_msg.markers.append(marker)

        return p_msg, m_msg


class ImageBagLoader():

    def __init__(self, bag_path):

        self.path = bag_path
        self.bag = rosbag.Bag(bag_path)

        self.images = []
        self.timeStamps = []

        self.width = None
        self.height = None
        

    def getImageTopics(self):
        topics = []
        #self.bag = rosbag.Bag(self.bag_path)
        types = self.bag.get_type_and_topic_info()
        #print(types)
        #print("")
        for i in types.topics:
            print(i)
            if types.topics[i].msg_type == "sensor_msgs/Image":
                topics.append(i)

        if len(topics) == 0:
            print("No image messages found!")
        #elif len(topics) == 1:
        #    self.loadData(topics[0])
        return topics          


    def loadData(self, topic):
        self.bag = rosbag.Bag(self.path)
        bridge = CvBridge()
        if len(self.images) > 0:
            self.images = []
        for topic,msg,t in self.bag.read_messages(topics=[topic]):
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.images.append(cv_image)
            self.timeStamps.append(t)
        self.bag.close()
        self.length = len(self.images)
        height, width, = self.images[0].shape
        self.width = width
        self.height = height
        print("Loaded %d image messages" % self.length)
