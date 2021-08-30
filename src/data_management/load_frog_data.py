#!/usr/bin/python3
# This Python file uses the following encoding: utf-8

import numpy as np
#import matplotlib.pyplot as plt
import os
import math
import csv
import json

#from numpy.core.defchararray import lstrip
import rosbag
import rospy
from sensor_msgs.msg import LaserScan
from people_msgs.msg import People, Person
from visualization_msgs.msg import MarkerArray, Marker

#from tensorflow.python.ops.numpy_ops.np_math_ops import isnan


class LoadData:

    def __init__(self):

        # laser info of FROG dataset
        self.nPoints = 720
        self.laserIncrement = 0.004363323096185923
        self.laserFoV = (self.nPoints-1)*self.laserIncrement  
        self.angleMin = -1.5707963705062866
        self.angleMax = 1.5664329528808594
        self.maxRange = 60.0
        self.myMaxRange = 25.0
        self.maxPeopleRange = 10.0 #We do not detect people farer than 10 meters
        self.maxDrowRange = 29.96

        # image info
        self.img_width = 200       # px  (columns)
        self.img_height = 200      # px  (rows)
        self.img_resolution = 0.05 # 5cm/px
        self.img_origin = [99, 99] # px



    def laser_angles(self, N, fov=None):
        fov = fov or self.laserFoV
        return np.linspace(-fov*0.5, fov*0.5, N)


    def rphi_to_xy(self, r, phi):
        return r * np.cos(phi), r * np.sin(phi)  # -np.sin(phi)


    def xy_to_rphi(self, x, y):
        r = math.sqrt(x*x + y*y)
        phi = math.atan2(y, x)
        return [r, phi]

    def scan_to_xy(self, scan, thresh=None, fov=None):
        s = np.array(scan, copy=True)
        if thresh is not None:
            s[s > thresh] = np.nan
        return self.rphi_to_xy(s, self.laser_angles(len(scan), fov))


    def load_numpy_formatted_data(self, file_path, type=1):
        data = []
        data = np.load(file_path)
        if type == 1:
            data = data.astype(np.float32)
            #data[np.isinf(data)] = self.maxPeopleRange + 1
            #data[data > self.maxPeopleRange] = self.maxPeopleRange + 1
            #Normalize -> already normalized!
            #data = data/(self.maxPeopleRange + 1)
            #assert ((np.all(data <= 1.0) and np.all(data >= 0.0))), 'x_data out of range!!!!'
            if np.any(np.isnan(data)):
                print('There is nan values!!!!!!!!')
                print(np.isnan(data))
            if (np.any(data > 1.0) or np.any(data < 0.0)):
                print('x_data out of range!!!!')
                print('data max:', data.max())
                print('data min:', data.min())
    
        else:
            data = data.astype(np.uint32) #np.uint8
            #remove other classes that are not people
            data[data == 2] = 0 #baby carts
            data[data == 3] = 0 #wheelchairs
            #assert ((np.all(data <= 1.0) and np.all(data >= 0.0))), 'y_data out of range!!!!'
            if ((np.any(data > 1) or np.any(data < 0))):
                print('y_data out of range!!!!')
                print('data max:', data.max())
                print('data min:', data.min())

        return data


    def load_numpy_file(self, file_path):
        data = []
        #try:
        data = np.load(file_path, allow_pickle=True)
        #except IOError:
        #    print("ERROR: file data %s, could NOT be loaded!" % file_path)
        #    return

        print('Data successfully loaded from:')
        print(file_path)
        print("data shape:")
        print(data.shape)
        #print("data[0]")
        #print(data[0])
        #data2 = dict(data)
        #data2 = {d for d in data}
        #print("data[0] dict id")
        #print(data[0]['id'])
        ids = [d['id'] for d in data]
        timestamps = [d['timestamp'] for d in data]
        scans = []
        if data[0]['ranges'] is not None:
            scans = [d['ranges'] for d in data]
            scans = scans.astype(np.float32)
            if np.any(np.isnan(scans)):
                print('There is nan values in the scan!!!!!!!!')
            scans[np.isinf(scans)] = self.maxPeopleRange + 1
            scans[scans > self.maxPeopleRange] = self.maxPeopleRange + 1
            #Normalize
            scans = scans/(self.maxPeopleRange + 1)
        elif data[0]['label'] is not None:
            scans = [d['label'] for d in data]
        return ids, timestamps, scans


    def load_old_csv_file(self, file_path, label=0):
        #dialect = csv.excel
        #dialect.quoting = csv.QUOTE_NONNUMERIC #csv.QUOTE_NONE  
        ids=[]
        timestamps=[]
        scans=[]
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONE, skipinitialspace=True)
            count = 0
            for row in reader:
                if count > 0:
                    #print(row)
                    id = int(row[0])
                    #print('id:', id)
                    ids.append(id)
                    timestamp = float(row[1])
                    #print('timestamp:', timestamp)
                    timestamps.append(timestamp)
                    scan = np.char.strip(row[2:], '"[]')
                    scan = scan.astype(np.float32)  #for ranges
                
                    # ranges (float values)
                    if label == 0:
                        scan[np.isinf(scan)] = self.maxRange + 1
                    # label ([0,1] values)
                    if label == 1:
                        scan = scan.astype(np.uint8)      #for labels

                    scans.append(scan)
                    #scan = row[2:]
                    #print('scan:', scan)
                count+=1
        print('scans length:', len(scans))
        return ids, timestamps, scans


    def load_csv_file(self, file_path):
        data = []
        dt = np.dtype([('id', np.int), ('timestamp', np.float64), ('scan', np.float32, (720,))])
        try:
            data = np.genfromtxt(file_path, dtype=dt, delimiter=',', skip_header=1) # delimiter=',',
        except IOError:
            print("ERROR: file data %s, could NOT be loaded!" % file_path)
            return

        print("data shape:")
        print(data.shape)
        #print('data[0]:', data[0])
        ids = np.array(data['id'], dtype=int)
        #print('ids[0]:', ids[0])
        timestamps = np.array(data['timestamp'], dtype=float)
        #print('timestamps[0]:', timestamps[0])
        scans = np.array(data['scan'], dtype=float)
        #print('scans[0]:', scans[0])

        return ids, timestamps, scans



    def load_data(self, path, type=0):

        ids=[]
        timestamps=[]
        ranges=[]
        c = 0
        for file in sorted(os.listdir(path)):

            print("loading file:", file)
            file_path = os.path.join(path, file)

            if file.endswith('.csv') or file.endswith('.txt'):
                id, timestamp, scan = self.load_csv_file(file_path)

            elif file.endswith('.npy'):
                id, timestamp, scan = self.load_numpy_file(file_path)

            if(type==0):
                # cut far values and normalize the ranges
                scan[np.isinf(scan)] = self.maxPeopleRange + 1
                # we check if there are nan values just in case
                scan[np.isnan(scan)] = self.maxPeopleRange + 1
                scan[scan > self.maxPeopleRange] = self.maxPeopleRange + 1
                #Normalize
                scan = scan/(self.maxPeopleRange + 1)
                scan = scan.astype(np.float32)
                if np.any(np.isnan(scan)):
                    print('Load_data. There is nan values in the scan!!!!!!!!')
            else:
                #scan = np.array(scan, dtype=int)
                scan = scan.astype(np.uint8)

            #x_data = np.concatenate((x_data, x_ranges), axis=0)
            if c == 0:
                ids = id
                timestamps = timestamp
                ranges = scan
            else:
                ids = np.concatenate((ids, id), axis=0)
                timestamps = np.concatenate((timestamps, timestamp), axis=0)
                ranges = np.concatenate((ranges, scan), axis=0)
            c+=1
            
        return ids, timestamps, ranges



    def join_data(self, x_data_path, y_data_path):

        #x_data_path = os.path.join(directory_path, 'scans')
        print("Loading data from", x_data_path)
        x_ids, x_ts, x_ranges = self.load_data(x_data_path, type=0)
        print('x_ranges.shape:', x_ranges.shape)
        #x_data = np.array(x_ranges, dtype=float)
        #x_data = x_ranges.astype(float)
        x_data = x_ranges

        #y_data_path = os.path.join(directory_path, 'labels')
        print("Loading data from", y_data_path)
        y_ids, y_ts, y_labels = self.load_data(y_data_path, type=1)
        #y_data = np.array(y_labels, dtype=int)
        y_data = y_labels

        return x_data, y_data

    
    def join_formatted_data(self, x_data_path, y_data_path):

        x_data = []
        y_data = []

        print("Loading data from", x_data_path)
        c = 0
        for file in sorted(os.listdir(x_data_path)):
            if file.endswith('.npy'):
                file_path = os.path.join(x_data_path, file)
                x_ranges = self.load_numpy_formatted_data(file_path, type=1)
                print("loading file:", file, 'shape:', x_ranges.shape)
                if c == 0:
                    x_data = np.copy(x_ranges)
                else:
                    x_data = np.concatenate((x_data, x_ranges), axis=0)
                c += 1
            else:
                print('Error! File %s is not in numpy format! (.npy)' % file)
                return False, x_data, y_data


        print("Loading data from", y_data_path)
        c = 0
        for file in sorted(os.listdir(y_data_path)):
            if file.endswith('.npy'):
                file_path = os.path.join(y_data_path, file)
                y_labels = self.load_numpy_formatted_data(file_path, type=2)
                print("loading file:", file, 'shape:', y_labels.shape)
                if c == 0:
                    y_data = np.copy(y_labels)
                else:
                    y_data = np.concatenate((y_data, y_labels), axis=0)
                c += 1
                
            else:
                print('Error! File %s is not in numpy format! (.npy)' % file)
                return False, x_data, y_data

        print('Final x_data.shape:', x_data.shape, 'type:', x_data.dtype)
        print('Final y_data.shape:', y_data.shape, 'type:', y_data.dtype)

        return True, x_data, y_data




    def save_data(self, x_data, y_data, save_dir, prefix=None):

        x_name = prefix + '_x_data.npy'
        x_data_dir = os.path.join(save_dir, x_name)
        np.save(x_data_dir, x_data)
        y_name = prefix + '_y_data.npy'
        y_data_dir = os.path.join(save_dir, y_name)
        np.save(y_data_dir, y_data)
        print('Data saved in:')
        print(x_data_dir)
        print(y_data_dir)



    def split_data(self, x_data, y_data, ptraining, pval, ptest):
        
        # input data
        n = x_data.shape
        x_train = x_data[0:int(n[0]*ptraining)]
        end_val = int(n[0]*(ptraining+pval))
        x_val = x_data[int(n[0]*ptraining):end_val]
        x_test = x_data[end_val:]

        # labels
        n = y_data.shape
        y_train = y_data[0:int(n[0]*ptraining)]
        end_val = int(n[0]*(ptraining+pval))
        y_val = y_data[int(n[0]*ptraining):end_val]
        y_test = y_data[end_val:]

        return x_train, y_train, x_val, y_val, x_test, y_test
        


    def scan_to_new(self, path, label=0):
        for file in sorted(os.listdir(path)):
            if file.endswith('_scans.csv') or file.endswith('_labels.csv'):
                print("Opening file:", file)
                file_path = os.path.join(path, file)
                ids, timestamps, scans = self.load_old_csv_file(file_path, label=label)
                filename = str(file).replace('.csv', '_new.csv')
                file_path = os.path.join(path, filename)

                #data = np.array(zip(ids, timestamps, scans))
                #np.savetxt(file_path, data, delimiter=',')

                with open(file_path, "w", newline='') as outfile:
                    c = 0
                    writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                    for (id, ts, sc) in zip(ids, timestamps, scans):
                        row = []
                        row.append(int(id))
                        row.append(float(ts)) 
                        for item in sc:
                            row.append(item)
                        writer.writerow(row)

                print("Stored file:", file_path)

                
    def scan_to_drow(self, path):
        #we just need to replace the values higher than self.maxDrowRange = 29.96
        for file in sorted(os.listdir(path)):
            if file.endswith('_scans.csv'):
                print("Opening file:", file)
                file_path = os.path.join(path, file)
                ids, timestamps, scans = self.load_csv_file(file_path)

                scans[np.isinf(scans)] = self.maxDrowRange
                scans[scans > self.maxDrowRange] = self.maxDrowRange
                scans = np.round(scans, decimals=3)

                filename = str(file).replace('.csv', '_drow.csv')
                file_path = os.path.join(path, filename)
                with open(file_path, "w", newline='') as outfile:
                    writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                    for (id, ts, sc) in zip(ids, timestamps, scans):
                        row = []
                        row.append(int(id))
                        row.append(float(ts)) 
                        for item in sc:
                            row.append(item)
                        writer.writerow(row)

                print("Stored file:", file_path)


    def circles_to_drow(self, path):
    
        for file in sorted(os.listdir(path)):
            if file.endswith('_circles.csv'):
                print("Opening file:", file)
                file_path = os.path.join(path, file)
                ids, timestamp, circles = self.load_circles_csv_file(file_path)

                #Now we need to transform our cartesian coordinates to polar
                polar_circles = self.circles_to_polar(circles)

                filename = str(file).replace('.csv', '_drow.wp')
                file_path = os.path.join(path, filename)
                with open(file_path, "w", newline='') as outfile:
                    writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                    for (id, pc) in zip(ids, polar_circles):
                        row = []
                        row.append(int(id))
                        #row.append(float(ts)) 
                        row.append(pc)
                        writer.writerow(row)

                print("Stored file:", file_path)


    def circles_to_polar(self, circles):
        # circles is a np array of dictionaries:
        # e.g.: [{"circles": [{'idp':2, 'x':8.83, 'y':2.0, 'r':0.4, 'type':1}, {'idp':3, 'x':0.5, 'y':-1.12, 'r':0.4, 'type':1}]}]
        # Must be tranform to:
        # [[r,phi],[r,phi]]
        polar_cir = []
        for circ in circles:
            #print('circ:', circ)
            circ_row = []
            if len(circ['circles']) > 0:
                for p in circ['circles']:
                    polar = np.round(self.xy_to_rphi(p['x'], p['y']), decimals=3)
                    circ_row.append(polar)

            polar_cir.append(circ_row)

        return polar_cir




    def csv_to_bag(self, scans_path, circles_path, path):

        # join the scans
        # If the file naming is correct, 'sorted' gives the correct order 
        c = 0
        scan_ids = []
        scan_ts = []
        scan_ranges = []
        for file in sorted(os.listdir(scans_path)):

            print("loading scan file:", file)
            file_path = os.path.join(scans_path, file)

            if file.endswith('_scans.csv'):
                id, timestamp, scan = self.load_csv_file(file_path)

            if c == 0:
                scan_ids = id
                scan_ts = timestamp
                scan_ranges = scan
            else:
                scan_ids = np.concatenate((scan_ids, id), axis=0)
                scan_ts = np.concatenate((scan_ts, timestamp), axis=0)
                scan_ranges = np.concatenate((scan_ranges, scan), axis=0)
            c+=1

        # join the circles files
        c = 0
        cir_ids = []
        cir_ts = []
        cir_people = []
        for file in sorted(os.listdir(circles_path)):

            print("loading circle file:", file)
            file_path = os.path.join(circles_path, file)

            if file.endswith('_circles.csv'):
                id, timestamp, circles = self.load_circles_csv_file(file_path)

            if c == 0:
                cir_ids = id
                cir_ts = timestamp
                cir_people = circles
            else:
                cir_ids = np.concatenate((cir_ids, id), axis=0)
                cir_ts = np.concatenate((cir_ts, timestamp), axis=0)
                cir_people = np.concatenate((cir_people, circles), axis=0)
            c+=1

        self.write_people_bag(path, scan_ranges, cir_people, scan_ts)

                

    def load_circles_csv_file(self, file_path):
        ids=[]
        timestamps=[]
        circles=[]

        file_circles = open(file_path, 'r')
        c = 0
        for line in file_circles:
            if c > 0:
                x = line.find('[')
                init = line[:(x)]
                init = init.rsplit(',')
                id = int(init[0])
                ts = float(init[1])
                rest = line[x:]
                #transform to a dictionary with json
                rest = rest.replace("'", '"')
                cir = '{"circles":'+rest+'}'
                dict = json.loads(cir)
                #print('id:', id)
                #print('ts:', ts)
                #print('dict:', dict)
                ids.append(id)
                timestamps.append(ts)
                circles.append(dict)
            c+=1

        ids = np.array(ids, dtype=np.int32)
        timestamps = np.array(timestamps, dtype=np.float64)
        circles = np.array(circles, dtype=object)
        print('data length: ', len(ids))

        return ids, timestamps, circles



    def write_people_bag(self, path, scans, people, scan_ts):

        people_topic = '/scan/people'
        marker_topic = '/scan/people/markers'

        # data of the frog laserscan
        scan_msg = LaserScan()
        scan_msg.angle_increment = 0.004363323096185923
        scan_msg.angle_max = 1.5664329528808594
        scan_msg.angle_min = -1.5707963705062866
        scan_msg.time_increment = 0.0000173611115
        scan_msg.scan_time = 0.02500000037252903
        scan_msg.range_min = 0.023000000044703484
        scan_msg.range_max = 60.0
        scan_msg.intensities = []
        scan_msg.header.frame_id = 'laserscan'

        try:
            bag_name = os.path.join(path, 'complete.bag')
            with rosbag.Bag(bag_name, 'w') as outbag:

                for i, ranges in enumerate(scans):
                        
                    scan_msg.ranges = ranges
                    ts = rospy.Time.from_sec(scan_ts[i])
                    scan_msg.header.stamp = ts
                    #sensor_msgs/LaserScan message
                    outbag.write('laserscan', scan_msg, ts)
                    #print('people detected scan ', i, ': ', len(people['circles']))
                    people_msg, marker_msg = self.build_messages(people[i], ts, scan_msg.header.frame_id) 
                    #people_msgs/People message
                    outbag.write(people_topic, people_msg, ts)
                    #visualization_makers/MarkerArray
                    outbag.write(marker_topic, marker_msg, ts)
                outbag.close()

        except:
            print("-------------------------------------")
            print('ERROR!!! Bag file could not be created!')
            print("-------------------------------------")
        print('')
        print('Bag succesfully recorded and stored in: ', bag_name)



    def build_messages(self, circles, time, frame):

        p_msg = People()
        p_msg.header.stamp = time #rospy.Time.from_sec(time)
        p_msg.header.frame_id = frame

        #print('build messages. circles:', circles)

        m_msg = MarkerArray()

        if len(circles['circles']) > 0:
            for p in circles['circles']:

                # Person
                person = Person()
                person.name = str(p['idp'])
                if "type" in p:
                    person.tagnames.append(str(p['type']))
                else:
                    person.tagnames.append('1')
                person.reliability = 1.0
                person.position.x = p['x']
                person.position.y = p['y']
                person.position.z = 0.0
                p_msg.people.append(person)
                
                # Marker
                marker = Marker()
                marker.header.frame_id = frame
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
                if "type" in p:
                    if(p['type'] == 1):
                        marker.color.g = 1.0
                    elif(p['type'] == 2):
                        marker.color.b = 1.0
                    elif(p['type'] == 3):
                        marker.color.r = 1.0
                        marker.color.g = 1.0
                else:
                    marker.color.g = 1.0
                
                m_msg.markers.append(marker)

        return p_msg, m_msg