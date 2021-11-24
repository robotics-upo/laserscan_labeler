#!/usr/bin/python3
# This Python file uses the following encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import csv
import json
import cv2
import rosbag
import rospy
from sensor_msgs.msg import LaserScan
from people_msgs.msg import People, Person
from visualization_msgs.msg import MarkerArray, Marker

#from tensorflow.python.ops.numpy_ops.np_math_ops import isnan


class LoadData:

    def __init__(self):

        #rospy.init_node('loadData')

        # laser info of FROG dataset
        self.frog_nPoints = 720
        self.frog_laserIncrement = 0.004363323096185923 # = 0.25 degrees
        self.frog_laserFoV = (self.frog_nPoints-1)*self.frog_laserIncrement  #~180 degrees
        self.frog_angleMin = -1.5707963705062866
        self.frog_angleMax = 1.5664329528808594
        self.maxRange = 60.0
        self.myMaxRange = 25.0
        

        # laser info of DROW dataset
        self.drow_maxRange = 29.96
        self.drow_nPoints = 450
        self.drow_laserIncrement = np.radians(0.5)
        self.drow_laserFoV = (self.drow_nPoints-1)*self.drow_laserIncrement #~225 degrees

        # laser data for learning
        self.nPoints = 720 #1440
        self.laserIncrement = np.radians(0.25) 
        self.laserFoV = (self.nPoints-1)*self.laserIncrement
        self.maxPeopleRange = 10.0 #We do not detect people farer than 10 meters


        # image info
        self.img_width = 400 #200       # px  (columns)
        self.img_height = 400 #200      # px  (rows)
        self.img_resolution = 0.05      # 5cm/px
        self.img_origin = [199, 199]#[99, 99] # px


    def laser_angles(self, N, fov=None):
        fov = fov or self.frog_laserFoV
        return np.linspace(-fov*0.5, fov*0.5, N)


    # In drow dataset, the laser scan starts from the left side:
    # def rphi_to_xy(r, phi):
    # return r * -np.sin(phi), r * np.cos(phi)
    # In frog, it starts from the right.
    def rphi_to_xy(self, r, phi, frog=True): 
        if frog == True:
            return r * np.cos(phi), r * np.sin(phi)  
        else:
            return r * -np.sin(phi), r * np.cos(phi)


    # for Drow dataset:
    #def xy_to_rphi(x, y):
    # Axes rotated by 90 CCW by intent, so tat 0 is top.
    # return np.hypot(x, y), np.arctan2(-x, y)
    def xy_to_rphi(self, x, y, frog=True):
        if frog == True:
            r = np.hypot(x, y)  #math.sqrt(x*x + y*y) 
            phi = np.arctan2(y, x) #math.atan2(y, x) 
            return [r, phi]
        else:
            return np.hypot(x, y), np.arctan2(-x, y)


    def scan_to_xy(self, scan, thresh=None, fov=None, frog=True):
        s = np.array(scan, copy=True)
        if thresh is not None:
            s[s > thresh] = thresh + 1
        return self.rphi_to_xy(s, self.laser_angles(len(scan), fov), frog=frog)


    def worldToImg(self, x, y):
        #x1 = np.array(x, copy=True)
        #y1 = np.array(y, copy=True)
        #x1 = x1[~np.isnan(x1)]
        #y1 = y1[~np.isnan(y1)]
        px = self.img_origin[0] + np.rint(x/self.img_resolution)
        py = self.img_origin[1] - np.rint(y/self.img_resolution) 

        # just in case remove points outside the map 
        index = []
        for i in range(len(px)):
            sq = math.sqrt(px[i]*px[i] + py[i]*py[i])
            if sq >= (self.maxPeopleRange + 1):
                index.append(i)
            if px[i] >= self.img_width or py[i] >= self.img_height:
                #print('cell x[', i, ']:', px[i], 'y[', i, ']:', py[i], ' out of upper bounds!!!')
                index.append(i)
            elif px[i] < 0 or py[i] < 0:
                #print('cell x[', i, ']:', px[i], 'y[', i, ']:', py[i], ' out of under bounds (<0)!!!')
                index.append(i)
            elif px[i] == self.img_origin[0] and py[i] == self.img_origin[1]: #remove origin points
                #print('cell x[', i, ']:', px[i], 'y[', i, ']:', py[i], ' equals to origin!!!')
                index.append(i)
        #print('len outliers: ', len(index))

        pxn = [xi for idx, xi in enumerate(px) if not idx in index]
        pyn = [yi for idy, yi in enumerate(py) if not idy in index]

        xc = np.asarray(pxn, dtype = np.uint32)
        yc = np.asarray(pyn, dtype = np.uint32)

        return xc, yc 


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
                print('data out of range!!!!')
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



    def load_class_and_loc_labels(self, y_data_path, norm_type=1, polar=False):

        c = 0
        class_labels = []
        loc_labels = []
        for file in sorted(os.listdir(y_data_path)):

            file_path = os.path.join(y_data_path, file)

            # Classification
            if file.endswith('_class_labels.csv'):
                print("loading file:", file)
                try:
                    cdata = np.genfromtxt(file_path, delimiter=',', skip_header=False) 
                except IOError:
                    print("ERROR: file data %s, could NOT be loaded!" % file_path)
                    return
                print("class data shape:", cdata.shape)
                classes = np.array(cdata, dtype=int)

                # Localization 
                locfile = str(file).replace('_class_labels.csv', '_loc_labels.csv')
                file2_path = os.path.join(y_data_path, locfile)
                print("loading file:", locfile)
                try:
                    ldata = np.genfromtxt(file2_path, delimiter=',', autostrip=True, skip_header=False) 
                except IOError:
                    print("ERROR: file data %s, could NOT be loaded!" % file_path)
                    return
                
                ldata[ldata > (self.maxPeopleRange + 1)] = 0.0
                locs = np.reshape(ldata, (-1, int(len(ldata[0])/2), 2))
                print("localization data shape:", locs.shape)
                
                # NORMALIZATION
                if norm_type == 1: # [0,1]
                    
                    if polar == False:
                        print("Normalizing cartesian loc coords into range [0,1]...")
                        # regular normalization of cartesian xy [-11,11] to [0,1]
                        locs = self.normalize(locs, -(self.maxPeopleRange + 1), ((self.maxPeopleRange + 1)*2))
                        # zero position (robot place) will take value 0.5
                        # let's try to change 0.5 by 0
                        #locs[locs == 0.5] = 0.0

                    else:
                        # polar r[0,11] to [0,1]
                        print("Normalizing polar loc coords into range [0,1]...")
                        locs[:,:,0] = locs[:,:,0]/(self.maxPeopleRange + 1)
                        # polar phi[-pi,pi] to [0,1]
                        locs[:,:,1] = self.normalize(locs[:,:,1], -np.pi, 2*np.pi)


                elif norm_type == 2: # [-1,1]
                    if polar == False:
                        # normalization of cartesian xy [-11,11] to the range [-1,1]
                        print("Normalizing cartesian loc coords into range [-1,1]...")
                        locs = self.normalizeCustom(locs, -(self.maxPeopleRange + 1), ((self.maxPeopleRange + 1)*2), -1, 1)
                    else:
                        # polar r[0,11] to [-1,1]
                        print("Normalizing polar loc coords into range [-1,1]...")
                        locs[:,:,0] = self.normalizeCustom(locs[:,:,0], 0, (self.maxPeopleRange + 1), -1, 1)
                        # polar phi[-pi,pi] to [-1,1]
                        locs[:,:,1] = self.normalizeCustom(locs[:,:,1], -np.pi, 2*np.pi, -1, 1)
                #else:
                #    print("NORMALIZATION TYPE (", norm_type, ") NOT ALLOWED! Using [0,1]!!!")
                #    locs = self.normalize(locs, -(self.maxPeopleRange + 1), ((self.maxPeopleRange + 1)*2))
            
                if norm_type > 0:
                    if polar == True:
                        print("Localization data (r,phi) has been normalized", end='')
                    else:
                        print("Localization data (x,y) has been normalized", end='')
                    if norm_type == 2:
                        print("in the range [-1,1]")
                    else:
                        print("in the range [-0,1]")
                print("max x norm:", np.max(locs[:,:,0]), "min x norm:", np.min(locs[:,:,0]))
                print("max y norm:", np.max(locs[:,:,1]), "min y norm:", np.min(locs[:,:,1]))
                #print("ldata[0] normalized:", locs[0])


                # accumulate the data
                if c == 0:
                    class_labels = classes
                    loc_labels = locs
                else:
                    class_labels = np.concatenate((class_labels, classes), axis=0)
                    loc_labels = np.concatenate((loc_labels, locs), axis=0)
                c+=1

        return class_labels, loc_labels

            

    def normalize(self, x, xmin, range):
        return ((x-xmin)/range)


    def normalizeCustom(self, x, xmin, range, a, b):
        return (a + (((x-xmin)*(b-a))/range))





    def load_csv_file(self, file_path, npoints=720, skiph=1):
        data = []
        dt = np.dtype([('id', np.int), ('timestamp', np.float64), ('scan', np.float32, (npoints,))])
        try:
            data = np.genfromtxt(file_path, dtype=dt, delimiter=',', skip_header=skiph) # delimiter=',',
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



    def load_data(self, path, nranges, type=0, norm=True):

        # type = 0, scan range data
        # type = 1, binary label
        # type = 2, non-binary label

        ids=[]
        timestamps=[]
        ranges=[]
        c = 0
        for file in sorted(os.listdir(path)):

            print("loading file:", file)
            file_path = os.path.join(path, file)

            if file.endswith('.csv') or file.endswith('.txt'):
                id, timestamp, scan = self.load_csv_file(file_path, npoints=nranges)

            elif file.endswith('.npy'):
                id, timestamp, scan = self.load_numpy_file(file_path)

            if(type==0): # scan range
                # cut far values and normalize the ranges
                scan[np.isinf(scan)] = self.maxPeopleRange + 1
                # we check if there are nan values just in case
                scan[np.isnan(scan)] = self.maxPeopleRange + 1
                scan[scan > self.maxPeopleRange] = self.maxPeopleRange + 1
                #Normalize
                if norm == True:
                    scan = scan/(self.maxPeopleRange + 1)
                scan = scan.astype(np.float32)
                if np.any(np.isnan(scan)):
                    print('Load_data. There is nan values in the scan!!!!!!!!')

            elif (type==1): # binary label
                scan = scan.astype(np.uint8)
            
            else:   # non-binary label
                scan = scan.astype(np.float32)

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



    def join_data(self, x_data_path, y_data_path, nranges, binary_label=True, norm=True):

        #x_data_path = os.path.join(directory_path, 'scans')
        print("Loading data from", x_data_path)
        x_ids, x_ts, x_ranges = self.load_data(x_data_path, nranges, type=0, norm=norm)
        print("Scan ranges has been normalized")
        print('x_ranges.shape:', x_ranges.shape)
        #x_data = np.array(x_ranges, dtype=float)
        #x_data = x_ranges.astype(float)
        x_data = x_ranges

        #y_data_path = os.path.join(directory_path, 'labels')
        print("Loading data from", y_data_path)
        if binary_label == True:
            y_ids, y_ts, y_labels = self.load_data(y_data_path, nranges, type=1, norm=norm)
        else:
            y_ids, y_ts, y_labels = self.load_data(y_data_path, nranges, type=2, norm=norm)
        #y_data = np.array(y_labels, dtype=int)
        y_data = y_labels

        return x_data, y_data



    def join_class_and_loc_data(self, x_data_path, y_data_path, nr, norm_type=1, polar=False):

        print("Loading data from", x_data_path)
        if norm_type > 0:
            x_ids, x_ts, x_ranges = self.load_data(x_data_path, nr, type=0, norm=True)
            print("Scan ranges has been normalized!")
        else:
            x_ids, x_ts, x_ranges = self.load_data(x_data_path, nr, type=0, norm=False)
        
        print('x_ranges.shape:', x_ranges.shape)
        x_data = x_ranges

        print("Loading data from", y_data_path)
        y_class_labels, y_loc_labels = self.load_class_and_loc_labels(y_data_path, norm_type=norm_type, polar=polar)

        return x_data, y_class_labels, y_loc_labels


    
    def join_formatted_data(self, x_data_path, y_data_path, binary_label=True):

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
                if binary_label == True:
                    y_labels = self.load_numpy_formatted_data(file_path, type=2)
                else:
                    y_labels = self.load_numpy_formatted_data(file_path, type=1)
                    
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



    def save_npy(self, data, save_name):
        np.save(save_name, data)
        



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
        # and flip the scan data
        for file in sorted(os.listdir(path)):
            if file.endswith('_scans.csv'):
                print("Opening file:", file)
                file_path = os.path.join(path, file)
                ids, timestamps, scans = self.load_csv_file(file_path)

                scans[np.isinf(scans)] = self.drow_maxRange  
                scans[scans > self.drow_maxRange] = self.drow_maxRange
                scans = np.round(scans, decimals=3)

                # FROG laser scan starts from the right side.
                # DROW laser scan starts from the left side.
                # So we flip the scan array
                scans = np.flip(scans, 1)

                filename = str(file).replace('_scans.csv', '_drow.csv')
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

                filename = str(file).replace('_circles.csv', '_drow.wp')
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
                    # FROG coordinates: x->forward, y->left
                    # DROW coordinates: x->forward, y->right (laser upsidedown?)
                    # So, we negate the y coordinate to comply with DROW laser frame
                    polar = np.round(self.xy_to_rphi(p['x'], -p['y']), decimals=3)
                    if p[0] <= (self.maxPeopleRange + 1):
                        circ_row.append(list(polar))

            polar_cir.append(circ_row)

        return polar_cir





    def drow_to_frog(self, path, store_dir, upsampling=False):
        store_path = os.path.join(path, store_dir)
        try:
            os.mkdir(store_path)
        except FileExistsError:
            print ("Directory already exists. Continue...")
        except OSError:
            print ("Creation of the directory %s failed!" % store_path)

        for file in sorted(os.listdir(path)):

            # only people for the moment
            if file.endswith('.wp'):
                print("loading file:", file)
                pfile_path = os.path.join(path, file)
                seqs, wp_dets = self.load_drow_people_dets(pfile_path)

                # Now take the scans according to the detection sequence numbers
                sfile_path = os.path.join(path, str(file).replace('.wp', '.csv'))
                scans, timestamps = self.load_drow_scan(sfile_path, seqs)
                # NOTE:
                # FROG laser scan starts from the right side.
                # DROW laser scan starts from the left side.
                # So we flip the scan array
                # That was said in the drow documentation, but empirically
                # does not seem to be that way. 
                #frog_scans = scans # np.flip(scans, 1)
                # Write the new scan file
                scan_filename = str(file).replace('.wp', '_frog_scans.csv')
                scan_path = os.path.join(store_path, scan_filename)
                frog_scans = self.drow_scan_to_frog(scan_path, seqs, scans, timestamps, upsampling)

                # Circles file
                circles = self.drow_rphi_to_frog_circles(wp_dets)
                cir_filename = str(file).replace('.wp', '_frog_circles.csv')
                cir_path = os.path.join(store_path, cir_filename)
                self.circles_to_frog_file(seqs, timestamps, circles, cir_path)

                # labels file
                label_filename = str(file).replace('.wp', '_frog_labels.csv')
                label_path = os.path.join(store_path, label_filename)
                self.generate_frog_labels(seqs, timestamps, frog_scans, circles, label_path) #, generate_images=True, max_imgs=50




    def load_drow_people_dets(self, wpf):
        """
        load the people detections of a file (*.wp)
        of the drow dataset
        """
        seqs, dets = [], []
        with open(wpf) as f:
            for line in f:
                seq, tail = line.split(',', 1)
                people = json.loads(tail)
                c = 0
                newp = []
                if len(people) > 0:
                    for person in people:
                        # remove detections farther than max detection dist for learning
                        if person[0] <= self.maxPeopleRange:
                            newp.append(person)
                seqs.append(int(seq))
                dets.append(newp)
        return seqs, dets


    def load_drow_detections(wcf, waf, wpf):
        def _load(fname):
            seqs, dets = [], []
            with open(fname) as f:
                for line in f:
                    seq, tail = line.split(',', 1)
                    seqs.append(int(seq))
                    dets.append(json.loads(tail))
            return seqs, dets

        print("Loading wcf detections: %s ..." % wcf)
        s1, wcs = _load(wcf)
        print("Loading waf detections: %s ..." % waf)
        s2, was = _load(waf)
        print("Loading wpf detections: %s ..." % wpf)
        s3, wps = _load(wpf)

        seqs = s1
        # [seq, [wheelchair pos], [walking pos], [person pos]]
        dets = [*zip(s1, wcs, was, wps)]
        #print(dets)
        return seqs, dets


    
    def load_drow_scan(self, scanfile, seqs):
        """
        Generates a filtered scan list based on the detection sequence numbers
        """

        print("Loading input scan: %s ..." % scanfile)
        data = np.genfromtxt(scanfile, delimiter=",")
        scan = data[:,2:].astype(np.float32)
        sqs = data[:,0].astype(np.uint32)
        tms = data[:,1].astype(np.float32)
        #scan[scan >= self.drow_maxRange] = self.maxRange + 1 #replace 29.69 by 61.0
        print("Sequences to take: %i ..." % len(seqs))
        
        indexes = [idx for idx, sq in enumerate(sqs) if sq in seqs]
        scans = [s for idx, s in enumerate(scan) if idx in indexes]
        # for s in scans:
        #     sxd, syd = self.scan_to_xy(s, self.maxPeopleRange, self.drow_laserFoV, frog=True)
        #     #plt.subplots(2)
        #     plt.scatter(sxd, syd, color='blue')
        #     plt.grid()
        #     titled = "DROW:" + str(len(sxd))
        #     plt.ylabel(titled)
        #     plt.show()
        scans = np.array(scans)
        scans = np.round(scans, decimals=3)
        timestamps = [t for idx, t in enumerate(tms) if idx in indexes]
        timestamps = np.array(timestamps)
        return scans, timestamps



    def format_drowscan_to_frog(self, scans, fill=True):
        
        # Drow angle increment = 0.5, frog and learning = 0.25
        basic_ranges = [self.maxRange] * self.nPoints   #[self.maxPeopleRange + 1]
        nr2 = self.drow_nPoints*2 

        frog_scans = []

        if self.drow_laserFoV > self.laserFoV:
            print("Drow FoV > frog FoV!")
            print("We must discard points in the input scan outside our FoV!")
            skip = int((self.drow_nPoints - (self.nPoints/2))/2)
            print("skipping a total of", skip*2, "points from input scan")
                
            for x in scans:
                d = x[skip:-skip]
                a = basic_ranges.copy() #[1.0] * self.nPoints
                i = 0
                for data in d:
                    a[i] = data
                    if fill == True:
                        a[i+1] = data
                    i+=2
                frog_scans.append(a)
            
        else: 
            print("Drow FoV <= frog FoV!")
            #points to skip at the begining and at the end  
            skip = int((self.nPoints - self.drow_nPoints)/2)
            print("skipping of filling a total of", skip*2, "points from frog scan")
            for x in scans:
                d = basic_ranges.copy()
                a = [self.maxRange] * nr2
                i = 0
                if fill == True:
                    for data in x:
                        a[i]=data
                        a[i+1]=data
                        i+=2
                else:
                    ii = 0
                    for i in range(len(a)):
                        if (i%2 == 0):
                            a[i]=x[ii]
                            ii += 1

                d[skip:-skip] = a
                frog_scans.append(d)
                    
        frog_scans = np.asarray(frog_scans, dtype=np.float32)
        return frog_scans



    def drow_scan_to_frog(self, store_file, seqs, scans, timestamps, upsampling=True):

        # Transform the scan to the network format:
        # angle_increment 0.25º, FoV 180º, points=720
        frog_scans = self.format_drowscan_to_frog(scans, fill=upsampling)
        # for i, data in enumerate(frog_scans):
            
        #     sxd, syd = self.scan_to_xy(scans[i], self.maxPeopleRange, self.drow_laserFoV, frog=False)
        #     sxf, syf = self.scan_to_xy(data, self.maxPeopleRange, self.frog_laserFoV, frog=False)

        #     fig, axs = plt.subplots(2)
        #     axs[0].scatter(sxd, syd, color='blue')
        #     axs[0].grid()
        #     titled = "DROW:" + str(len(sxd))
        #     axs[0].set_ylabel(titled)
        #     axs[1].scatter(sxf, syf, color='red')
        #     axs[1].grid()
        #     titlef = "FROG:" + str(len(sxf))
        #     axs[1].set_ylabel(titlef)
        #     #end_file = "_" + str(i) + ".png"
        #     #img_filename = str(store_file).replace('.csv', end_file)
        #     #plt.savefig(img_filename)
        #     plt.show()
            

        with open(store_file, "w") as outfile: 
            csv_columns = ['id','timestamp','ranges']
            writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
            writer.writerow(csv_columns)
            for i, data in enumerate(frog_scans):
                row = []
                row.append(int(seqs[i]))
                row.append(timestamps[i]) 
                for item in data:
                    row.append(item)
                writer.writerow(row)

        print('Raw scan file created: ', store_file)
        return frog_scans



    def drow_rphi_to_frog_circles(self, wp_dets):

        circles = []
        r = 0.35
        t = 1
        id = 1
        for people in wp_dets:
            row_circles = []
            if len(people) > 0:
                for p in people:
                    # if the person is outside the net FoV,
                    # we do not add him
                    if(abs(p[1]) <= self.laserFoV/2.0):
                        x, y = self.rphi_to_xy(p[0], p[1])
                        x = np.round(x, decimals=2)
                        y = np.round(y, decimals=2)
                        row_circles.append({"idp": id, "x": x, "y": y, "r": r, "type": t}) 
                        id += 1
            circles.append(row_circles)
        return circles


    def circles_to_frog_file(self, seqs, timestamps, circles, out_file):

        people = []
        for s, tm, cir in zip(seqs, timestamps, circles):
            people.append({"id": s, "timestamp": tm, "circles": cir})

        with open(out_file, "w") as outfile: 
            csv_columns = ['id','timestamp','circles']
            writer = csv.DictWriter(outfile, fieldnames=csv_columns, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
            writer.writeheader()
            for data in people:
                writer.writerow(data)

        print('Circles file created: ', out_file)



    def generate_frog_labels(self, seqs, timestamps, scans, circles, store_file, generate_images=False, max_imgs=None):

        labels = []
        img_counter = 0
        if max_imgs is None:
            max_imgs = len(seqs)

        for s, seq, cs in zip(scans, seqs, circles):
            
            sx, sy = self.scan_to_xy(s, self.maxPeopleRange, self.laserFoV, frog=True) 
            scanxy = [*zip(sx, sy)]
            scanxy = np.array(scanxy)

            # crear un vector de info de los clusters
            # para cada punto del scan guardar:
            # p_info = [dist al circle más cercano, clase]
            # clases:0-none, 1-person
            # Con esa info, generar los archivos de label con cada fila:
            # [seq, [clase_p0, clase_p1, clase_p2, ..., clase_p450]]
            p_info = [self.maxRange, 0]
            clusters = []
            for i in range(len(scanxy)):
                clusters.append(p_info)

            # iterate for the pedestrian centers
            for ip, p in enumerate(cs):
                # iterate for each range and fill the clusters vector
                for i in range(len(scanxy)):

                    px = scanxy[i][0] - p['x']
                    py = scanxy[i][1] - p['y']
                    pd = math.sqrt(px*px + py*py)
                    if pd <= (p['r']+0.15) and pd < clusters[i][0]:
                        clusters[i] = [pd, ip+1] 
                        
            array = np.asarray(clusters)
            label = array[:,1].astype(np.uint8)
            #labels.append(array[:,1].astype(np.uint8))
            l2 = np.array(label, copy=True)
            for ic in range(1, len(cs)+1):
                #index_mask = label == ic
                indices = [i for i, l in enumerate(label) if l == ic]
                for j in range(len(indices)-1):
                    if indices[j+1]-indices[j] > 1:
                        l2[indices[j]:indices[j+1]] = ic
                    
            # we need to insert an empty range between people that are very close
            # so we can distinguish between only one person or more
            for ib in range(len(l2)-1):
                for ic in range(1, len(cs)+1):
                    if ic == 1:
                        if l2[ib]==ic and l2[ib+1]>l2[ib]:
                            l2[ib+1] = 0

                    elif l2[ib]==ic and (l2[ib+1]>l2[ib] or (l2[ib+1]<l2[ib] and l2[ib+1]!=0)):
                        l2[ib+1] = 0

            # Put all the identifiers to 1
            l2[l2>1] = 1
            l2[0] = 0
            l2[-1] = 0
            labels.append(l2)

            # Fill the label image
            if generate_images and img_counter < max_imgs:
                
                if(len(cs) >= 1): 
                    classes = l2
                    sx_label = np.asarray([xi for idx, xi in enumerate(sx) if classes[idx] == 1], dtype=np.float32)
                    sy_label = np.asarray([yi for idy, yi in enumerate(sy) if classes[idy] == 1], dtype=np.float32)

                    xaxis = list(np.array(np.linspace(0, 719, 720), dtype=np.int32))
                    fig, axs = plt.subplots(2)
                    #fig.suptitle('Vertically stacked subplots')
                    axs[0].plot(xaxis, l2)
                    #axs[0].scatter(xaxis, center_ranges, color='red', marker='x')
                    axs[0].set_ylim([0, 1.05])
                    axs[0].set_ylabel("drow people binary")
                    axs[0].grid()

                    axs[1].scatter(sx, sy, color='blue')
                    axs[1].grid()
                    axs[1].scatter(sx_label, sy_label, color='green', marker='x')
                    axs[1].scatter([p['x'] for p in cs], [p['y'] for p in cs], color='red', marker='x')
                    end_file = "_" + str(seq) + ".png"
                    img_filename = str(store_file).replace('.csv', end_file)
                    plt.savefig(img_filename)
                    #plt.show()
                    img_counter += 1

                # label_img = np.zeros((self.img_height, self.img_width), dtype=int)
                # classes = array[:,1].astype(np.uint8)
                
                # sx_label = np.asarray([xi for idx, xi in enumerate(sx) if classes[idx] == 1], dtype=np.float32)
                # sy_label = np.asarray([yi for idy, yi in enumerate(sy) if classes[idy] == 1], dtype=np.float32)
                # px_label, py_label = self.worldToImg(sx_label, sy_label)
                # label_img[py_label, px_label] = 255

                # end_file = "_" + str(seq) + ".jpg"
                # img_filename = str(store_file).replace('.csv', end_file)
                # print("Saving label image: %s" % img_filename)
                # cv2.imwrite(img_filename, label_img)

                # input_img = np.zeros((self.img_height, self.img_width), dtype=int)
                # px, py = self.worldToImg(sx, sy)
                # input_img[py, px] = 255
                # end_file = "scan_" + str(seq) + ".jpg"
                # input_filename = str(store_file).replace('labels.csv', end_file)
                # print("Saving  scan image: %s" % input_filename)
                # cv2.imwrite(input_filename, input_img)
                # img_counter += 1


        # write the label file
        with open(store_file, "w") as outfile: 
            csv_columns = ['id','timestamp','label']
            writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
            writer.writerow(csv_columns)
            for i, data in enumerate(labels):
                row = []
                row.append(int(seqs[i]))
                row.append(timestamps[i]) 
                for item in data:
                    row.append(item)
                writer.writerow(row)

        print('Label file created: ', store_file)




    def circles_to_class_and_loc_labels(self, circles_path, labels_path, max_people, order_type=1, nranges=720, ares=0.25, polar=True):

        #for polar
        def _rangeOrder(e):
            return e[0]

        # order by polar angle
        def _angleOrder(e):
            return e[1]

        # cartesian
        def _elementOrder(e):
            return np.hypot(e[0], e[1])


        try:
            os.mkdir(labels_path)
        except FileExistsError:
            print ("Labels directory already exists. Continue...")
        except OSError:
            print ("Creation of the directory %s failed!" % labels_path)

        for file in sorted(os.listdir(circles_path)):

            print("loading circles file:", file)
            circles_file = os.path.join(circles_path, file)
            idcir, timestampcir, circles = self.load_circles_csv_file(circles_file)

            classfile = str(file).replace("_circles.csv", "_class_labels.csv")
            locfile = str(file).replace("_circles.csv", "_loc_labels.csv")
            #print("Gererating new label files", classfile, "and", locfile, "...")

            class_labels = []
            loc_labels = []

            # Order by the closest people at the beggining of the array 
            if order_type == 1:

                for cir in circles:
                    class_label = np.zeros(max_people, dtype=np.int32)
                    loc_label = np.zeros((max_people, 2), dtype=np.float32)
                    pdetected = len(cir['circles']) if len(cir['circles']) <= max_people else max_people
                    if pdetected > 0:
                        class_label[:pdetected] = 1
                        people = []
                        for i, c in enumerate(cir['circles']):
                            if polar == True:
                                rphi = self.xy_to_rphi(c['x'], c['y'])
                                people.append(rphi)
                            else:
                                people.append([c['x'], c['y']])
                        #we order the people in ascending range 
                        if polar == True:
                            people.sort(key=_rangeOrder)
                        else:   
                            people.sort(key=_elementOrder)

                        for i in range(pdetected):
                            loc_label[i]=people[i]
                        
                    class_labels.append(class_label)
                    loc_labels.append(loc_label)


            # order people according to the scan order
            else:
                binsize = np.floor(nranges/max_people)
                for cir in circles:
                    class_label = np.zeros(max_people, dtype=np.int32)
                    loc_label = np.zeros((max_people, 2), dtype=np.float32)
                    pdetected = len(cir['circles']) if len(cir['circles']) <= max_people else max_people
                    count = 0
                    if pdetected > 0:
                        #people_cartesian = []
                        people_polar = []
                        for i, c in enumerate(cir['circles']):
                            rphi = self.xy_to_rphi(c['x'], c['y'])
                            people_polar.append(rphi)
                            count+=1
                            if count == pdetected:
                                break
                        people_polar.sort(key=_angleOrder)
                        for p in people_polar:
                            index = np.rint((np.pi/2 + p[1])/np.radians(ares)) # - 1
                            bin = int(np.floor(index/binsize))
                            print("index:", index, " bin:", bin)
                            if bin == max_people:
                                bin = bin-1
                                print("\tbin fix:", bin)
                            class_label[bin] = 1
                            if polar == True:
                                loc_label[bin] = [p[0], p[1]]
                            else:
                                x, y = self.rphi_to_xy(p[0], p[1])
                                loc_label[bin] = [x, y]

                    class_labels.append(class_label)
                    loc_labels.append(loc_label)


            # store the new class labels file
            newl_file = os.path.join(labels_path, classfile)
            # write the label file
            with open(newl_file, "w") as outfile: 
                #csv_columns = ['id','timestamp','label']
                writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                #writer.writerow(csv_columns)
                for i, cls in enumerate(class_labels):
                    row = []
                    for item in cls:
                        row.append(int(item))
                    writer.writerow(row)
            print('New Label file created: ', newl_file)

            # store the new localization labels file
            newl_file = os.path.join(labels_path, locfile)
            # write the label file
            with open(newl_file, "w") as outfile: 
                #csv_columns = ['id','timestamp','label']
                writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                #writer.writerow(csv_columns)
                for i, l in enumerate(loc_labels):
                    row = []
                    for item in l:
                        row.append(np.round(item[0], decimals=3))
                        row.append(np.round(item[1], decimals=3))
                    writer.writerow(row)
            print('New Label file created: ', newl_file)

            







    def binary_to_peoplebin_label(self, scans_path, circles_path, labels_path, gaussianlabel=False):

        for file in sorted(os.listdir(scans_path)):

            new_labels = []

            # load scans
            print("loading scan file:", file)
            scan_file = os.path.join(scans_path, file)
            idscan, timestampscan, scan = self.load_csv_file(scan_file)
            la = self.laser_angles(len(scan[0]))


            # load circles
            cfile = str(file).replace("_scans.csv", "_circles.csv")
            print("loading circles file:", cfile)
            circles_file = os.path.join(circles_path, cfile)
            idcir, timestampcir, circles = self.load_circles_csv_file(circles_file)

            
            lfile = str(file).replace("_scans.csv", "_binpeople_labels.csv")
            if gaussianlabel == True:
                lfile = str(file).replace("_scans.csv", "_regpeople_labels.csv")

            print("Gererating new label file:", lfile, "...")
            for i, s in enumerate(scan):
                sx, sy = self.scan_to_xy(s, self.maxPeopleRange, self.frog_laserFoV) 
                scanxy = [*zip(sx, sy)]
                scanxy = np.array(scanxy)

                p_info = [61., 0]
                clusters = []
                for j in range(len(scanxy)):
                    clusters.append(p_info)

                l2 = np.zeros(len(s), dtype=np.int32)
                l3 = np.zeros(len(s), dtype=np.float32)
                circ = circles[i]
                if len(circ['circles']) > 0:
                    for ci, c in enumerate(circ['circles']):
                        #print("person", ci+1)
                        #polar = self.xy_to_rphi(c[0],c[1])            
                        for idx, sp in enumerate(scanxy):
                            px = sp[0] - c['x']
                            py = sp[1] - c['y']
                            dist = math.sqrt(px*px + py*py)
                            #print("Distance to person", ci+1, ":", dist)
                            if dist <= float(c['r']) and dist < clusters[idx][0]:
                                #print("Distance to person", ci+1, ":", dist)
                                clusters[idx] = [dist, ci+1]
                
                    array = np.asarray(clusters)
                    # store the label array
                    label = array[:,1].astype(np.int32)
                    #print(label)
                    l2 = np.array(label, copy=True)
                    for ic in range(1, len(circ['circles'])+1):
                        #index_mask = label == ic
                        indices = [i for i, l in enumerate(label) if l == ic]
                        for j in range(len(indices)-1):
                            if indices[j+1]-indices[j] > 1:
                                l2[indices[j]:indices[j+1]] = ic
                    
                    # we need to insert an empty range between people that are very close
                    # so we can distinguish between only one person or more
                    for ib in range(len(l2)-1):
                        for ic in range(1, len(circ['circles'])+1):
                            if ic == 1:
                                if l2[ib]==ic and l2[ib+1]>l2[ib]:
                                    l2[ib+1] = 0

                            elif l2[ib]==ic and (l2[ib+1]>l2[ib] or (l2[ib+1]<l2[ib] and l2[ib+1]!=0)):
                                l2[ib+1] = 0

                    # Put all the identifiers to 1
                    l2[l2>1] = 1
                    l2[0] = 0
                    l2[-1] = 0
                   
                    if gaussianlabel==True:
                        l3 = np.array(l2, dtype=np.float32, copy=True)
                        l3[0] = 0.0
                        l3[-1] = 0.0

                        #transform to a set of gaussian
                        prev_cl = 0
                        init = -1
                        end = -1
                        for id, cl in enumerate(l3):
                            if cl == 1 and prev_cl == 0:
                                init=id
                            if cl == 0 and prev_cl == 1:
                                end=id-1
                    
                            prev_cl = int(l3[id])

                            if(init!=-1 and end!=-1 and (end-init)>1):
                                #print("Person found! init:", init, "end:", end, ". len:", (end-init)+1)
                                center=[]
                                #TODO: count the ranges and split it in two
                                # take into account we need one or two central ranges.
                                # divide 0.6 between the number of slots of each side
                                even = True if((end - init)+1)%2==0 else False
                                # two central ranges
                                if even == True:
                                    center.append(init + ((end - init)+1)/2 - 1)
                                    center.append(init + ((end - init)+1)/2)
                                    #print("Even! centers:", center)
                                    nbins = ((end - init)+1)/2 - 1
                                    inc = 0.6/nbins
                                else:
                                    center.append(init + (end - init)/2)
                                    #print("Odd! centers:", center)
                                    nbins = (end - init)/2
                                    inc = 0.6/nbins

                                for icount in range(1, int(nbins)+1):
                                    l3[int(center[-1])+icount] = self.gaussian(inc*icount)
                                    l3[int(center[0])-icount] =  self.gaussian(inc*icount)

                                                
                                init = -1
                                end = -1
                        
                
                if gaussianlabel==True:
                    new_labels.append(l3)
                else:
                    new_labels.append(l2)


            # store the new labels file
            newl_file = os.path.join(labels_path, lfile)
            # write the label file
            with open(newl_file, "w") as outfile: 
                csv_columns = ['id','timestamp','label']
                writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                writer.writerow(csv_columns)
                for i, data in enumerate(new_labels):
                    row = []
                    row.append(int(idscan[i]))
                    row.append(timestampscan[i]) 
                    for item in data:
                        if gaussianlabel==True:
                            row.append(np.round(item, decimals=3))
                        else:
                            row.append(int(item))
                    writer.writerow(row)

            print('New Label file created: ', newl_file)

        





    def binary_to_gaussian_label(self, scans_path, circles_path, labels_path):

        ming = 0.4
        # join the scans
        # If the file naming is correct, 'sorted' gives the correct order 
        for file in sorted(os.listdir(scans_path)):

            new_labels = []

            print("loading scan file:", file)
            scan_file = os.path.join(scans_path, file)

            idscan, timestampscan, scan = self.load_csv_file(scan_file)

            labelfile = str(file).replace("_scans.csv", "_labels.csv")
            label_file = os.path.join(labels_path, labelfile)
            idscan, timestampscan, label = self.load_csv_file(label_file)
            label = np.array(label, dtype=np.int8)

            la = self.laser_angles(len(scan[0]))

            cfile = str(file).replace("_scans.csv", "_circles.csv")
            print("loading circles file:", cfile)
            circles_file = os.path.join(circles_path, cfile)
            idcir, timestampcir, circles = self.load_circles_csv_file(circles_file)

            
            lfile = str(file).replace("_scans.csv", "_gauss_labels.csv")
            print("Gererating new label file:", lfile, "...")
            for i, s in enumerate(scan):
                sx, sy = self.scan_to_xy(s, self.maxPeopleRange, self.frog_laserFoV) 
                scanxy = [*zip(sx, sy)]
                scanxy = np.array(scanxy)

                # array with the distances to the centers and Gaussian values
                p_info = [self.maxRange+1, 0.]
                gau_ranges = []
                for j in range(len(scanxy)):
                    gau_ranges.append(p_info)

                #print("gaussian:", gau_ranges)

                circ = circles[i]
                if len(circ['circles']) > 0:
                    #print("scan", i+1, "has", len(circ['circles']), "people")
                    # iterate for the pedestrian centers
                    for p in circ['circles']:
                        polar = self.xy_to_rphi(p['x'],p['y'])
                        min = 2*math.pi
                        index = -1
                        for i in range(len(la)):
                            angdiff = abs(polar[1] - la[i])
                            if angdiff < min:
                                min = angdiff
                                index = i 

                        # center range to value 1
                        gau_ranges[index] = [gau_ranges[index][0], 1.0]
                        
                        # iterate for each range and fill the gua_ranges vector
                        for i in range(len(scanxy)):
                            px = scanxy[i][0] - p['x']
                            py = scanxy[i][1] - p['y']
                            pdist = math.sqrt(px*px + py*py)
                            gv = self.gaussian(pdist)
                            if gv >= ming and pdist < gau_ranges[i][0] and gau_ranges[i][1] < 1.0:
                                gau_ranges[i] = [pdist, gv] 
                            elif pdist < gau_ranges[i][0]:
                                gau_ranges[i] = [pdist, gau_ranges[i][1]]

                        #Try to fill the gaussian value of ranges in the vecinity of a center that are far.
                        # check 20 ranges to each side of the center range, and interpolate values
                        # lower_idx = (index-20) if (index-20)>0 else 0 
                        # upper_idx = (index+20) if (index+20)<(len(gau_ranges)-1) else (len(gau_ranges)-1)
                        # for i in range(index, lower_idx, -1):
                        #     gvalue = gau_ranges[i][1]
                        #     if gvalue == 0.0:


                         

                    #print("Gauranges:", gau_ranges)

                # new label array
                array = np.array(gau_ranges, copy=True)
                label = array[:,1].astype(np.float32)
                new_labels.append(label)

            # store the new labels file
            newl_file = os.path.join(labels_path, lfile)
            # write the label file
            with open(newl_file, "w") as outfile: 
                csv_columns = ['id','timestamp','label']
                writer = csv.writer(outfile, delimiter=',', quotechar='', quoting=csv.QUOTE_NONE, escapechar=' ')
                writer.writerow(csv_columns)
                for i, data in enumerate(new_labels):
                    row = []
                    row.append(int(idscan[i]))
                    row.append(timestampscan[i]) 
                    labs = np.round(data, decimals=3)
                    for item in labs:
                        row.append(item)
                    writer.writerow(row)

            print('New Label file created: ', newl_file)


                



    def gaussian(self, x, mu=0., sig=0.39894228):
        return np.exp(-np.power(x -mu, 2.) /( 2* np.power(sig, 2.)))
        


        




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

        #try:
        bag_name = os.path.join(path, 'complete.bag')
        with rosbag.Bag(bag_name, 'w') as outbag:
            t = 0
            for i, ranges in enumerate(scans):
                
                #if(i == 0):
                #    t = scan_ts[i]
                #else:
                #    t += 0.025
                scan_msg.ranges = ranges
                ts = rospy.Time.from_sec(scan_ts[i]) #t
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

        # except:
        #     print("-------------------------------------")
        #     print('ERROR!!! Bag file could not be created!')
        #     print("-------------------------------------")
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


    # ranges data (x_data) already normalized!
    # angle_inc_deg is in degrees
    # This only allows angle increments of 0.25 or 0.5 degrees
    def format_data_for_learning(self, x_data, y_data, nr, angle_inc_deg, fill=True, data_normalized=True, binary_label=True):

        print("format_data_for_learning:")
        xdata = []
        ydata = []
        if data_normalized == False:
            # Normalize data
            print("Normalizing data..")
            x_data[np.isinf(x_data)] = self.maxPeopleRange + 1
            x_data[x_data > self.maxPeopleRange] = self.maxPeopleRange + 1
            x_data = x_data/(self.maxPeopleRange + 1)


        basic_ranges = [1.0] * self.nPoints #[self.maxPeopleRange + 1] * self.nPoints
        if binary_label == True:
            basic_label = [0] * self.nPoints
        else:
            basic_label = [0.0] * self.nPoints
        numr = len(x_data[0])  # must be equal to nr
        inputFoV = (numr - 1) * np.radians(angle_inc_deg)
        
        # same angle icrement
        print("Input data format. npoints:", numr, "angle_inc:", angle_inc_deg, 'degrees')
        if abs(np.radians(angle_inc_deg) - self.laserIncrement) < 0.001:

            # same number of ranges
            if numr == self.nPoints:
                print("Data is already in the correct format!")
                if data_normalized == False:
                    # denormalize
                    print("DENormalizing data..")
                    x_data = x_data * (self.maxPeopleRange + 1)
                return x_data, y_data

            # different FoV - self.laserFoV > inputFoV 
            else:
                print("Same angle increment! Different FoV!")
                #points to skip at the begining and at the end
                skip = int((self.nPoints - numr)/2)
                    
                for x, y in zip(x_data, y_data):
                    d = basic_ranges.copy()
                    d[skip:-skip] = x
                    xdata.append(d)
                    l = basic_label.copy()
                    l[skip:-skip] = y
                    ydata.append(l)

                xdata = np.asarray(xdata, dtype=np.float32)
                if binary_label == True:
                    ydata = np.asarray(ydata, dtype=np.int8)
                else:
                    ydata = np.asarray(ydata, dtype=np.flaot32)

                # denormalize
                if data_normalized == False:
                    print("DENormalizing data..")
                    xdata = xdata * (self.maxPeopleRange + 1)

                return xdata, ydata


        # different angle increment (0.5)
        elif abs(angle_inc_deg - 0.5) < 0.01:
            nr2 = len(x_data[0])*2

            if self.laserFoV > 6.2 and nr2 > self.nPoints:  # 
                print("ERROR! range data can not be transformed into learning format!!!")
                return xdata, ydata

            # laserFoV < inputFov:
            if inputFoV > self.laserFoV:
                print("Different angle increment! and Input FoV > output FoV!")
                # we must discard points in x_data outside our FoV
                skip = int((nr - (self.nPoints/2))/2)
                print("skipping a total of", skip*2, "points from input scan")
                
                for x, y in zip(x_data, y_data):
                    d = x[skip:-skip]
                    a = basic_ranges.copy() #[1.0] * self.nPoints
                    i = 0
                    for data in d:
                        a[i] = data
                        if fill == True:
                            a[i+1] = data
                        i+=2
                    xdata.append(a)
            
                    l = y[skip:-skip]
                    b = basic_label.copy()
                    i = 0
                    for data in l:
                        b[i] = data
                        if fill == True:
                            b[i+1] = data
                        i+=2
                    ydata.append(b)
                    

            # inputFoV <= laserFoV
            else: 
                print("Different angle increment! and Input FoV <= output FoV!")
                #points to skip at the begining and at the end  
                skip = int((self.nPoints - nr)/2)
                for x, y in zip(x_data, y_data):
                    d = basic_ranges.copy()
                    a = [self.maxPeopleRange + 1] * nr2
                    i = 0
                    if fill == True:
                        for data in x:
                            a[i]=data
                            a[i+1]=data
                            i+=2
                    else:
                        ii = 0
                        for i in range(len(a)):
                            if (i%2 == 0):
                                a[i]=x[ii]
                                ii += 1
                    d[skip:-skip] = a
                    xdata.append(d)

                    l = basic_label.copy()
                    b = [0] * nr2
                    i = 0
                    if fill == True:
                        for data in y:
                            b[i]=data
                            b[i+1]=data
                            i+=2
                    else:
                        ii = 0
                        for i in range(len(b)):
                            if (i%2 == 0):
                                b[i]=y[ii]
                                ii += 1
                    l[skip:-skip] = b
                    ydata.append(l)

        xdata = np.asarray(xdata, dtype=np.float32)
        if binary_label == True:
            ydata = np.asarray(ydata, dtype=np.int8)
        else:
            ydata = np.asarray(ydata, dtype=np.flaot32)

        # denormalize
        if data_normalized == False:
            print("DENormalizing data..")
            xdata = xdata * (self.maxPeopleRange + 1)

        return xdata, ydata