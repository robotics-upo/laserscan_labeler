#!/usr/bin/python3
# -*- coding: utf-8 -*- 

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *  
import rospy
import json
import csv
import numpy as np
import math
import matplotlib
import sys, os, random
import signal
import copy
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy.io as sio
import cv2

from bag_loader import ScanBagLoader, ImageBagLoader
from labeling import *

# Handle ctrl-c
signal.signal(signal.SIGINT, signal.SIG_DFL)

#def signal_handler(sig, frame):
#    sys.exit(0)

#signal.signal(signal.SIGINT, signal_handler)



class AppForm(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self, None)
        self.setWindowTitle('LaserScan Labeler')

        self.points = None
        self.dragging_id = None
        self.play_timer = None
        #self.circles = CircleRegionManager()

        self.current_index = 0
        self.init_scan = 0
        self.recording = False

        self.people = peopleManager()

        self.topic = 'scan'
        self.info_label = QLabel('')

        self.show_images = False
        self.img_index = 0

        self.scan_step_ms = 100
        
        self.data = None
        self.create_menu()
        self.create_main_frame()
        self.setChildrenFocusPolicy(Qt.NoFocus)

        self.on_draw()


    def create_menu(self):        
        self.file_menu = self.menuBar().addMenu("&File")
        
        save_file_action = self.create_action("&Save bag file...",
            shortcut="Ctrl+S", slot=self.save_as, 
            tip="Save a labeled bag file")
        # save_as_action = self.create_action("&Save As...",
        #     shortcut="Ctrl+Shift+S", slot=self.save, 
        #     tip="Save a label file to another path")
        load_file_action = self.create_action("&Open scan bag...",
            shortcut="Ctrl+O", slot=self.open, 
            tip="Open a bag file")
        load_video_action = self.create_action("&Open image bag...",
            shortcut="Ctrl+I", slot=self.openImage, 
            tip="Open an image bag file")
        export_action = self.create_action("&Export...",
            shortcut="Ctrl+E", slot=self.export, 
            tip="Export labeled data")
        quit_action = self.create_action("&Quit", slot=self.close, 
            shortcut="Ctrl+Q", tip="Close the application")
        
        self.add_actions(self.file_menu, 
            (load_file_action, None, load_video_action, None, save_file_action, None, export_action, None, quit_action)) #save_file_action, save_as_action, None,


    # def save(self):
    #     print("Save!")
    #     if self.path is None:
    #         self.save_as()
    #     else:
    #         self.save_file(self.path)

    def save_as(self):
        print("Save as!")
        file_choices = "BAG (*.bag)"
        path = QFileDialog.getSaveFileName(self, 
                        'Save bag file', '', 
                        file_choices)
        #print("path:", path)
        path = path[0]
        if not path.endswith(".bag"):
            path = path + ".bag"
        #self.path = path
        self.save_bag_file(path)
        print(path)


    def save_bag_file(self, path):
        people_data = {'people': []}
        for i in range(self.init_scan, self.spinbox.value()):
            scanp = {'id': i, 'timestamp': self.data.timeStamps[i].to_sec(), 'circles': []}
            for p in self.people.data[i].people:
                xp = (p.x - 0.5) * self.data.range_max*2.0
                yp = (p.y - 0.5) * self.data.range_max*2.0
                rp = p.r * self.data.range_max*2.0 
                #print("------ x:", xp, "y:", yp, "r:", rp, "-----")
                scanp['circles'].append({'idp': p.person_id, 'x': xp, 'y': yp, 'r': rp, 'type': p.type}) 
            people_data['people'].append(scanp)

        scan_topic = self.topicsCB.currentText()
        self.info_label.setStyleSheet("color : orange")
        self.info_label.setText("Saving bag file...")
        self.data.save_bag(people_data, scan_topic, path)
        self.info_label.setStyleSheet("color : green")
        self.info_label.setText("Bag file saved successfully!")




    # def save_file(self, path):
    #     with gzip.open(path, 'wb') as f:
    #         pickle.dump([path, self.data, self.circles], f)



    def open(self):
        print("Open!")
        file_choices = "BAG (*.bag)" #LSL or BAG (*.lsl *.bag);; LSL (*.lsl);; 
        #path = unicode(QFileDialog.getOpenFileName(self, 
        #                'Open bag or lsl file', '', file_choices))
        path = QFileDialog.getOpenFileName(self, 
                    'Open bag or lsl file', '', file_choices)

        path = path[0]
        print("path: ", path)
        
                
        if path.endswith(".bag"): 
            text = "Loading bag file..."
            self.info_label.setStyleSheet("color : orange")
            self.info_label.setText(text)
            self.data = ScanBagLoader(path)
            topics = self.data.getScanTopics()
            print("length topics: ", len(topics))

            self.img_data = ImageBagLoader(path)
            img_topics = self.img_data.getImageTopics()
            
            if len(topics) == 0:
                text = "NO BAG LOADED!!!"
                self.info_label.setStyleSheet("color : red")
                self.info_label.setText(text)
                self.play_button.setEnabled(False)
                self.prev_button.setEnabled(False)
                self.next_button.setEnabled(False)
            elif len(topics) == 1:
                self.topicsCB.addItem(topics[0])
                self.data.loadData(topics[0])
                text = "BAG LOADED SUCCESFULLY!!!"
                self.info_label.setText(text)
                self.info_label.setStyleSheet("color : green")
                self.play_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
                #self.topicsCB.currentIndexChanged.connect(self.topicsComboChanged)
            else:
                for t in topics:
                    print("scan topic: ", t)
                    self.topicsCB.addItem(t)
                text = "BAG LOADED. NOW, CHOOSE THE DESIRED SCAN TOPIC!!!"
                #self.topicsCB.currentIndexChanged.connect(self.topicsComboChanged)
                self.info_label.setText(text)
                self.info_label.setStyleSheet("color : blue")
                self.play_button.setEnabled(True)
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)

            if len(img_topics) == 1:
                self.imgTopicsCB.addItem(topics[0])
                self.img_data.loadData(topics[0])
                self.show_images = True
            elif len(img_topics)>1:
                for t in img_topics:
                    print("image topic: ", t)
                    self.imgTopicsCB.addItem(t)
                self.show_images = True

            #self.circles.cleanup()
            #self.circles = CircleRegionManager()
            self.people.cleanup()
            self.people = peopleManager()
            # Set the UI elements that depend on data or need resetting
            self.spinbox.setValue(0)
            self.spinbox.setMaximum(self.data.length-1)
            self.ax_p.set_rmax(self.data.range_max)
            self.ax_p.set_rticks(np.arange(0,self.data.range_max+1, 1.0))  # less radial ticks

            total_frames = ' of ' + str(self.data.length) + ' scans'
            self.total_label.setText(total_frames)

            # Open window
            self.show()

            # Re-render everything
            self.on_draw()
        
        else:
            self.info_label.setStyleSheet("color : red")
            self.info_label.setText("No valid bag file")


    def openImage(self):
        print("Open image bag!")
        file_choices = "BAG (*.bag)"
        #path = unicode(QFileDialog.getOpenFileName(self, 
        #                'Open bag or lsl file', '', file_choices))
        path = QFileDialog.getOpenFileName(self, 
                    'Open bag file containing images', '', file_choices)

        path = path[0]
        print("path: ", path)
                
        if path.endswith(".bag"): 
            text = "Loading image bag file..."
            self.info_label.setStyleSheet("color : orange")
            self.info_label.setText(text)
            self.img_data = ImageBagLoader(path)
            topics = self.img_data.getImageTopics()
            print("length topics: ", len(topics))
            
            if len(topics) == 0:
                text = "NO IMAGES FOUND IN THE BAG!!!"
                self.info_label.setText(text)
                self.info_label.setStyleSheet("color : red")
                self.show_images = False
                
            elif len(topics) == 1:
                self.imgTopicsCB.addItem(topics[0])
                self.img_data.loadData(topics[0])
                text = "IMAGE BAG LOADED SUCCESFULLY!!!"
                self.info_label.setText(text)
                self.info_label.setStyleSheet("color : green")
                self.show_images = True
            else:
                for t in topics:
                    print("image topic: ", t)
                    self.imgTopicsCB.addItem(t)
                text = "IMAGE BAG LOADED. NOW, CHOOSE THE DESIRED IMAGE TOPIC!!!"
                #self.topicsCB.currentIndexChanged.connect(self.topicsComboChanged)
                self.info_label.setText(text)
                self.info_label.setStyleSheet("color : blue")
                self.show_images = True

        
            # Re-render everything
            self.on_draw()
        
        else:
            self.info_label.setStyleSheet("color : red")
            self.info_label.setText("No valid image bag file")

        


    def export(self):
        """ Export labeled data as a mat file
        """
        print("export start")

        # Get the save path
        file_choices = "CSV (*.csv);; JSON (*.json);; NUMPY (*.npy);; MAT (*.mat)"
        #path = unicode(QFileDialog.getSaveFileName(self, 
        #                'Export mat', '', 
        #                file_choices))
        path = QFileDialog.getSaveFileName(self, 
                        'Export data', '', 
                        file_choices)
        print("path:", path)
        ext = path[1]
        path = path[0]

        n_scans = self.spinbox.value() - self.init_scan

        self.info_label.setStyleSheet("color : orange")
        self.info_label.setText("Exporting data to file. It may take some minutes...")

        # Get all data into a dict
        scan_data = {'scan_info': {'range_max': self.data.range_max,
                'total_scans': n_scans, #self.data.length,
                'ranges_per_scan': len(self.data.data[0]),
                'angle_min': self.data.angle_min,
                'angle_max': self.data.angle_max,
                'angle_increment': self.data.angle_increment}}       #'theta': self.data.theta.tolist()}
        
        scan_data['scans'] = []
        for i in range(self.init_scan, self.spinbox.value()):
            scan = {'id': i, 'timestamp': self.data.timeStamps[i].to_sec(), 'ranges': self.data.data[i].tolist()}
            scan_data['scans'].append(scan)

        
        label_data = {'labels': []}
        for i in range(self.init_scan, self.spinbox.value()):
            scanxy = self.data.get_cartesian(i)
            labels = self.people.data[i].getLabeledScan(scanxy)
            label = {'id': i, 'timestamp': self.data.timeStamps[i].to_sec(), 'label': labels.tolist()} #self.people.get_classes(self.data, i).tolist()
            label_data['labels'].append(label)


        people_data = {'people': []}
        for i in range(self.init_scan, self.spinbox.value()):
            scanp = {'id': i, 'timestamp': self.data.timeStamps[i].to_sec(), 'circles': []}
            for p in self.people.data[i].people:
                xp = (p.x - 0.5) * self.data.range_max*2.0
                yp = (p.y - 0.5) * self.data.range_max*2.0
                rp = p.r * self.data.range_max*2.0 
                #print("------ x:", xp, "y:", yp, "r:", rp, "-----")
                scanp['circles'].append({'idp': p.person_id, 'x': xp, 'y': yp, 'r': rp, 'type': p.type}) 
            people_data['people'].append(scanp)

        print("exporting data...")

        error = False

        # MAT format
        if ext == 'MAT (*.mat)': 
            #if not path.endswith(".mat"):
            scan_path = path + "_scans.mat"
            sio.savemat(scan_path, scan_data)
            label_path = path + "_labels.mat"
            sio.savemat(label_path, label_data)
            people_path = path + "_circles.mat"
            sio.savemat(people_path, people_data)

        # JSON format
        if ext == 'JSON (*.json)':
            try:
                scan_path = path + "_scans.json"
                with open(scan_path, "w") as outfile: 
                    json.dump(scan_data, outfile, indent = 4)

                label_path = path + "_labels.json"
                with open(label_path, "w") as outfile: 
                    json.dump(label_data, outfile, indent = 4)

                people_path = path + "_circles.json"
                with open(people_path, "w") as outfile: 
                    json.dump(people_data, outfile, indent = 4)
            except:
                text = "JSON file could not be written"
                print(text)
                self.info_label.setStyleSheet("color : red")
                self.info_label.setText(text)
                error = True


        # NUMPY format
        if ext == 'NUMPY (*.npy)':
            try:
                scan_path = path + "_scans.npy"
                data_array = np.array(scan_data['scans'])
                np.save(scan_path, data_array)

                label_path = path + "_labels.npy"
                data_array = np.array(label_data['labels'])
                np.save(label_path, data_array)

                people_path = path + "_circles.npy"
                data_array = np.array(people_data['people'])
                np.save(people_path, data_array)

            except:
                text = "Numpy file could not be written"
                print(text)
                self.info_label.setStyleSheet("color : red")
                self.info_label.setText(text)
                error = True

        # CSV format
        if ext == 'CSV (*.csv)':
            try:
                scan_path = path + "_scans.csv"
                with open(scan_path, "w") as outfile: 
                    csv_columns = ['id','timestamp','ranges']
                    writer = csv.DictWriter(outfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in scan_data['scans']:
                        writer.writerow(data)

                label_path = path + "_labels.csv"
                with open(label_path, "w") as outfile: 
                    csv_columns = ['id','timestamp','label']
                    writer = csv.DictWriter(outfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in label_data['labels']:
                        writer.writerow(data)

                people_path = path + "_circles.csv"
                with open(people_path, "w") as outfile: 
                    csv_columns = ['id','timestamp','circles']
                    writer = csv.DictWriter(outfile, fieldnames=csv_columns)
                    writer.writeheader()
                    for data in people_data['people']:
                        writer.writerow(data)
            except:
                text = "CSV file could not be written"
                print(text)
                self.info_label.setStyleSheet("color : red")
                self.info_label.setText(text)
                error = True

        if error is not True:
            print("export done")
            self.info_label.setStyleSheet("color : green")
            self.info_label.setText("Data exported successfully")
        


    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)



    def create_action(self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            #self.connect(action, SIGNAL(signal), slot)
            #QObject.connect(action, SIGNAL(signal), slot)
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action


    # click on the canvas
    def press(self, event):
        #print(event, event.inaxes)
        print("press x:", event.xdata, "press y:", event.ydata) #, event)
        
        self.dragging_id = self.people.get_person_index(event.xdata, event.ydata, self.spinbox.value())

        # If we click on a current circle
        if self.dragging_id is not None:

            if event.button == 3:    # right button mouse
                print("DELETING")
                self.dragging_id = None
                self.people.removePerson(event.xdata, event.ydata, self.spinbox.value())

            elif event.button == 2:  # middle button mouse (change the type of the circle)
                print("CHANGING TYPE")
                self.people.change_type(self.spinbox.value(), self.dragging_id)
                self.dragging = None
            else:
                print("MOVING")
                index = self.spinbox.value()
                self.people.drag(index, event.xdata, event.ydata)

        else: # Create new region!
            print("CREATING NEW CIRCLE")
            self.people.newPerson(event.xdata, event.ydata, self.spinbox.value())

        self.on_draw()


    # Scrolling on a circle
    def scroll(self, event):
        
        if self.data is not None:
            delta = 0.1/self.data.range_max
            if event.button == "down":
                delta = -delta  # invert
            print(delta)
            index = self.spinbox.value()
            target = self.people.get_person_index(event.xdata, event.ydata, index)
            if target is not None:
                self.people.resize(index, event.xdata, event.ydata, delta)

            self.on_draw()


    # Drag a circle
    def motion(self, event):
        if self.dragging_id is not None:
            index = self.spinbox.value()
            self.people.drag(index, event.xdata, event.ydata)
            self.on_draw()


    # Release a circle
    def release(self, event):
        #print("release", event.xdata, event.ydata)
        self.dragging_id = None
    


    def setChildrenFocusPolicy (self, policy):
        def recursiveSetChildFocusPolicy (parentQWidget):
            for childQWidget in parentQWidget.findChildren(QWidget):
                childQWidget.setFocusPolicy(policy)
                recursiveSetChildFocusPolicy(childQWidget)
        recursiveSetChildFocusPolicy(self)


    # Redraw the changes
    def on_draw(self):
        """ Redraws the figure
        """
        if self.data is None:  # Don't get ahead of ourselves
            return

        index = self.spinbox.value()
        #self.circles.set_index(index)
        # Filter out max range points of "no return"
        #data_filtered = [r if r<self.data.range_max else None for r in self.data.data[index]]
        colors = self.people.get_colors(self.data, index)
        idx = np.array(self.data.data[index]) < self.data.range_max
        #self.lines.set_data(self.data.theta, data_filtered)
        if self.points is not None:
            self.points.remove()
        self.points = self.ax_p.scatter(self.data.theta[idx], self.data.data[index][idx], 3, colors[idx], zorder=5)
        
        #self.circles.render(self.ax_c)
        # noe
        self.people.render(self.ax_c, index)
        
        self.canvas.draw()

        # if we have images to show...
        if(self.show_images == True):

            min_i = 0
            if self.data is not None:
                #look for the closest image to the scan based on the timestamps
                scan_t = self.data.timeStamps[index]
                min_t = 999999
                for i in range(-10, 11): #range of 10 timesteps if we pass scans 10 by 10 (shift+arrow)
                    if (self.img_index+i) >= 0 and (self.img_index+i)<len(self.img_data.timeStamps):
                        img_t = self.img_data.timeStamps[self.img_index+i]
                        t = abs((scan_t - img_t).to_sec())
                        if(t<min_t):
                            min_t = t
                            min_i = copy.deepcopy(i)

            self.img_index += min_i


            image = QImage(self.img_data.images[self.img_index].data, self.img_data.images[self.img_index].shape[1], 
                self.img_data.images[self.img_index].shape[0], QImage.Format_Grayscale8)  #, QImage.Format_RGB888) #.rgbSwapped()

            w = 640 #self.img_data.images[self.img_index].shape[1]
            h = 480 #self.img_data.images[self.img_index].shape[0]
            pixmap = QPixmap(image)
            self.pic.resize(QSize(w,h))
            # self.pic.resize(self.img_data.images[self.img_index].shape[1], self.img_data.images[self.img_index].shape[0])
            # w = self.pic.width() # min(pixmap.width(),  self.pic.maximumWidth()) 
            # h = self.pic.height() #min(pixmap.height(), self.pic.maximumHeight())
            pixmap = pixmap.scaled(QSize(w, h), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.pic.setScaledContents(True)
            
            self.pic.setPixmap(pixmap)
            
            self.pic.show() 
            self.img_topic_label.setVisible(True)
            self.imgTopicsCB.setVisible(True)
            self.pic.setVisible(True)


    # move backwards
    def prev(self, event):
        mods = QApplication.keyboardModifiers()
        if bool(mods & Qt.ShiftModifier):
            self.spinbox.setValue(self.spinbox.value()-10)
        else:
            self.spinbox.setValue(self.spinbox.value()-1)
        msg = ""
        self.info_label.setText(msg)


    # move forward (or pause) in steps of 'scan_step_ms'
    def play(self, event):
        msg = ''
        if self.play_timer is None:
            self.play_timer = QTimer()
            self.play_timer.timeout.connect(self.next)
            self.play_timer.start(self.scan_step_ms)
            msg = "Playing"
        else:
            if self.play_timer.isActive():
                self.play_timer.stop()
                msg = "Paused"
            else:
                self.play_timer.start(self.scan_step_ms)
        if self.play_timer.isActive():
            self.play_button.setText(u"⏸") #⏸ #u"II"
            #self.play_timer.start(self.scan_step_ms)
            msg = "Playing"
            
        else:
            self.play_button.setText("\u23EF") #\u23EF #u"▶"
            msg = "Paused"

        print("Play")
        self.info_label.setStyleSheet("color : blue")
        self.info_label.setText(msg)


    # keyboard arrows
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self.prev(event)
        elif event.key() == Qt.Key_Right:
            self.next(event)
        elif event.key() == Qt.Key_Space:
            self.play(event)


    # move forward
    def next(self, *args):
        mods = QApplication.keyboardModifiers()
        if bool(mods & Qt.ShiftModifier):
            self.spinbox.setValue(self.spinbox.value()+10)
        else:
            # noe
            #if(self.spinbox.value() + 1) < self.data.length:
            #    scanxy = self.data.get_cartesian(self.spinbox.value()+1)
            #    self.people.update(self.spinbox.value()+1, scanxy)
            self.spinbox.setValue(self.spinbox.value()+1)
        msg = ""
        self.info_label.setText(msg)


    # Activate the recording from a given frame. Initial frame otherwise
    def record(self):
        if self.recording == False:
            self.init_scan = self.spinbox.value()
            self.record_button.setStyleSheet("color : red")
            self.record_button.setText(("Recording from scan " + str(self.init_scan)))
            self.recording = True
            #self.record_button.setEnabled(False)
        else:
            self.record_button.setStyleSheet("color : black")
            self.record_button.setText("Start REC")
            self.init_scan = 0
            self.recording = False



    # when the value of the scan counter changes
    def valueChanged(self, value):
        # do the tracking (update) only if we move one step forward
        count = value - self.current_index
        if count == 1 and value < self.data.length:
            scanxy = self.data.get_cartesian(value)
            self.people.update(value, scanxy)

        elif count > 1 and value < self.data.length:
            for c in range(self.current_index+1, value+1):
                scanxy = self.data.get_cartesian(c)
                self.people.update(c, scanxy)

        self.current_index = value
        self.on_draw()
        #print("ValueChanged. Scan %i, people: %i" % (int(value), len(self.people.data[int(value)].people)))


    # Change the slider of the time step between scans
    def changeStepValue(self, value):
        self.scan_step_ms = value
        print(value)
        self.info_label.setStyleSheet("color : blue")
        msg = "Time step between scans updated to: " + str(value) + " ms"
        self.info_label.setText(msg)
        #self.play_timer.start(self.scan_step_ms)


    # If we change the scan topic
    def topicsComboChanged(self):
        top = self.topicsCB.currentText()
        self.data.loadData(top)
        self.info_label.setStyleSheet("color : green")
        msg = "LASER SCAN " + top + " LOADED SUCCESFULLY!!!"
        self.info_label.setText(msg)
        self.spinbox.setMaximum(self.data.length-1)
        total_frames = ' of ' + str(self.data.length) + ' scans'
        self.total_label.setText(total_frames)
        self.play_button.setEnabled(True)
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        # Re-render everything
        self.on_draw()

    
    # if we change the image topic
    def imgTopicsComboChanged(self):
        imgt = self.imgTopicsCB.currentText()
        self.img_data.loadData(imgt)
        self.info_label.setStyleSheet("color : green")
        msg = "IMAGES FROM TOPIC '" + imgt + "' LOADED SUCCESFULLY!!!"
        self.info_label.setText(msg)
        # Re-render everything
        self.on_draw()
    


    def create_main_frame(self):
        self.main_frame = QWidget()

        #self.info_label = QLabel("")
        
        # Create the figure
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Create axes

        # the polar axis:
        # rect : This parameter is the dimensions [left, bottom, width, height] of the new axes.
        rect = [0,0,1,1] 
        self.ax_p = self.fig.add_axes(rect, polar=True, frameon=False, aspect='equal')  
        self.ax_c = self.fig.add_axes(rect, aspect='equal', frameon=False) 

        # Set up the cartesian plot
        self.ax_c.get_xaxis().set_visible(False)
        self.ax_c.get_yaxis().set_visible(False)

        # Set up the polar plot
        self.ax_p.set_rlabel_position(0)  # get radial labels away from plotted line
        self.ax_p.grid(True)
        self.ax_p.autoscale(False) #False
        self.ax_p.set_xticklabels(['0\u00B0', '45\u00B0', '90\u00B0', '135\u00B0', '\u00B1180\u00B0', '-135\u00B0', '-90\u00B0', '-45\u00B0'])

        # Patch
        self.people.render(self.ax_c, 0)

        # Render initial values
        #self.lines, = self.ax_p.plot([0], [0],'r-')
        
        # Bind the 'pick' event for clicking on one of the bars
        #
        # self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('motion_notify_event', self.motion)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.canvas.mpl_connect('scroll_event', self.scroll)
        
        # GUI controls
        # 
        self.prev_button = QPushButton('\u23EA') #u'23EA'
        self.prev_button.clicked.connect(self.prev) 
        self.prev_button.setEnabled(False)

        self.play_button = QPushButton('\u23EF') #\u23EF #u"▶"
        self.play_button.clicked.connect(self.play)
        self.play_button.setEnabled(False)

        self.next_button = QPushButton('\u23E9') #u'23E9'  u'23F5'
        self.next_button.clicked.connect(self.next)
        self.next_button.setEnabled(False)
        
        topic_label = QLabel("Topic:")
        #self.topic_edit = QLineEdit(self.topic)
        #self.topic_edit.setEnabled(True)
        self.topicsCB = QComboBox()
        #self.topic_edit.setMaxLength(10)
        self.topicsCB.currentIndexChanged.connect(self.topicsComboChanged)

        spinbox_label = QLabel('Scan #')
        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, 0)
        self.spinbox.setValue(0)
        self.spinbox.valueChanged.connect(self.valueChanged)
        self.spinbox.setFocusPolicy(Qt.NoFocus)

        total_frames = " of # scans"
        self.total_label = QLabel(total_frames)
        
        #
        # Button layout
        # 
        hbox = QHBoxLayout()
        
        for w in [  self.prev_button, self.play_button, self.next_button, topic_label, 
                    self.topicsCB, spinbox_label, self.spinbox, self.total_label]:
            hbox.addWidget(w)
            hbox.setAlignment(w, Qt.AlignVCenter)


        hbox2 = QHBoxLayout()
        self.record_button = QPushButton('Start REC') #u'23EA'
        self.record_button.clicked.connect(self.record)

        step_label = QLabel('scan step [ms]: 0')
        slider = QSlider(Qt.Horizontal, self)
        #slider.setFocusPolicy(Qt.StrongFocus)
        tickIntervals = 8
        maxValue = 300
        slider.setGeometry(30, 40, 200, 30)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(tickIntervals)
        slider.setRange(0, maxValue)
        slider.setValue(100)
        slider.valueChanged[int].connect(self.changeStepValue)
        end_step_label = QLabel(str(maxValue))
        hbox2.addWidget(self.record_button)
        hbox2.addWidget(step_label)
        hbox2.addWidget(slider)
        hbox2.addWidget(end_step_label)
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.info_label)
        vbox.addWidget(self.canvas)
        # vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        vbox.addLayout(hbox2)


        # right side to show video if we have it
        vbox2 = QVBoxLayout()
        self.pic = QLabel()
        #self.pic.setPixmap(QPixmap("Q107.png"))
        #self.pic.show() 
        self.img_topic_label = QLabel("Image Topic:")
        self.imgTopicsCB = QComboBox()
        self.imgTopicsCB.currentIndexChanged.connect(self.imgTopicsComboChanged)
        #self.img_topic_label.setVisible(False)
        #self.imgTopicsCB.setVisible(False)
        #self.pic.setVisible(False)
        vbox2.addWidget(self.img_topic_label)
        vbox2.addWidget(self.imgTopicsCB)
        vbox2.addWidget(self.pic)


        hbox2 = QHBoxLayout()
        hbox2.addLayout(vbox)
        hbox2.addLayout(vbox2)
        #hbox2.setEnabled(False)
        #hbox2.setAlignment(w, Qt.AlignVCenter)

        
        self.main_frame.setLayout(hbox2)
        self.setCentralWidget(self.main_frame)


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()

if __name__ == "__main__":
    main()
