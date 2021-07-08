# laserscan_labeler

Tool for labeling laser scan data from ROS bags.

Based on the scan labeler of Joseph Duchesne (https://github.com/josephduchesne/laserscan_labeler)

![](https://github.com/robotics-upo/laserscan_labeler/blob/master/images/app.jpeg)

## Prereqs

It should work on any version of Ubuntu with ROS (from 'Kinetic' distro onwards) and python-qt5.
Other Python deps required:
* numpy
* matplotlib
* json
* csv

## Usage

To get started, run:

`python src/qt_labeler.py`

Then, open a bag file from the file menu. 


### File Menu

![](https://github.com/robotics-upo/laserscan_labeler/blob/master/images/menu.jpeg)

- First, you must open a bag file containing laser scans using the option "Open scan bag..." of the file menu.
In case that the bag contains also images, those will be shown in the right side of the scan panel.

- Optionally, you can open another bag file with images by the option "Open image bag...".

- Once we have finished the labeling, different options to save the data are presented:
  - To save a new bag file ("Save bag file..."). this bag will contain the laser scan of the primary bag, the TF topic in case it exists, and two new additional topics:
    - **/scan/people** [*people_msgs/People*]. Topic with the positions and ids of the people detected.
    - **/scan/people/markers** [*visualization_msgs/MarkerArray*]. For RViz visualization purposes.
    
  - To export the data in different formats ("Export..." option). The available formats are: CSV, JSON, NUMPY or MAT. In all cases, three files will be created:
  
    - *[given_name]_scans*. It will contain a list of all the exported scans. Each range array is accompanied with an id and a timestamp. 
    
    *[id, timestamp, range_array(r1, r2, ..., rX)]*
    
    - *[given_name]_labels*. It will contain a list of all the exported labeled scans. Each label array is accompanied with an id and a timestamp that can be used to match the data with the other files. A label consists on an array, of the same length of the range array, in which each range of the scan is classified in two classes: 0-> no person, 1-> person.
    
    *[id, timestamp, label_array(c0, c1, ..., cX)]* 
    
    - *[given_name]_circles*. It contains all the circles that represent the people labeled for each scan. Again and id and a timestamp for each scan is included along with a list of circles. For each circle we provide:
      - A person id that is mantained along the scans. 
      - The x,y position of the circle center (with origin in the scan frame).
      - And the circle radius.
      
      *[id, timestamp, circles_array()]*
  
- Finally, we have the "Quit" option to exit the application.


### Scan area panel

On this panel, the points of the laser scan (in red color) are represented in a grid. To label the scan point we can perform the following actions:

- Left clicking on a region to create a new labeled scan group based on the scan points contained in the new circle (a green circle will appear).
- Click and drag a circle to move it over the scan area.
- Right clicking a circle deletes it (from that scan on).
- Using the scroll wheel over a circle enlarges or shrinks its radius.


### Bottom set of controls

![](https://github.com/robotics-upo/laserscan_labeler/blob/master/images/botton_buttons2.jpg)

- Reproduction menu. It has the classical buttons to play/pause the reproduction and to move forward and backward in steps of one scan.
- The app automatically loads the first laser scan topic it finds. To run it on other topics (if they exist in the bag), use the scan topic selector.
- Scan indicator. This indicador shows the current index of the scan according to the total number of scans found in the indicated topic in the bag file. After the labeling process, the current scan number will be used as the final scan to be saved/exported. 
- "Star REC" button. This button can be employed if we want to start the recording (to be saved later) from a particular number of scan and to discard all the previous scans.
To use it, move the reproduction to the desired scan and press the "Start REC" button. The button will take the current scan number indicated in the scan indicator as the initial scan to be stored later. If the button is not used, the zero scan will be used as the initial by default.
- Player velocity slider. It allows to change the velocity of reproduction of the sucesive scans when we are playing the bag. 


### Time travel options

Besides the reproduction buttons to play/pause and move step forward and backward, we have other options:
- Press space to play/pause the reproduction.
- Use the arrow keys of the keyboard (left/right) to move forward/backward in time.
- Besides, if the shift key is pressed along with the arrow keys, the scan will travel in time in steps of 10 scans. 


### Person Tracking

Each person circle will try to track the movement of the person between consecutive scans by computing the average point of the points located inside the circle.
At any moment, the reproduction can be paused and moved backwards and the circles can be modified, deleted or created. 


## TODO

* The app currently only offers binary labeling (0=default, 1=in a marked region). This could be extended to label more classes. 
