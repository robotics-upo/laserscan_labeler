# laserscan_labeler

Tool for labeling laser scan data from ROS bags.

Based on the scan labeler of Joseph Duchesne (https://github.com/josephduchesne/laserscan_labeler)

![](https://github.com/robotics-upo/laserscan_labeler/blob/master/images/app.jpeg)

## Prereqs

The tool is implemented using Python 3, PyQt5 and ROS Noetic.
For previous ROS distros not using Python 3, a virtual environment may be employed.
E.g.:

```python
python3 -m venv ~/VirtualEnvs/laserscan_labeler
. ~/VirtualEnvs/laserscan_labeler/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install rospkg numpy matplotlib PyQt5 pyyaml scipy opencv-python-headless pycryptodomex gnupg lz4
```

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
    
      > *[id, timestamp, range_array(r1, r2, ..., rX)] in which r0, r1, ..., rX belongs to [0, inf] meters*
    
    - *[given_name]_labels*. It will contain a list of all the exported labeled scans. Each label array is accompanied with an id and a timestamp that can be used to match the data with the other files. A label consists on an array, of the same length of the range array, in which each range of the scan is classified in 4 classes: 0-> no person, 1-> person, and classes 2 and 3 are left for classifying other "things" (e.g. pushchairs) that the user would like to.
    
      > *[id, timestamp, label_array(c0, c1, ..., cX)] in which c0, c1, ..., cX belongs to [0,1,2,3]* 
    
    - *[given_name]_circles*. It contains all the circles that represent the people labeled for each scan. Again and id and a timestamp for each scan is included along with a list of circles. For each circle we provide:
      - A person id that is mantained along the scans. 
      - The x,y position of the circle center (with origin in the scan frame).
      - The circle radius.
      - The type of the label. Besides people (class 1), other "things" can be labeled. Two more classes are available (classes 2 and 3).
      
      > *[id, timestamp, circles_array(cir1, ..., cirX)] in which each circle contains [person_id, x, y, radius, type]*
  
- Finally, we have the "Quit" option to exit the application.


### Scan area panel

On this panel, the points of the laser scan (in red color) are represented in a grid. To label the scan point we can perform the following actions:

- Left clicking on a region to create a new labeled scan group based on the scan points contained in the new circle (a green circle will appear).
- Click and drag a circle to move it over the scan area.
- Right clicking a circle deletes it (from that scan on).
- Using the scroll wheel over a circle enlarges or shrinks its radius.
- Optionally, we can change the class of the label by clicking in the circle with the middle button of the mouse. We can change between the different classes available: green color for people - class 1 by default, blue color - class 2, yellow color - class 3. It is recommended to use class 1 for people, and the other two classes for other stuff that the user would like to label. 


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

Each person circle (or other class) will try to track the movement of the person between consecutive scans by computing the average point of the points located inside the circle.
At any moment, the reproduction can be paused and moved backwards and the circles can be modified, deleted or created. 

