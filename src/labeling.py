#!/usr/bin/python3
# -*- coding: utf-8 -*- 

import numpy as np
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import copy


class PersonRegion():

    def __init__(self, x=0, y=0, r=0, person_id=0, scan_id=0, type=1):
        self.person_id = person_id
        self.scan_id = scan_id
        #self.manager = manager
        self.type = type
        self.x = x
        self.y = y
        self.r = r
        print("person circle created: ")
        print("scan_id:%i, person_id:%i, x:%.2f, y:%.2f, r:%.2f" % (self.scan_id, self.person_id, self.x, self.y, self.r))
        

    def get_data(self):
        return (self.x, self.y, self.r, self.person_id, self.scan_id, self.type)

    def set_data(self, data):
        self.x, self.y, self.r, self.person_id, self.scan_id = data

    def contains(self, px, py):
        return (px-self.x)**2 + (py-self.y)**2 <= self.r**2
    

    def render(self):
        c = None
        if self.type == 1: 
            c = mpatches.Circle((self.x, self.y), self.r, ec="none", color=(0,1,0))  # Green default region
        elif self.type == 2:
            c = mpatches.Circle((self.x, self.y), self.r, ec="none", color=(0,0,1))  # Blue for type 2
        elif self.type == 3:
            c = mpatches.Circle((self.x, self.y), self.r, ec="none", color=(1,1,0))  # Yellow for type 3
            
        return c


    def change_type(self, newt):
        self.type = newt


    def move(self, x, y, index):
        self.x = x
        self.y = y
        self.scan_id = index
        print("Updating person %i: x=%.3f, y:%.3f of scan %i" % (self.person_id, self.x, self.y, self.scan_id))

        

    def resize(self, delta):
        r = max(abs(delta), self.r + delta)
        self.r = r
        print("Resizing radius of person %i of scan %i" % (self.person_id, self.scan_id))

            
            
class PeopleScan():

    def __init__(self, scan_id=0):

        self.scan_id = scan_id
        self.people = []
        self.labeled_scan = [] 
        #self.scan = []
        self.people_id = 0


    def insert(self, person):
        if person.scan_id == self.scan_id:
            self.people.append(person)
        else:
            print('Scan id of the person (%i) does not match this scan_id (%i). Not inserting...' % (person.scan.id, self.scan_id))

    def insert(self, x, y, r, type=1):
        self.people_id += 1
        p = PersonRegion(x, y, r, self.people_id, self.scan_id, type)
        self.people.append(p)


    def change_type(self, person_id):
        idx = [i for i, p in enumerate(self.people) if p.person_id == person_id]
        if idx is not None:
            t = self.people[idx[0]].type
            t = (t+1)%4
            if t == 0:
                t = 1
            self.people[idx[0]].change_type(t)
            print("Changing type of id: %i of scan %i, to type: %i" % (self.people[idx[0]].person_id, self.scan_id, t))


    def remove(self, person_id):
        person = [p for p in self.people if p.person_id == person_id]
        if len(person) > 0:
            print("Removing person id: %i of scan %i" % (person[0].person_id, self.scan_id))
            self.people.remove(person[0])
            

    # def remove(self, x, y):
    #     for p in self.people:
    #         if(p.contains(x, y)):
    #             #print(self.people)
    #             print("Removing person id: %i of scan %i" % (p.person_id, self.scan_id))
    #             self.people.remove(p)
    #             #print(self.people)


    def resize(self, x, y, delta):
        for p in self.people:
            if(p.contains(x, y)):
                p.resize(delta)


    # def getLabeledScan(self, data, index):
    #     cartesian = data.get_cartesian(index)
    #     classes = []
    #     for i in range(len(cartesian)):
    #         if self.get_patch_index(*cartesian[i]) is None:
    #             classes.append(0)
    #         else:
    #             classes.append(1)
    #     return np.array(classes)

    def getLabeledScan(self, scan_xy):
        #if scan_id == self.scan_id:
        self.labeled_scan = [0] * len(scan_xy)
        for idx, s in enumerate(scan_xy):
            for p in self.people:
                if(p.contains(s[0], s[1]) == True):
                    self.labeled_scan[idx] = p.type
        return np.array(self.labeled_scan)


    def getPeople(self):
        return self.people




class peopleManager():

    def __init__(self):

        self.data = []  #list of PeopleScan
        self.data.append(PeopleScan())
        #self.current_index = 0
        self.patch = None


    def newPerson(self, x, y, scan_id):
        self.data[scan_id].insert(x, y, r=0.05)
    

    def removePerson(self, x, y, scan_id):
        print("removing person from", scan_id, "to", len(self.data))
        id = -1
        for p in self.data[scan_id].people:
            if p.contains(x, y):
                id = p.person_id
        if id != -1:
            for i in range(scan_id, len(self.data)):
                self.data[i].remove(id)


    def change_type(self, scan_id, person_id):
        self.data[scan_id].change_type(person_id)


    def update(self, index, scanxy):

        # in case we jump (lot of steps) to start for an advanced frame
        if index > len(self.data)+1:
            print("filling %i scans with empty people" % (index - len(self.data)))
            for r in range(len(self.data), index+1):
                self.data.append(PeopleScan())

        else:
            ps = copy.deepcopy(self.data[index-1])
            ps.scan_id = index
            #print("update. people in scan", index-1, ":", len(self.data[index-1].people))
            for c in ps.people:
                x = 0
                y = 0
                count = 0
                #print("prev position. x:", c.x, "y:", c.y)
                for p in scanxy:
                    if c.contains(p[0], p[1]):
                        x = x + p[0]
                        y = y + p[1]
                        count += 1
                #compute new center
                if(count > 0):
                    x = x/count
                    y = y/count
                    c.move(x, y, index)
                    #print("new position. x:", c.x, "y:", c.y)
            #print("update. people in scan", index, ":", len(ps.people))
            if index >= len(self.data):
                # add new PeopleScan
                self.data.append(ps)
            else:
                # udpate the PeopleScan if it was already played
                self.data[index] = ps


    def drag(self, scan_id, x, y):
        #print("Dragging to pose x: %.3f, y: %.3f in scan %i" % (x, y, scan_id))
        for p in self.data[scan_id].people:
            if p.contains(x, y):
                p.move(x, y, scan_id)


    def resize(self, scan_id, x, y, delta):
        self.data[scan_id].resize(x, y, delta)


    def cleanup(self):
        if self.patch is not None:
            self.patch.remove()
            self.patch = None


    def render(self, ax, scan_id):
        patches = []
        #for i in range(len(self.current)):
        #    patches.append(self.current[i].render(self.index))
        if len(self.data) > scan_id:
            for p in self.data[scan_id].people:
                patches.append(p.render())

        self.cleanup()
        if len(patches)>0:
            self.patch = PatchCollection(patches, alpha=0.4, match_original=True)
            ax.add_collection(self.patch)


    def get_person_index(self, x, y, scan_id):
        if len(self.data) > scan_id: 
            for p in self.data[scan_id].people:
                if p.contains(x, y):
                    return p.person_id
        return None


    def get_colors(self, data, scan_id):
        cartesian = data.get_cartesian(scan_id)
        colors = []
        for i in range(len(cartesian)):
            if self.get_person_index(cartesian[i][0], cartesian[i][1], scan_id) is None:
                colors.append('r')
            else:
                colors.append('b')
        return np.array(colors)