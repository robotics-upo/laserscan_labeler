#!/usr/bin/python3
# This Python file uses the following encoding: utf-8

import sys
import os
from load_frog_data import LoadData



def print_main_menu():
    print('')
    print('----------------------------------------------------------')
    print('MENU:')
    print('----------------------------------------------------------')
    print('DATA PREPROCESSING:')
    print('\t1) Merge the data sessions (csv files) into formatted numpy files.')
    print('\t2) Merge the data of multiple numpy files (output numpy files of option 1).')
    print('\t3) Transform labeler output files (csv) to DROW dataset format.')
    print('\t4) Transform DROW dataset files to scanlabeler format.')
    print('\t5) Generate a bag file from a set of csv session files.')
    print('\t6) Generate classification and localization labels from circles csv files.')
    print('\t7) Generate Gaussian labels from binary scanlabeler files (csv).')
    print('\t8) Generate binary labels covering full people from binary scanlabeler files (csv).')
    print('')
    print('\t9) Exit')
    print()
    return input("Chose an option: ")




def save_data_menu(dataloader, x_data, y_data, path):
    save = input('Do you want to save the data in npy files? (type "y" or "n"): ')
    if(save == 'y'):
        print('The data will be saved in', path, 'with names "[prefix]_x_data.npy" and "[prefix]_y_data.npy"')
        prefix = input('Type a prefix for the new files: ')
        dataloader.save_data(x_data, y_data, path, prefix)
    



def merge_files(dataloader):
    x_data = []
    y_data = []
    print('--- Merging session files (output files from the labeler tool [csv]) ---')
    bin = int(input('Are the labels binary? (type "1" for yes, or "0" for no, or "2" if classification and regression labels are used): '))
    if(bin==1):
        print('The program will merge the files of different sessions located in the directories "scans" (to build x_data) and "labels" (to build y_data).')
    elif(bin==0):
        print('The program will merge the files of different sessions located in the directories "scans" (to build x_data) and "regression_labels" (to build y_data).')
    else:
        print('The program will merge the files of different sessions located in the directories "scans" (to build x_data) and "class_and_loc_labels" (to build y_data).')
    path = input('So please, type the base directory containing these directories: ')
    x_data_path = os.path.join(path, 'scans')
    if(bin==1):
        y_data_path = os.path.join(path, 'labels')
    elif(bin==0):
        y_data_path = os.path.join(path, 'regression_labels')
    else:
        y_data_path = os.path.join(path, 'class_and_loc_labels')

    normalize = input('Do you want to normalize the data? Type "y" or "n": ')
    if(normalize == 'y' or normalize == "yes"):
        normalize = True
    else:
        normalize = False

    nr = int(input("please type the number of ranges of the laser data (720 in Frog, 450 in Drow): "))
    angle_inc_degrees = float(input("please introduce the angle increment in degrees of the laser data (0.25 in Frog, 0.5 in Drow): "))
    
    if(bin<2):
        try:
            x_data, y_data = dataloader.join_data(x_data_path, y_data_path, nr, binary_label=bin, norm=normalize)
        except:
            print("ERROR: file data could NOT be loaded from: %s" % path)
            return False  #, x_data, y_data

        print("Data loaded:")
        print('x_data shape:', x_data.shape, 'type:', x_data.dtype)
        print('y_data shape:', y_data.shape, 'type:', y_data.dtype)
        print()
        format = input('Do you want to save the data in the format of the learning network [1440 ranges, 0.25 degrees inc]? (type "y" or "n"): ')
        if(format == 'y'):
            print('Now the data will be transformed to the format of the learning network...')
            x_data, y_data = dataloader.format_data_for_learning(x_data, y_data, nr, angle_inc_degrees, data_normalized=normalize)
            print()
            print("New data shape:")
            print('x_data shape:', x_data.shape, 'type:', x_data.dtype)
            print('y_data shape:', y_data.shape, 'type:', y_data.dtype)

        print()
        print('The data will be saved in', path, 'with names "[prefix]_x_data.npy" and "[prefix]_y_data.npy"')
        prefix = input('Type a prefix for the new files: ')
        dataloader.save_data(x_data, y_data, path, prefix)


    else:
        if normalize == True:
            norm_type = int(input('Localization data will be normalized. Which range do you want to use? Type 1 for [0,1], or 2 for [-1,1]: '))
            is_polar = int(input("Is the localization data in polar coordinates or cartesian? Type '0' for cartesian or '1' for polar: "))
        else:
            norm_type = 0
            is_polar = 0
        try:
            x_data, yc_data, yl_data = dataloader.join_class_and_loc_data(x_data_path, y_data_path, nr, norm_type=norm_type, polar=is_polar)
            print("Data loaded:")
            print('x_data shape:', x_data.shape, 'type:', x_data.dtype)
            print('y_class_data shape:', yc_data.shape, 'type:', yc_data.dtype)
            print('y_loc_data shape:', yl_data.shape, 'type:', yl_data.dtype)
            print()
            print('The data will be saved in', path, 'with names "[prefix]_x_data.npy", "[prefix]_y_class.npy" and "[prefix]_y_loc.npy"')
            prefix = input('Type a prefix for the new files: ')
            x_name = prefix + '_x_data.npy'
            x_data_dir = os.path.join(path, x_name)
            dataloader.save_npy(x_data, x_data_dir)
            yc_name = prefix + '_y_class.npy'
            yc_data_dir = os.path.join(path, yc_name)
            dataloader.save_npy(yc_data, yc_data_dir)
            yl_name = prefix + '_y_loc.npy'
            yl_data_dir = os.path.join(path, yl_name)
            dataloader.save_npy(yl_data, yl_data_dir)
            print('Data saved in:')
            print(x_data_dir)
            print(yc_data_dir)
            print(yl_data_dir)
        except:
            print("ERROR: file data could NOT be loaded from: %s" % path)
            return False  #, x_data, y_data
    
    #save_data_menu(dataloader, x_data, y_data, path)
    return True #, x_data, y_data
    


def merge_npy(dataloader):
    x_data = []
    y_data = []
    print('--- Merging pre-joined session files (formatted npy files) ---')
    bin = bool(input('Are the labels to be merged binary? (type "1" for yes, or "0" for no): '))
    print('The program will load and merge the numpy files of complete bags located in the directories "x_data" and "y_data".')
    path = input('So please, type the base directory containing these directories: ')
    x_data_path = os.path.join(path, 'x_data')
    y_data_path = os.path.join(path, 'y_data')
    #try:
    ok, x_data, y_data = dataloader.join_formatted_data(x_data_path, y_data_path, binary_label=bin)
    if ok == False:
        return False #, x_data, y_data
    #except:
    #print("ERROR: file data could NOT be loaded from: %s" % path)
    #return False, x_data, y_data
    print("Data loaded:")
    print('x_data shape:', x_data.shape, 'type:', x_data.dtype)
    print('y_data shape:', y_data.shape, 'type:', y_data.dtype)
    save_data_menu(dataloader, x_data, y_data, path)
    return True #, x_data, y_data



def frog_to_drow(dataloader):
    print('')
    print('First, the input scan files will be transformed to drow format.')
    spath = input('Introduce the base directory in which your scans ([whatever]_scans.csv files) are: ')
    dataloader.scan_to_drow(spath)
    print('Secondly, the people detections will be transformed to drow format.')
    ppath = input('Introduce the base directory in which your detections ([whatever]_circles.csv files) are: ')
    dataloader.circles_to_drow(ppath)



def drow_to_frog(dataloader):
    print('')
    path = input('Introduce the base directory in which your drow sequence files are (*.csv, *.wc, *.wa, *.wp): ')
    print('The new files will be created in the folder: ', '/frog_data')
    print('NOTE: only the people(*.wp) will be used for the moment.')
    type = input("Do you want to upsample the scan data? (type 'y' or 'n'):")
    upsampling = False
    if type == 'y':
        upsampling = True
    dataloader.drow_to_frog(path, 'frog_data', upsampling)


def to_new_format(dataloader):
    print('')
    print('First, the raw scan files will be transformed to the new format.')
    path = input('Introduce the base directory in which your raw scans ([whatever]_scans.csv files) are: ')
    dataloader.scan_to_new(path, label=0)
    path = input('Now, introduce the base directory in which your labels ([whatever]_labels.csv files) are: ')
    dataloader.scan_to_new(path, label=1)




def csv_to_bag(dataloader):
    print('')
    print('To generate the bag files (like those generated by the laserscan_labeler), the [X]_scans.csv and [X]_circles.csv files are necessary.')
    print('The program will look for these files in the folders "scans" and "circles".')
    print('IMPORTANT: this is only valid for frog laserscans, so as to use other data, please use the laserscan_labeler tool.')
    path = input('So please, introduce the base directory containing these folders: ')
    scans_path = os.path.join(path, 'scans')
    circles_path = os.path.join(path, 'circles')
    #try:
    dataloader.csv_to_bag(scans_path, circles_path, path)
    #except:
    #    print("ERROR: scans and circles files could not be loaded from: %s" % path)


def circles_to_class_and_loc_labels(dataloader):
    print('')
    print('To generate the new label files, the circle files ([X]_circles.csv) are necessary.')
    print('The program will look for these files in the folder "circles", ')
    print('and will write the new label files in the new folder "class_and_loc_labels".')
    path = input('So, introduce the base directory containing the circles folder: ')
    circles_path = os.path.join(path, 'circles')
    labels_path = os.path.join(path, 'class_and_loc_labels')
    max_people = int(input('How many people could be detected at the same time (the maximum): '))
    print("Which order do you want to use?")
    ordertype = int(input("Type '1' for ordering by the closest people; or type '2' for ordering according to the scan order: "))
    nranges = int(input('How many range values do we have? [720 for frog scan]: '))
    angle_res = float(input('What is the angle resolution of the scan? [0.25 degrees for frog scan]: '))
    print('Do you want to store the localization data in polar coords or cartesian coords?')
    coord_type = int(input("Type '1' for polar or '0' for cartesian: "))
    dataloader.circles_to_class_and_loc_labels(circles_path, labels_path, max_people, order_type=ordertype, nranges=nranges, ares=angle_res, polar=coord_type)



def binary_to_gaussian_labels(dataloader):
    print('')
    print('To generate the Gaussian labels files, the [X]_scans.csv and [X]_circles.csv files are necessary.')
    print('The program will look for these files in the folders "scans" and "circles", ')
    print('and will write the new label files in the folder "labels".')
    path = input('So please, introduce the base directory containing these folders: ')
    scans_path = os.path.join(path, 'scans')
    circles_path = os.path.join(path, 'circles')
    labels_path = os.path.join(path, 'labels')
    dataloader.binary_to_gaussian_label(scans_path, circles_path, labels_path)


def binary_to_people_labels(dataloader, g=0):
    print('')
    print('To generate the people labels files, the [X]_scans.csv and [X]_circles.csv files are necessary.')
    print('The program will look for these files in the folders "scans" and "circles", ')
    print('and will write the new label files in the folder "labels".')
    path = input('So, introduce the base directory containing these folders: ')
    scans_path = os.path.join(path, 'scans')
    circles_path = os.path.join(path, 'circles')
    labels_path = os.path.join(path, 'labels')
    dataloader.binary_to_peoplebin_label(scans_path, circles_path, labels_path, gaussianlabel=g)



if __name__ == '__main__':

    # data management
    dataLoader = LoadData()


    options = {'MERGECSV': 1, 'MERGENPY': 2, 'TODROW': 3, 'TOFROG': 4, 'GENBAG': 5,
     'LOC': 6, 'GAUSS': 7, 'NEWBIN': 8, 'EXIT': 9}

    var = 0
    ok = False

    while int(var) != options['EXIT']:
        var = print_main_menu()
        if int(var) == options['MERGECSV']:
            ok = merge_files(dataLoader)

        elif int(var) == options['MERGENPY']:
            ok = merge_npy(dataLoader)

        elif int(var) == options['TODROW']:
            frog_to_drow(dataLoader)

        elif int(var) == options['TOFROG']:
            drow_to_frog(dataLoader)

        elif int(var) == options['GENBAG']:
            csv_to_bag(dataLoader)

        elif int(var) == options['LOC']:
            circles_to_class_and_loc_labels(dataLoader)

        elif int(var) == options['GAUSS']:
            binary_to_people_labels(dataLoader, g=1)

        elif int(var) == options['NEWBIN']:
            binary_to_people_labels(dataLoader, g=0)


            

