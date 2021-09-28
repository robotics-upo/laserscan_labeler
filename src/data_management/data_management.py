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
    print('\t6) Concatenate bag files (not implemented yet).')
    print('')
    print('\t7) Exit')
    print()
    return input("Chose an option: ")




def save_data_menu(dataloader, x_data, y_data, path):
    save = input('Do you want to save the data? (type "y" or "n"): ')
    if(save == 'y'):
        print('The data will be saved in', path, 'with names "[prefix]_x_data.npy" and "[prefix]_y_data.npy"')
        prefix = input('Type a prefix for the new files: ')
        dataloader.save_data(x_data, y_data, path, prefix)
    



def merge_files(dataloader):
    x_data = []
    y_data = []
    print('--- Merging session files (output files from the labeler tool [csv]) ---')
    print('The program will merge the files of different sessions located in the directories "scans" (to build x_data) and "labels" (to build y_data).')
    print('Range data will be also normalized.')
    path = input('So please, type the base directory containing these directories: ')
    x_data_path = os.path.join(path, 'scans')
    y_data_path = os.path.join(path, 'labels')
    nr = int(input("please type the number of ranges of the laser data (720 in Frog, 450 in Drow): "))
    angle_inc_degrees = float(input("please introduce the angle increment in degrees of the laser data (0.25 in Frog, 0.5 in Drow): "))
    try:
        x_data, y_data = dataloader.join_data(x_data_path, y_data_path, nr)
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
        x_data, y_data = dataloader.format_data_for_learning(x_data, y_data, nr, angle_inc_degrees)
        print()
        print("New data shape:")
        print('x_data shape:', x_data.shape, 'type:', x_data.dtype)
        print('y_data shape:', y_data.shape, 'type:', y_data.dtype)
        # for i, d in enumerate(x_data):
        #     if i>0:
        #         comparison = x_data[i] == x_data[i-1]
        #         if comparison.all():
        #             print ("scan", i, "equals to scan", i-1)
    print()
    print('The data will be saved in', path, 'with names "[prefix]_x_data.npy" and "[prefix]_y_data.npy"')
    prefix = input('Type a prefix for the new files: ')
    dataloader.save_data(x_data, y_data, path, prefix)
    
    #save_data_menu(dataloader, x_data, y_data, path)
    return True #, x_data, y_data
    


def merge_bags(dataloader):
    x_data = []
    y_data = []
    print('--- Merging pre-joined session files (formatted npy files) ---')
    print('The program will load and merge the numpy files of complete bags located in the directories "x_data" and "y_data".')
    path = input('So please, type the base directory containing these directories: ')
    x_data_path = os.path.join(path, 'x_data')
    y_data_path = os.path.join(path, 'y_data')
    #try:
    ok, x_data, y_data = dataloader.join_formatted_data(x_data_path, y_data_path)
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
    dataloader.drow_to_frog(path, 'frog_data')


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






if __name__ == '__main__':

    # data management
    dataLoader = LoadData()


    options = {'MERGECSV': 1, 'MERGEBAGS': 2, 'TODROW': 3, 'TOFROG': 4, 'GENBAG': 5,
     'CONBAG': 6, 'EXIT': 7}

    var = 0
    ok = False

    while int(var) != options['EXIT']:
        var = print_main_menu()
        if int(var) == options['MERGECSV']:
            ok = merge_files(dataLoader)

        elif int(var) == options['MERGEBAGS']:
            ok = merge_bags(dataLoader)

        elif int(var) == options['TODROW']:
            frog_to_drow(dataLoader)

        elif int(var) == options['TOFROG']:
            drow_to_frog(dataLoader)

        #elif int(var) == options['REFORMAT']:
        #    to_new_format(dataLoader)

        elif int(var) == options['GENBAG']:
            csv_to_bag(dataLoader)

            

