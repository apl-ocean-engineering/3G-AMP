#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Mon March 11 15:06:00 2019

@author: mitchell
"""

from amp3GImageProc import AMP3GImageProc
import datetime
import time
import os
import glob 

#########USER SET PARAMATERS##############
main_directory = "/Users/AMP/Desktop/Data/"
base_folder_directory = os.path.abspath('..')
data_directory = base_folder_directory + '\data'
sleep_time = 10

##String manipuation for windows...
homography_transform = data_directory + '\homography_transform.txt'
homography_transform.replace('\\', '/')[2:]

def runamp3G(base_path):
    #Create amp3G object and run
    amp3G = AMP3GImageProc(save_directory = data_directory, 
                homography_transform = homography_transform)
    amp3G.image_overlap(base_path, display_images = False, save = True)
               

def main(sleep):
    now = datetime.datetime.now()
    current_date = now.strftime("%Y_%m_%d")
    date_directory = main_directory + current_date + "/*"
    sub_folders = sorted(glob.glob(date_directory), reverse = True)
    while(1):
        if (current_date != now.strftime("%Y_%m_%d")):
            now = datetime.datetime.now()
            current_date = now.strftime("%Y_%m_%d")
            date_directory = main_directory + current_date + "/*"
        new_directory = sorted(glob.glob(date_directory), reverse = True)
        for most_recent in new_directory:
           if most_recent not in sub_folders and most_recent[-1].isdigit():
               folder_name_date = most_recent[-19:] #always 9 digits long
               print("Checking folder: ", folder_name_date)
               #manipulate string...
               most_recent_folder = main_directory + current_date + "/" + folder_name_date + "/"
               runamp3G(most_recent_folder)
               
        sub_folders = new_directory        
        time.sleep(sleep)


if __name__ == '__main__':
    main(sleep_time)