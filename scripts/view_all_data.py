#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:21:00 2019

@author: mitchell
"""

from amp3GImageProc import AMP3GImageProc
import logging
import time
import glob
import datetime
import cv2

###################### CHANGE THESE FOLDER LOCATIONS ##########################
base_folder = "/media/3GAMP/" #Make sure the last backslash is there!
start_date = ' '#Set null string (' ') to search over full space

def show_images(images):
    for image in images:
        img = cv2.imread(image)
        cv2.imshow('img', img)
        cv2.waitKey(1)

def beyond_date(date, start_date):
    if start_date == ' ':
        return True
    else:
        year = int(date.split('_')[0])
        month = int(date.split('_')[1])
        day = int(date.split('_')[2])
        date1 = datetime.date(year=year, month=month, day=day)
        start_date = start_date.split("/")[3]
        start_year = int(start_date.split('_')[0])
        start_month = int(start_date.split('_')[1])
        start_day = int(start_date.split('_')[2])
        date2 = datetime.date(year=start_year, month=start_month, day=start_day)
        if time.mktime(date1.timetuple()) > time.mktime(date2.timetuple()):
            return True
        return False
        

def main(base_folder, start_date):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    sub_folders = sorted(glob.glob(base_folder))
    #Something is fishy with these files...ignore them
    ignore_dates = []
    #Loop through all folders under the root directory
    for folder in sub_folders:
        #print(folder)
        current_folder = 'Current folder: %s' % (str(folder))
        logger.info(current_folder)
        
        date = folder.split("/")[3]        
        #Ignore folders that aren't of specific dates
        if date[0:2] == '20' and folder not in ignore_dates:
            beyond = beyond_date(date, start_date)
            if beyond:
                #Only search over daylight hours
                WIP = AMP3GImageProc(root_dir = folder, hour_min = 8, hour_max = 18)
                WIP.display_images()
                '''
                subdirs = WIP.sub_directories
                for subdir in subdirs:
                    #print(subdir)
                    images = glob.glob(subdir +'/Manta 1/*.jpg')
                    show_images(images)
                '''
        
            
if __name__ == '__main__':
    base_folder = base_folder + '*'
    main(base_folder, start_date)
        
    
        