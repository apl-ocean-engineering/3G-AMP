#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 12:21:00 2019

@author: mitchell
"""

from amp3GImageProc import AMP3GImageProc, BasePath
import logging
import time
import glob
import datetim

######################## CHANGE FOLDER VARIABLES ##############################
base_folder = "/media/3GAMP"
start_date = ' '#'/media/3GAMP/2019_03_03/2019_03_08'#Set null string (' ') to search over full space
time_delay = 1 #ms



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
    
def parse_false_positives(path):
    with open(path) as f:
        pass
    
def view_data_lst(path):
    pass

    
def view_data(path):
    amp3G = AMP3GImageProc()
    #amp3G.image_overlap(path, display_images = True, display_overlap = True, save = False)
        

def main(base_folder, start_date):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    sub_folders = sorted(glob.glob(base_folder + "/*"))
    
    for subdir in sub_folders:
        b = BasePath(root_dir = subdir)
        display_dates = b.sub_directories
        date = subdir.split("/")[3]   
        print(subdir)
        if date[0:2] == '20':
            beyond = beyond_date(date, start_date)
            if beyond:
                for date in display_dates:
                    amp3G = AMP3GImageProc(save_directory = data_directory, 
                                    homography_transform = homography_transform)
                    amp3G.image_overlap(date)

            
                
if __name__ == '__main__':
    base_folder = base_folder
    #main(base_folder, start_date)
    #view_data('/media/3GAMP/2019_03_04/2019_03_04 10_31_18/')
    #view_data('/media/3GAMP/2019_03_07/2019_03_07 12_52_42/')
    view_data('/media/3GAMP/2019_01_21/2019_01_21 01_10_55/')
    #view_data("/home/mitchell/3G-AMP-Calibration/2018_12_12 14_16_56/")
    #view_data_lst(data_directory + "/view_data.txt")
        