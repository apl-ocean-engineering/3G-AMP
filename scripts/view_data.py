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
import datetime
import os

######################## CHANGE FOLDER VARIABLES ##############################
base_folder = "/media/3GAMP"
start_date = ' ' #'/media/3GAMP/2019_03_01/2019_03_01'#Set null string (' ') to search over full space
time_delay = 1 #ms

main_directory = '/'.join(os.path.dirname(os.path.realpath(__file__)).split('/')[:-1])
data_directory = main_directory + '/data'
homography_transform = data_directory + '/3Ghomography_transform.txt'


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
    
def view_data_lst(path):
    f = open(path)
    lines = f.readlines()
    f.close()

    for i, line in enumerate(lines): 
        print(line)
        date = line.rstrip().split(' ')
        full_path = base_folder + "/" + date[0] + "/" + ' '.join(date) + "/"
        #print(full_path)
        amp3G = AMP3GImageProc(save_directory = data_directory, homography_transform = homography_transform)
        amp3G.image_overlap(full_path, display_images = True, display_overlap = True, save = False)        

    
def view_data(path):
    amp3G = AMP3GImageProc(save_directory = data_directory, homography_transform = homography_transform)
    amp3G.image_overlap(path, display_images = True, display_overlap = True, save = False)
        

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
    view_data('/media/3GAMP/2019_01_17/2019_01_17 10_03_49/')
    #view_data('/media/3GAMP/2019_01_17/2019_01_17 10_05_45/')
    #view_data_lst(data_directory + "/view_data.txt")
        
    
        