#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Mon March 11 15:06:00 2019

@author: mitchell
"""

from amp3GImageProc import AMP3GImageProc
import threading
import time
import os

import inotify.adapters

main_directory = os.path.dirname(os.path.realpath(__file__))
data_directory = main_directory + '/data'
homography_transform = data_directory + '/3Ghomography_transform.txt'

def runamp3G(base_path):
    amp3G = AMP3GImageProc(save_directory = data_directory, 
                homography_transform = homography_transform)
                
    amp3G.image_overlap(base_path, save = False)
               


def main(base_path, sleep):
    i = inotify.adapters.InotifyTree(base_path, block_duration_s=0.1)
    for event in i.event_gen():
        if event:
            if event[1][0] == "IN_ATTRIB":
                #Add a small delay incase both images have not been added
                time.sleep(0.5)
                
                t = threading.Thread(name='runamp3G', target=runamp3G, args=(base_path,))
                t.start()    


if __name__ == '__main__':
    base_path = main_directory + "/tempAMP/"
    main(base_path, 10)