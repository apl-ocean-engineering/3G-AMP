#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Fri March 8 12:21:00 2019

@author: mitchell
"""

from amp3GImageProc import imageTransforms

#transforms = imageTransforms("/home/mitchell/3G-AMP-Calibration/2018_12_12 14_16_56/", images_path2 = "/home/mitchell/3G-AMP-Calibration/no_checkerboard/")
transforms = imageTransforms("/media/3GAMP/2019_03_04/2019_03_04 10_31_18/", images_path2 = "/media/3GAMP/2019_03_04/2019_03_04 10_31_18/")

print(transforms.find_homography_check(save = True, path = '../../'))