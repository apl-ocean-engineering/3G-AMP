#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
Created on Fri March 8 12:21:00 2019

@author: mitchell
"""

from amp3GImageProc import imageTransforms

transforms = imageTransforms("/home/mitchell/3G-AMP-Calibration/2018_12_12 14_16_56/")

print(transforms.find_homography(save = True, path = '../../'))