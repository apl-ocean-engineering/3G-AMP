#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 09:26:47 2019

@author: mitchell
"""

from amp3GImageProc import imageTransforms

#transforms = imageTransforms("/home/mitchell/3G-AMP-Calibration/2018_12_12 14_16_56/")
  
transforms = imageTransforms("/media/3GAMP/2019_03_01/2019_03_01 10_34_53/")


transforms.stereo_rectify(path="/home/mitchell/WAMP_workspace/calibration/3G-AMP/")