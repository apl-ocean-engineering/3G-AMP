#!/usr/bin/env python2.7
"""
Created on Wed March 7 16:07:34 2019

@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import glob
import cv2
import numpy as np
import os
import sys
import signal
import logging
import time 
import datetime

def get_paths(directory, flag = False):
    """
    Get list of all files and folders under a specific directory satisfying
    the input form
    
    Input:
        directory (str): Name of directory in which files should be, PLUS 
        the file type, if desired. Must have '*' symbol were searchign will 
        take place. It will return all files that fit the
        specified format, filling in for the '*'
        
        Example 1: path + *.jpg will return all .jpg files in the location
        specified by path
        
        Example 2: path + */ will return all folders in the location 
        specified  by oath
        
        Example 3: path + * will return all folders AND files in the 
        location specified by path
        
    Return:
        paths (list<strs>): List of strings for each file/folder which 
        satisfies the in input style.
        Empty list if no such file/folder exists
    """
    paths = sorted(glob.glob(directory))
    return paths

def sigint_handler(signum, frame):
    """
    Exit system if SIGINT
    """
    sys.exit()     

class BasePath(object):
    def __init__(self, root_dir = " ", affine_transformation = " ", 
                 perspective_transfrom = " ", hour_min = 0.0, hour_max = 24.0): 
        """
        Args:
            [root_dir(string)]: Location of the root directory where images are
            [affine_transformation(str)]: Path to file containing 
                affine_tranformation
            [hour_min(float)]: Minimium hour to consider images. Images which
                are below this amount will be discarded
            [hour_max(float)]: Maximium hour to consider images. Images which
                are above this amount will be discarded
        """
        
        #Dictionary to transform dates from motnh to number
        self.dates = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'April':'04', 
                      'May':'05', 'June':'06', 'July':'07', 'Aug':'08',
                      'Sep':'09', 'Oct':'10', 'Nov':'11', 'Dec':'12'}
        
        #Find physical location of root directory
        if root_dir == " ":
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir
            
        #Specify bounds on system by hour
        self.hour_min = hour_min
        self.hour_max = hour_max
        
        #Point to Triggers.txt file           
        trigger_path = self.root_dir + "/Triggers.txt"
        trigger_file_present = os.path.isfile(trigger_path)
        
        #Check if file exits. If so, load data
        if trigger_file_present:
            #Load all trigger dates
            self.trigger_dates = self._trigger_files(trigger_path)
        else:
            self.trigger_dates = []
            
        #Find all subdirectory folders under root directory
        self.sub_directories = self._subdirs()
        
        #Handle logging
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
    def get_hour(self, name):
        """
        Convert from folder name type to hour in 24-hour format
        
        Input:
            name (str): Input in form YYYY_MM_DD hh_mm_ss
        Return:
            Hour (int): Hour in 0-24 hour form
        """
        full_time = name.split(" ")
        hour = full_time[1].split("_")[0]
               
        return int(hour)             

    def _subdirs(self):
        """
        Returns list of subdirectories under main directory which satisifes
        the time constraint 
        
        Input:
            None
        Return:
            return_subdirs(list<str>): List of all subdirectories which satisfy
                                       input paramaters
        """
        #Get list of all folders in the current directory
        all_subdirs = get_paths(self.root_dir + "/*/")
        
        #Only return sub directories that contain images, i.e. begin with 2018
        return_subdirs = []
        for path in all_subdirs:
            names = path.split('/')
            #all folders start with 20. If a folder starts with 2018, append
            #to return list
            folder_name = names[len(names) - 2]
            
            if folder_name[0:2] == "20":
                #print(names[len(names) - 2])
                hour = self.get_hour(names[len(names) - 2])
                
                if hour > self.hour_min and hour < self.hour_max:
                    
                    return_subdirs.append(path)
        return return_subdirs  
        
    
    def _trigger_files(self, path):
        '''
        Determine all 'trigger events' from file Trigger.txt
        
        Input:
            path(str): Path which poitns to Trigger.txt file
        Return:
            None
        '''
        trigger_dates = set()
        #Open path location
        with open(path, "r") as f:
            for line in f:
                '''
                File is listed out of order and in a different format than
                the file systems where the events actually happen. Must 
                transform
                '''
                day = (line.split(" ")[3]).split("-") #Get the year, month, and date
                time_stamp = (line.split(" ")[4]).split("-")[0].replace(':', '_')[:-1] #Hour, min, and second
                month = self.dates[day[1]] #Determine the month number from month (e.g. 01 for Jan)
                #Transform to folder date form, YYYY-MM-DD
                new_date = day[2] + "_" + month + "_" + day[0] + "_" + time_stamp
                #Append to list and return
                trigger_dates.add(new_date)
                
        
        

class AMP3GImageProc():
    """
    Class containign modules to help process 3G-AMP image data
    
    Attributes:
        -homog (np.mat): Homography transformation matrix between images
        -time_delay_allowed: Maximium allowable time between images timestamps
        -save_directory(str): Location to save events
        
    Methods:
        -image_overlap: Check overlap between stereo images
        -background_subtraction: Runs openCv createBackgroundSubtractorMOG2 alg.
            for all subdirectories under the root directory
        -single_directiory_background_subtraction: Runs openCv 
            createBackgroundSubtractorMOG2 alg. for one subdirectory
        -get_hour: Determines hour from full image/folder name
    """
    
    def __init__(self,  save_directory = ' ', time_delay_allowed = 0.1, 
                                                homography_transform = ' '):
        """
        Class containign modules to help process 3G-AMP image data
        
        Input:
            -[save_directory(str)]: Location to save events
            -[time_delay_allowed(float)]: Maximium allowable time between images timestamps
            -[homography_transform(string)]: Location of homography transform file
        
        Return:
            None
        """        
        if homography_transform == " ":
            self.homog = np.identity(3)
          
        else:
            try:
                file = open(homography_transform, "r") 
                self.homog = np.array(file.read().split(',')[0:9], 
                                     dtype=np.float32).reshape((3,3))  
            except:
                print("Homography file not found")
                sys.exit(0)

        self.time_delay_allowed = time_delay_allowed
        self.save_directory = save_directory
        self.overlap_sum_threshold = 16423336#2000000 #Threshold for overlap
    
    def display_images(self, path):
        """
        Display all images in a directory
        
        Input:
            path(str): Base location of Manta 1 and Manta 2 folders
            
        Return: 
            None
        """
        cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame1', 1200,1200) 

        self._display_images(path + "/Manta 1/*.jpg", 
                    path + "/Manta 2/*.jpg")  
        
    def image_overlap(self, path, display_images = False, display_overlap = False, save = True):
        """
        Check the overlap of images, check image intensity
        Inputs:
            path(str): Base location of Manta 1 and Manta 2 folders
            [display_images(bool)]: Display the images
            [display_overlap(bool)]: Display the overlap between transformed 
                images
        Returns:
            -overlap_intensity(list<float>): List of image overlaps from the 
                defined location
        """ 
        
        overlap_intensity = self._overlap(path + "Manta 1/*.jpg", 
                path + "Manta 2/*.jpg", overlap = True, 
                display_images = display_images, 
                display_overlap= display_overlap,
                color = True, date = None, save = save)    
        #print(overlap_intensity)
        return overlap_intensity
        
            
            
    def _display_images(self, d1, d2, time_delay=1):
        """
        Display all jpgs under d1 and d2 path
        Inputs:
            -d1(str): Points to Manta1 images
            -d1(str): Points to Manta2 images
            -[time_delay(int)]: Time (in ms) to display image for
        Return:
            None
        """           
        images1 = sorted(get_paths(d1), reverse = True)
        images2 = sorted(get_paths(d2), reverse = False)

        images = zip(images1, images2)
        
        for fname1, fname2 in images:
            
            #signal.signal(signal.SIGINT, sigint_handler)
            img1 = cv2.imread(fname1)
            cv2.imshow('frame1',img1)
            k = cv2.waitKey(time_delay)
            if k == 99:
                cv2.destroyAllWindows()
                sys.exit() 
                
    def _overlap(self, d1, d2, overlap = False, 
                                display_images = False, color = False, 
                                display_overlap=False, 
                                date = None, save = True):
        """
        Run background subtraction algorithm using openCv 
        createBackgroundSubtractorMOG2. Will do background subtraction for all
        images in d1 and d2 (directories 1 and 2, respectively). d1 and d2
        inputs should be the directories where each of the stereo camera iamges 
        are located.
        
        The function will also claculate overlap, if desired, and return the 
        intensity of the image(s) overlap. Image one's frame will be transformed 
        into image one's frame and will check for overlap. If false, return 
        empty list
        
        Input:
            d1(str): Directory 1 containing images (i.e subdir + Manta 1)
            d2(str): Directory 2 containing images
            [overlap(bool)]: Calcuate image overlap
            [display_images(bool)]: Display images
            [color(bool)]: Display color or grayscale images
            [display_overlap(bool)]: Display overlapped image
            [date(str)]: Current date (YYYY_MM_DD HH_MM_SS) for saving
            [save(bool)]:To save or not to save
            
        Return:
            overlap_intensity(list<float>): List of all image intensities
        """       
        #get list of all images from current directory
        images1 = sorted(get_paths(d1), reverse = True)
        images2 = sorted(get_paths(d2), reverse = True) 
        
        
        
        #create a background subtraction object
        fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        
        #zip images so that we can loop through image names
        images = zip(images1, images2)
        
        #initialize window size
        if display_images:
            cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame1', 1200,1200) 
            cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame2', 1200,1200) 
        if display_overlap:
            cv2.namedWindow('overlap', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('overlap', 800,800)       
            
        #Create list of image_intensity
        overlap_intensity = []
        
        #Kernel for median blur
        kernel = np.ones((250,250),np.uint8)
        overlap_count = 0
        i = 0
        """
        Log file
        """
        with open(self.save_directory + '\records\log.txt', 'a+') as f:
                now = datetime.datetime.now()
                f.write("%s, %s\n" % (d1.split('/')[-3], now.strftime("%Y_%m_%d %H::%M::%S")))    
                
        for fname1, fname2 in images:
            
            """
            Loop through all image frames and run background subtraction.
            If overlap is selected, compare the overlap between the two 
            images
            """
            signal.signal(signal.SIGINT, sigint_handler)              
            
            
            check_date = self._check_date(fname1, fname2)
            if not check_date:
                fname1, fname2 = self._find_date(images, i)
                if fname1 == None: #No images within allowable time
                    check_date = False
                else:
                    check_date = True
            if check_date:
                #Read images and inport
                img1 = cv2.imread(fname1)
                img2 = cv2.imread(fname2)               

                #Apply the mask
                img1b = fgbg1.apply(img1)#, learningRate=0.035)
                img2b = fgbg2.apply(img2)#, learningRate=0.035)
                
                
                
                ret, thresh1 = cv2.threshold(img1b, 125,255,cv2.THRESH_BINARY)
                ret, thresh2 = cv2.threshold(img2b, 125,255,cv2.THRESH_BINARY)

                #Apply a median blur to reduce noise
                blur1 = cv2.medianBlur(thresh1, 25)
                blur2 = cv2.medianBlur(thresh2, 25)
                
                if overlap:

                    blur1_trans = cv2.warpPerspective(blur2, self.homog, 
                                   (blur1.shape[1],blur1.shape[0]))

                    blur1_trans_dilate = cv2.dilate(
                            blur1_trans,kernel,iterations = 1)
                    blur2_dilate = cv2.dilate(blur1,kernel,iterations = 1)
                    
                    #Check Overlap between images using bitwise_and
                    overlap_img = np.bitwise_and(
                            blur2_dilate, blur1_trans_dilate)
                    overlap_sum = np.sum(overlap_img)
                    #print(overlap_sum, overlap_sum>self.overlap_sum_threshold)
                    if overlap_sum > self.overlap_sum_threshold: #IT ALWAYS FAILS ON THE FIRST TRY
                        overlap_count += 1
                    else:
                        overlap_count = 0
                    if overlap_count >= 4 and save:
                        print("HIGH OVERLAP DETECTED FOR DATE:", d1.split('/')[-3])
                        with open(self.save_directory + '\records\highStereoData_Q.txt', 'a+') as f:  
                            f.write("%s, %s, %s, %s\n" % (d1.split('/')[-3], '/'.join(fname1.split('/')[-2:]), '/'.join(fname2.split('/')[-2:]), now.strftime("%Y_%m_%d %H::%M::%S")))  
                        break
                    

                    overlap_intensity.append(overlap_sum)
                    if display_overlap:
                        cv2.imshow('overlap', overlap_img)
                
                if display_images:
                    if color:
                                               
                        cv2.imshow('frame1',img1)
                        cv2.imshow('frame2',img2)
                    else:
                        cv2.imshow('frame1',blur1)
                        cv2.imshow('frame2',blur2)                            

                if display_images or display_overlap:
                    if i == 0:
                        k = cv2.waitKey(100)
                    else:
                        k = cv2.waitKey(100)
                      
                    if k == 99:
                        cv2.destroyAllWindows()
                        sys.exit()  
            i+=1
            #print(i)
                        
            #Return list of overlap intensities or empty list
        return overlap_intensity
    
    def _check_date(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name
        
        Return:
            Bool: If timestamps are close enough together
            
        
        """
        #return True
        time1 = float('.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float('.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))
        
        if abs(time1 - time2) < self.time_delay_allowed:
            return True
        
        return False
    
    def _time_diff(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name
        
        Return:
            Bool: If timestamps are close enough together
            
        
        """
        time1 = float('.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float('.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))
        
        return abs(time1 - time2)
    
    def _check_day_hour(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name
        
        Return:
            Bool: If timestamps are close enough together
            
        
        """
        day1 = f1.split('/')[-1].split('_')
        day1 = '_'.join(day1[:6])
        day2 = f2.split('/')[-1].split('_')
        day2 = '_'.join(day2[:6])
        if day1 == day2:
            return True
        
        return False
    
    
    def _find_date(self, images, i):
        fname1 = images[i][0]
        fname2 = images[0][1]
        prev_time_diff = self._time_diff(fname1, fname2)
        
        for loc in range(0, len(images)):
            if self._check_day_hour(fname1, images[loc][1]):
                if self._check_date(fname1, images[loc][1]):
                    
                    return fname1, images[loc][1]
                #print(fname1, images[loc][1], self._time_diff(fname1, images[loc][1]) > prev_time_diff)
                if self._time_diff(fname1, images[loc][1]) > prev_time_diff:
                    #Diverging, break
                    return None, None #fname1, images[loc-1][1]
                prev_time_diff = self._time_diff(fname1, images[loc][1]) 
        return None, None
                

      
class imageTransforms(object):
    """
    Class to help determine transformation between frames in two 3G-AMP cameras
    
    Attributes:
        -Images_path(str): Path to directory containing images for calibration
        -x#_points, y#_points (list<float>): 4 lists containing corresponding
            points in each camera frame
        -image1, image2 (np.mat<float>): Images
        -m1_subdirectories, m2_subdirectories (list<str>): List 
            containing image paths
    Methods:
        -corresponding_image_points: Manual correspondance of points between 
            two image frames
        -find_perspective: Calculates the perspective transform matrix
        -find_homography: Calculates the homography transform matrix
        -find_affine: Calculates the affine transform matrix
        -get_points: Returns corresponding image points between the two frames
        
    """
    def __init__(self, images_path, images_path2 = " "):
        """
        Args:
            images_path(str): Path pointing to location of images
        """
        
        self.images_path = images_path
        self.images_path2 = images_path2
        self.x1_points = []
        self.y1_points = []
        self.x2_points = []
        self.y2_points = []
        self.image1 = np.zeros([0,0])
        self.image2 = np.zeros([0,0])
        
        self.m1_subdirectories, self.m2_subdirectories, self.m1_subdirectories2, self.m2_subdirectories2  = self._subdirs()
        
        self.time_delay_allowed = 0.05
        
    def corresponding_image_points(self):
        """
        Determine coressponding image points between the frames
        
        Will display two WAMP images. User must click on identical point
        in two frames. x#_points, and y#_points will populate as the user 
        clicks on points            
        """
        #Initalzie image windows
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image1', 1200,1200)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image2', 1200,1200)
        #Define mouse callback functions
        cv2.setMouseCallback('image1',self._mouse_click1)
        cv2.setMouseCallback('image2',self._mouse_click2)
        #print(self.m1_subdirectories)
        #Loop through all images in subdirectory location
        print("Click on the same point in both images")
        print("Press enter to move to next corresponding images")
        print("Press 'f' to finish")
        print("Press cntrl+c to quit")
        for i in range(0, len(self.m1_subdirectories)):
            signal.signal(signal.SIGINT, self._sigint_handler)
            #Get img1 and img2 from the subdirectories
            f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]
            
            self.img1, self.img2 = cv2.imread(f1), cv2.imread(f2)
            #Show images
            cv2.imshow('image1', self.img1)
            cv2.imshow('image2', self.img2)
            #Press 'enter' to move on, f to finish, cntrl+c to quit
            k = cv2.waitKey(0)
            if k == 99:
                cv2.destroyAllWindows()
                sys.exit()  
            if k == 102:
                cv2.destroyAllWindows()
                break            
        cv2.destroyAllWindows()
        
        
    def find_perspective(self, save = False, path=""):
        """
        Calculate perpsective transformation matrix from corresponding points
        
        NOTE: Must be exactly 4 points, or error will raise
        
        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired
        
        Return: 
            perspective_transform (np.mat<float>): (3X3) transformation matrix    
        """
        
        #Get corresponding points
        pnts1, pnts2 = self._image_points()
        if len(pnts1)!=4 or len(pnts2)!=4:
            raise ValueError("Must have exactly four corresponding image points")
        #Get transform
        perspective_transform = cv2.getPerspectiveTransform(pnts1, pnts2)
        if save:
            #Save data to text file
            np.savetxt(path+"perspective_transform.txt", 
                       perspective_transform.reshape(1,9), 
                       delimiter=',', fmt="%f") 
        
        return perspective_transform
    
    def find_homography(self, save = False, path =""):
        """
        Calculate homography transformation matrix from corresponding points
        
        Should use at least four points to be accruate
        
        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired
        
        Return: 
            homography_transform (np.mat<float>): (2X3) homography matrix
        
        #Get corresponding points
        pnts1, pnts2 = self._image_points()
        #Get transform
        homography_transform = cv2.findHomography(pnts1, pnts2)
        
        if save:
            #Save data to text file
            np.savetxt(path+"3Ghomography_transform.txt", 
                       np.array(homography_transform[0]).reshape(1,9), 
                       delimiter=',', fmt="%f")         
        
        return homography_transform   
        
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image1', 800,800)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image2', 800,800)
        cv2.namedWindow('image3', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image3', 800,800)
        """
        images = zip(self.m1_subdirectories, self.m2_subdirectories)
        for i in range(0, len(self.m1_subdirectories)):
            #print(i)
            try:
                signal.signal(signal.SIGINT, sigint_handler)
                #Get img1 and img2 from the subdirectories
                f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]
                
                check_date = self._check_date(f1, f2)
            
                if not check_date:
                    f1, f2 = self._find_date(images, i)
                    if f1 == None: #No images within allowable time
                        check_date = False
                    else:
                        check_date = True
                if check_date:
                    img1, img2 = cv2.imread(f1), cv2.imread(f2)
                    
                    orb = cv2.ORB_create()
                    kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),None)
                    kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),None)
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    
                    matches = bf.match(des1,des2)
                    for mat in matches:
                        #print(mat.queryIdx)
                        #print(mat.trainIdx)
                        #pnts1.append(kp1[mat.queryIdx].pt)
                        #pnts2.append(kp2[mat.trainIdx].pt)
                        self.x1_points.append(kp1[mat.queryIdx].pt[0])
                        self.y1_points.append(kp1[mat.queryIdx].pt[1])
                        self.x2_points.append(kp2[mat.trainIdx].pt[0])
                        self.y2_points.append(kp2[mat.trainIdx].pt[1])
                        #print(pnts1)
                        """
                        matches = sorted(matches, key = lambda x:x.distance)
                        #Show images
                        cv2.imshow('image1', img1)
                        cv2.imshow('image2', img2)
                        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],img1, flags=2)
                        cv2.imshow('image3', img3)
                        #print(matches)
            
                        k = cv2.waitKey(1)
                        if k == 99:
                            cv2.destroyAllWindows()
                            sys.exit()  
                        if k == 102:
                            cv2.destroyAllWindows()
                            break 
                        """
                        
                        
                    
            except: 
                pass
        pnts1, pnts2 = self.get_points()
        
        fundmental = cv2.findFundamentalMat(pnts1, pnts2)
        
        print(fundmental)
        
        ret, H1, H2 = cv2.stereoRectifyUncalibrated(pnts1, pnts2, fundmental[0], (img2.shape[1],img2.shape[0]))
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img', 800,800)
        cv2.namedWindow('overlay', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('overlay', 800,800)
        """
        cv2.namedWindow('overlay1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('overlay1', 800,800)
        cv2.namedWindow('overlay2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('overlay2', 800,800)   
        """
        
        for i in range(0, len(self.m1_subdirectories)):
            f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]
            
            check_date = self._check_date(f1, f2)
            
            if not check_date:
                f1, f2 = self._find_date(images, i)
                if f1 == None: #No images within allowable time
                    check_date = False
                else:
                    check_date = True
            if check_date:
            
                img1 = cv2.imread(f1)
                img2 = cv2.imread(f2)        
                dst1 = cv2.warpPerspective(img1, H1, (img1.shape[1],img1.shape[0]))
                dst2 = cv2.warpPerspective(img2, H2, (img2.shape[1],img2.shape[0]))
                
                overlay1 = cv2.addWeighted(dst1, 1.0, dst2, 1.0, 0.0)
                overlay2 = cv2.addWeighted(dst1, 1.0, img2, 1.0, 0.0)
                overlay3 = cv2.addWeighted(dst2, 1.0, img1, 1.0, 0.0)
                
                
                """
               
                homography_transform = cv2.findHomography(pnts1, pnts2)
                
                affine_transformation = cv2.getAffineTransform(pnts1, pnts2)
                
                print(affine_transformation)
                
                dst = cv2.warpAffine(img1, affine_transformation, (img2.shape[1],img2.shape[0]))
                
                
                dst = cv2.warpPerspective(img1, homography_transform[0], (img2.shape[1],img2.shape[0]))
                
                dst2 = cv2.addWeighted(img2, 1.0, dst, 1.0, 0.0)
                """
                
                #cv2.imshow("dst", dst1)
                cv2.imshow("img", img1)
                #cv2.imshow("dst2", dst2)
                #print(f1, f2)
                cv2.imshow("overlay", overlay1)  
                #cv2.imshow("overlay1", overlay2) 
                #cv2.imshow("overlay2", overlay3) 
                
                cv2.waitKey(100)
                """
                print(homography_transform)
                if save:
                    #Save data to text file
                    np.savetxt(path+"3Ghomography_transform.txt", 
                               np.array(homography_transform[0]).reshape(1,9), 
                               delimiter=',', fmt="%f")         
            
            return homography_transform 
            """
    
    def stereo_rectify(self,path ="",save = False):
        intrinsics1 = path + "camera1/intrinsic_matrix.csv"
        file = open(intrinsics1, "r") 
        K1 = np.array((file.read().replace('\n', ',')).split(',')[0:9], 
                              dtype=np.float64).reshape((3,3))   
        intrinsics2 = path + "camera2/intrinsic_matrix.csv"
        file = open(intrinsics2, "r") 
        K2 = np.array((file.read().replace('\n', ',')).split(',')[0:9], 
                              dtype=np.float64).reshape((3,3))        
        distortion1 = path + "camera1/distortion_coeffs.csv"
        file = open(distortion1, "r") 
        d1 = np.array((file.read().replace('\n', ',')).split(',')[0:5], 
                              dtype=np.float64).reshape((1,5))[0]           
        distortion2 = path + "camera2/distortion_coeffs.csv"
        file = open(distortion2, "r") 
        d2 = np.array((file.read().replace('\n', ',')).split(',')[0:5], 
                              dtype=np.float64).reshape((1,5))[0]        
        imsize = (2056, 2464)
        rotation = path + "rotation_matrix.csv"
        file = open(rotation, "r") 
        R = np.array((file.read().replace('\n', ',')).split(',')[0:9], 
                              dtype=np.float64).reshape((3,3))       
        translation = path + "translation_matrix.csv"
        file = open(translation, "r") 
        T = np.array((file.read().replace('\n', ',')).split(',')[0:3], 
                              dtype=np.float64).reshape((1,3))[0]
        
        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(K1, d1, K2, d2,  imsize,
                                                            R, T, alpha=-1)
        
        cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img1', 800,800)
        cv2.namedWindow('blur1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blur1', 800,800)
        cv2.namedWindow('blur2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('blur2', 800,800)        
        
        
        fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        kernel = np.ones((15,15),np.uint8)
        images = zip(self.m1_subdirectories, self.m2_subdirectories)
        for i in range(0, min(len(self.m1_subdirectories), 100)):
                signal.signal(signal.SIGINT, sigint_handler)
                #Get img1 and img2 from the subdirectories
                f1, f2 = self.m1_subdirectories[i], self.m2_subdirectories[i]
                
                check_date = self._check_date(f1, f2)
            
                if not check_date:
                    f1, f2 = self._find_date(images, i)
                    if f1 == None: #No images within allowable time
                        check_date = False
                    else:
                        check_date = True                
                
                if check_date:
                    img1, img2 = cv2.imread(f1), cv2.imread(f2)       
                    img1 = cv2.imread(f1)
                    img2 = cv2.imread(f2)
                    
                    img1b = fgbg1.apply(img1)#, learningRate=0.035)
                    img2b = fgbg2.apply(img2)#, learningRate=1.0)
                    #gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    #gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    stereo = cv2.StereoBM_create(numDisparities=320, blockSize=5)
                    disparity = stereo.compute(img1b,img2b)
                    cv2.imshow("img1", img1b)
                    cv2.imshow("blur1", disparity)
                    """
                    stereo = cv2.StereoBM_create(numDisparities=640, blockSize=9)
                    #stereo = cv2.StereoSGBM_create(numDisparities = 16, blockSize=9)#P1=33948672,P2=135794668)
                    #stereo = cv2.StereoSGBM_create(minDisparity=10)
                    disparity = stereo.compute(gray1,gray2)
                    disparity_blur = np.array(disparity)#, dtype=np.uint8)
                    #disparity_blur = cv2.convertScaleAbs(disparity_blur)
                    #disparity_blur = cv2.bitwise_not(disparity_blur)
                    _max = disparity_blur.max()
                    #print(disparity_blur[disparity_blur>-16])
                    #disparity_blur[disparity_blur < 16] = -255
                    #disparity_blur[disparity_blur != -16] = -16
                    #disparity_blur[disparity_blur == -255] = _max
                    #print(disparity_blur[disparity_blur != -16])
                    #for i,val in enumerate(disparity_blur):
                    #    print(val)
                    #    if val < 15:
                    #        disparity_blur[i] = 255
                    #    else:
                    #        disparity_blur[i] = 0
                    #disparity_blur[disparity_blur > 250] =0# disparity_blur.max()
                    #disparity_blur[disparity_blur == -1] =255
                    #print(disparity_blur.min(), disparity_blur.max())
                    #disparity_blur = cv2.fastNlMeansDenoising(disparity_blur)#, h = 200, templateWindowSize=9)
                    
                    
                    _3dImg = cv2.reprojectImageTo3D(disparity, Q)
                    proj_image = (_3dImg, R, T, K1, d1)
                    proj_image = np.array(proj_image[0])
                    img1b = fgbg1.apply(img1)#, learningRate=0.035)
                    img2b = fgbg2.apply(proj_image)#, learningRate=1.0)
                    proj_image[proj_image == np.inf] = 0
                    proj_image[proj_image == -np.inf] = 0
                    proj_gray = cv2.cvtColor(proj_image, cv2.COLOR_BGR2GRAY)
                    np.nan_to_num(proj_gray)
                    #ret, thresh1 = cv2.threshold(proj_gray, 0,255,cv2.THRESH_BINARY)
                    #proj_gray = np.array(proj_gray, dtype=int)
                    
                    blur1 = cv2.medianBlur(img1b, 3)
                    blur2 = cv2.fastNlMeansDenoising(img2b, h = 50, templateWindowSize=9)
                    #blur2 = cv2.medianBlur(blur2, 25)
                    proj_image[proj_image == np.inf] = 0
                    proj_image[proj_image == -np.inf] = 0
                    proj_gray = cv2.cvtColor(proj_image, cv2.COLOR_BGR2GRAY)
                    np.nan_to_num(proj_gray)
                    print(proj_gray.dtype)
                    #proj_gray[proj_gray != np.float32] = 0
                    
                    #proj_gray[proj_gray<=255]=0
                    #dilate2 = cv2.dilate(
                    #        blur2,kernel,iterations = 1)
                    #print(img1b)
                    
                    #cv2.imshow("img1", img1b)
                    cv2.imshow("blur1", proj_gray)
                    #proj_gray = np.array(proj_gray, dtype=np.uint8)
                    #print(proj_gray.max())
                    #cv2.imshow("blur2", disparity)
                    #print(gray1.dtype, proj_gray.dtype)
                    #overlay1 = cv2.addWeighted(gray1, 1.0, proj_gray, 1.0, 0.0)
                            """           
                    k = cv2.waitKey(100)
                    if k!=-1:
                        sys.exit(1)
            
        
    def find_affine(self, save=False, path = ""):
        """
        Calculate affine transformation matrix from corresponding points
        
        NOTE: Must be exactly 3 points, or error will raise
        
        Args:
            [save(bool)]: If the data should be saved or not
            [path(str)]: Path to save, is desired
        
        Return: 
            homography_transform (np.mat<float>): (2X3) homography matrix
        """     
        #Get corresponding points
        pnts1, pnts2 = self._image_points()
        if len(pnts1)!=3 or len(pnts2)!=3:
            raise ValueError("Must have exactly three corresponding image points")   
        #Get affine transform
        affine_transform = cv2.getAffineTransform(pnts1, pnts2)
        
        if save:
            #Save data to text file
            np.savetxt(path+"affine_transform.txt", 
                       affine_transform.reshape(1,6), delimiter=',', 
                       fmt="%f") 
            
        return affine_transform     
      
    def get_points(self):
        """
        Return corresponding image points
        
        Return:
            points1, points2 (list<tuple<float>>): Corresponding points
        """
        points1, points2 = self._image_points()

        return points1, points2
    
            
    def _image_points(self):
        """
        Organize image points into two lists of corresponding tuples
        
        Return:
            pnts1, pnts2 (list<tuple<float>>): Corresponding points
        """
        #Check that points clicked are equal
        if len(self.x1_points) != len(self.x2_points):
            raise AttributeError("Unequal Points Clicked")
        #Organize points
        pnts1 = []
        pnts2 = []
        points1 = []
        points2 = []
        for i in range(0, len(self.x1_points)):
            pnts1.append(np.array([[np.float32(self.x1_points[i]), 
                                    np.float32(self.y1_points[i])]]))
            pnts2.append(np.array([[np.float32(self.x2_points[i]), 
                                    np.float32(self.y2_points[i])]]))
        
        #Must be float 32s to work in OpenCV
        pnts1 = np.float32(pnts1)
        pnts2 = np.float32(pnts2)
        '''
        pnts1 = [pnts1]
        pnts2 = [pnts2]
        for i in range(0, 4):
            points1.append(pnts1)
            points2.append(pnts2)
        
        return points1[0], points2[0]
        '''
        return pnts1, pnts2
    
    def _mouse_click1(self,event,x,y,flags,param):
        """
        Callback function for mouse click event on image1 frame
        
        Places clicked points into x1_ and y1_points lists
        """
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x1_points.append(x)
            self.y1_points.append(y)
            #Draw circle where clicked
            cv2.circle(self.img1,(x,y), 20, (255,0,0), -1)
            cv2.imshow('image1', self.img1)
            
    
    def _mouse_click2(self,event,x,y,flags,param):
        """
        Callback function for mouse click event on image2 frame
        
        Places clicked points into x2_ and y2_points lists
        """        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.x2_points.append(x)
            self.y2_points.append(y)
            #Draw circle where clicked
            cv2.circle(self.img2,(x,y), 20, (255,0,0), -1)
            cv2.imshow('image2', self.img2)
    
    def _subdirs(self):
        """
        Return list of all subdirectories under current directory containing
        the Manta 1 and Manta 2 images
        
        Return:
            -manta1_subdirs, manta2_subdirs (list<str>): Paths for all images
        """
        #Get list of all folders in the current directory
        manta1_subdirs = sorted(glob.glob(self.images_path + "/Manta 1/*.jpg"), reverse = True)
        manta2_subdirs = sorted(glob.glob(self.images_path + "/Manta 2/*.jpg"), reverse = True)
        manta1_subdirs2 = None
        manta1_subdirs2 = None
        if self.images_path2 != " ":
            manta1_subdirs2 = sorted(glob.glob(self.images_path2 + "/Manta 1/*.jpg"), reverse = True)
            manta1_subdirs2 = sorted(glob.glob(self.images_path2 + "/Manta 2/*.jpg"), reverse = True)
        
        return manta1_subdirs, manta2_subdirs, manta1_subdirs2, manta1_subdirs2
    
    def _sigint_handler(self, signum, frame):
        """
        Exit system if SIGINT
        """
        sys.exit()  
        
    def _check_date(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name
        
        Return:
            Bool: If timestamps are close enough together
            
        
        """
        #return True
        time1 = float('.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float('.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))
        
        if abs(time1 - time2) < self.time_delay_allowed:
            return True
        
        return False
            


    def _time_diff(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name
        
        Return:
            Bool: If timestamps are close enough together
            
        
        """
        time1 = float('.'.join(f1.split('/')[-1].split('_')[-1].split('.')[0:2]))
        time2 = float('.'.join(f2.split('/')[-1].split('_')[-1].split('.')[0:2]))
        
        return abs(time1 - time2)
    
    def _check_day_hour(self, f1, f2):
        """
        Verify that the image timestamps are less than self.time_delay_allowed apart
        Inputs:
            f1(str): Frame1 name
            f2(str): Frame2 name
        
        Return:
            Bool: If timestamps are close enough together
            
        
        """
        day1 = f1.split('/')[-1].split('_')
        day1 = '_'.join(day1[:6])
        day2 = f2.split('/')[-1].split('_')
        day2 = '_'.join(day2[:6])
        if day1 == day2:
            return True
        
        return False
    
    
    def _find_date(self, images, i):
        fname1 = images[i][0]
        fname2 = images[0][1]
        prev_time_diff = self._time_diff(fname1, fname2)
        
        for loc in range(0, len(images)):
            if self._check_day_hour(fname1, images[loc][1]):
                if self._check_date(fname1, images[loc][1]):
                    
                    return fname1, images[loc][1]
                #print(fname1, images[loc][1], self._time_diff(fname1, images[loc][1]) > prev_time_diff)
                if self._time_diff(fname1, images[loc][1]) > prev_time_diff:
                    #Diverging, break
                    return None, None #fname1, images[loc-1][1]
                prev_time_diff = self._time_diff(fname1, images[loc][1]) 
        return None, None    


    def _beyond_date(self, date, start_date):
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
    