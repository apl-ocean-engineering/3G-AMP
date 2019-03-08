#!/usr/bin/env python2.7
"""
Created on Wed Nov 14 16:07:34 2018

@author: Mitchell Scott
@contact: miscott@uw.edu
"""
import glob
import cv2
import numpy as np
import os
import sys
import signal
import time
import logging


class AMP3GImageProc:
    """
    Class containign modules to help process AMP3G image data
    
    Attributes:
        -root_dir (str): Location of root directory
        -H (np.mat): Linear transformation matrix between images
        -trigger_dates: Dates which were specified by 'Trigger.txt'
        -high_overlap (float): Overlaps above this value are specfied as having
            a higher than normal overlap
        -high_overlap_list (list<str>): List of all folders which contains a 
            set of images with a background subtracted overlap above 
            trigger_value
        -sub_directories (list<str>): List of all subdirectories
        -ignore_events (list<str>): All events which should be ignored in a
            background subtraction event
        
    Methods:
        -image_overlap: Check overlap between stereo images
        -background_subtraction: Runs openCv createBackgroundSubtractorMOG2 alg.
            for all subdirectories under the root directory
        -single_directiory_background_subtraction: Runs openCv 
            createBackgroundSubtractorMOG2 alg. for one subdirectory
        -get_hour: Determines hour from full image/folder name
    """
    
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
        ValueError:
            If the AMP3GImageProc is initated such that no directories and 
            images are found, a ValueError will be raised
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
        
        #Affine transformation
        if affine_transformation == " ":
            self.H = np.array([[0.948657,0.262937,-226.404411], 
                               [-0.210863,0.981926, 285.607620]])      

        else:
            file = open(affine_transformation, "r") 
            self.H = np.array(file.read().split(',')[0:6], 
                              dtype=np.float32).reshape((2,3))
            
        if perspective_transfrom == " ":
            self.pers = np.identity(3)
          
        else:
            file = open(perspective_transfrom, "r") 
            self.pers = np.array(file.read().split(',')[0:9], 
                                 dtype=np.float32).reshape((3,3))            
        
            
        #Point to Triggers.txt file           
        trigger_path = self.root_dir + "/Triggers.txt"
        trigger_file_present = os.path.isfile(trigger_path)
        #Check if file exits. If so, load data
        if trigger_file_present:
            #Load all trigger dates
            self.trigger_dates = self._trigger_files(trigger_path)
        else:
            self.trigger_dates = []
        
        #Find all subdirectory folders under roo directory
        self.sub_directories = self._subdirs()
        #Specify which value will trigger a high_value event
        self.high_overlap = 1.60957598*10**8
        #Initalized attributes
        self.high_overlap_list = []       
        self.ignore_events = []
        
        self.logger = logging.getLogger()
        handler = logging.StreamHandler()
        logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        
        

    def image_overlap(self, display_images = False, display_overlap = False,
                          ignore_triggers = False, only_triggers = False):
        """
        Check the overlap of images, check image intensity
        Inputs:
            [display_images(bool)]: Display the images
            [display_overlap(bool)]: Display the overlap between transformed 
                images
        Returns:
            None
        """         
        overlap_intensity = []
        for subdir in self.sub_directories:
            #subdir is a string
            #Call self._background_subtraction with overlap on
            print(subdir)
            date =  subdir.split('/')[4].split(' ')
            date = date[0] + '_' + date[1]
            if date in self.trigger_dates:
                overlap_intensity.extend(
                        self._background_subtraction(subdir + "Manta 1/*.jpg", 
                        subdir + "Manta 2/*.jpg", overlap = True, 
                        display_images = display_images, 
                        display_overlap= display_overlap,
                        ignore_triggers = ignore_triggers, 
                        only_triggers = only_triggers, color = True, date = date))    
        return overlap_intensity
            
    def background_subtraction(self, ignore_triggers = False, 
                               only_triggers = False, display_images = False):
        """
        Run a background subtraction algorithm on all images in desired directory
        
        Input:
            [display_images(bool)]: Display the images
        Return:
            None
        """            
        if ignore_triggers:
            self.ignore_events = self.trigger_dates      
        else:
            self.ignore_events = []
        
        for subdir in self.sub_directories:
            #subdir is a string
            #Call self._background_subtraction. Return value is Null
            
            self._background_subtraction(subdir + "Manta 1/*.jpg", 
                    subdir + "Manta 2/*.jpg", display_images = display_images,
                    ignore_triggers = ignore_triggers, 
                    only_triggers = only_triggers)     
            
    def display_images(self, ignore_dates = False,  display_images = [], time_delay = 1):
        """
        Display all images in a directory
        """
        cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame1', 1200,1200) 
        #cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('frame2', 800,800) 
        for subdir in self.sub_directories: 
            date =  subdir.split('/')[4].split(' ')
            date = date[0] + '_' + date[1]
            if ignore_dates:
                if date in display_images:
                    #print(subdir)
                    self._display_images(subdir + "Manta 1/*.jpg", 
                            subdir + "Manta 2/*.jpg", time_delay = time_delay)   
            else:
                self._display_images(subdir + "Manta 1/*.jpg", 
                        subdir + "Manta 2/*.jpg", time_delay = time_delay)                
            
            
    def single_directiory_background_subtraction(self, directory, 
                                                 only_triggers = False,
                                                 display_overlap= True):
        '''
        Runs background subtraction on all images in desired subfolder location
        
        Input:
            directory(str): Directory to run background subtraction on
        Return:
            None
        '''
        #Specify folder lcoation
        dir1 = self.root_dir + '/' + directory.strip('\n') + "/Manta 1/*.jpg"
        dir2 = self.root_dir + '/' + directory.strip('\n') + "/Manta 2/*.jpg"
        #Run background subtraction
        self._background_subtraction(dir1, dir2, overlap = True, 
                            display_images = True, color = True,
                            display_overlap= display_overlap, flag = True,
                            only_triggers = only_triggers)
            
        

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
        all_subdirs = self._get_paths(self.root_dir + "/*/")
        #Only return sub directories that contain images, i.e. begin with 2018
        return_subdirs = []
        for path in all_subdirs:
            names = path.split('/')
            #all folders start with 2018. If a folder starts with 2018, append
            #to return list
            folder_name = names[len(names) - 2]
            if folder_name[0:2] == "20":
                hour = self.get_hour(names[len(names) - 2])
                if hour > self.hour_min and hour < self.hour_max:
                    return_subdirs.append(path)
        
        return return_subdirs    

        
    def _get_paths(self, directory, flag = False):
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
                time = (line.split(" ")[4]).split("-")[0].replace(':', '_')[:-1] #Hour, min, and second
                month = self.dates[day[1]] #Determine the month number from month (e.g. 01 for Jan)
                #Transform to folder date form, YYYY-MM-DD
                new_date = day[2] + "_" + month + "_" + day[0] + "_" + time
                #Append to list and return
                trigger_dates.add(new_date)
        
        return trigger_dates
                
                
    
    def _background_subtraction(self, d1, d2, overlap = False, 
                                display_images = False, color = False, 
                                display_overlap=False, flag = False, 
                                ignore_triggers = False, only_triggers = False, date = None):
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
            [display_overlap(bool)]: Display overlapped image
        Return:
            overlap_intensity(list<float>): List of all image intensities
        """
        folder = d1.split('/')[4].replace(' ', '_')
        folder_triggered = folder in self.trigger_dates
        
        #In Some cases, we may want to skip folders which were or were no triggered
        run = False
        #If this folder was not in Trigger.txt but it should only run when it 
        #was in Trigger.txt, skip
        if not ignore_triggers and not only_triggers:
            run = True
        else:
            if ignore_triggers:
                if not folder_triggered:
                    run = True
            elif only_triggers:
                if folder_triggered:
                    run = True
                
        if run:
            #get list of all images from current directory
            images1 = self._get_paths(d1)
            images2 = self._get_paths(d2)
            #create a background subtraction object
            fgbg1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            fgbg2 = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
            #zip images so that we can loop through image names
            images = zip(images1, images2)
            #initialize window size
            if display_images:
                cv2.namedWindow('frame1', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame1', 800,800) 
                cv2.namedWindow('frame2', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('frame2', 800,800) 
            if display_overlap:
                cv2.namedWindow('overlap', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('overlap', 800,800)    
            if display_overlap or display_images:
                print("To exit, press cntrl + c")
            #Create list of image_intensity
            overlap_intensity = []
            #Kernel for median blur
            kernel = np.ones((5,5),np.uint8)
            i = 0
            for fname1, fname2 in images:
                """
                Loop through all image frames and run background subtraction.
                If overlap is selected, compare the overlap between the two 
                images
                """
                date_time1 = fname1.split("/")
                #print(date_time1)
                if date_time1[-1][:-7] not in self.ignore_events:
                    signal.signal(signal.SIGINT, self._sigint_handler)
                    #Read images and inport
                    img1 = cv2.imread(fname1)
                    img2 = cv2.imread(fname2)
                    #Apply the mask
                    img1b = fgbg1.apply(img1)
                    img2b = fgbg2.apply(img2)
                    #Apply a median blur to reduce noise
                    blur1 = cv2.medianBlur(img1b, 3)
                    blur2 = cv2.medianBlur(img2b, 3)
                    #Display images
                        #If cntrl + c, exit          
                    if overlap:
                        #Transform image one by an affine transformation
                        blur1_trans = cv2.warpAffine(blur1, self.H, 
                                        (blur1.shape[1],blur1.shape[0]))
                        blur1_trans_dilate = cv2.dilate(
                                blur1_trans,kernel,iterations = 1)
                        blur2_dilate = cv2.dilate(blur2,kernel,iterations = 1)
                        #Check Overlap between images using bitwise_and
                        overlap_img = np.bitwise_and(
                                blur1_trans_dilate, blur2_dilate)
                        overlap_sum = np.sum(overlap_img)
                        if overlap_sum > self.high_overlap:
                            self.high_overlap_list.append(fname1.split("/")[4])
                        overlap_intensity.append(overlap_sum)
                        if display_overlap:
                            cv2.imshow('overlap', overlap_img)
                            #cv2.imshow('overlap2', overlap2)
                    if display_images:
                        if color:
                            cv2.imshow('frame1',img1)
                            cv2.imshow('frame2',img2)
                        else:
                            cv2.imshow('frame1',blur1)
                            cv2.imshow('frame2',blur2)                            
                            
                    #If cntrl+c is pressed, exit
                    #time.sleep(0.1)
                    if display_images or display_overlap:
                        k = cv2.waitKey(1)
                        #print('pics/' + date + '.png')
                        #if k == 13:
                        #cv2.imwrite('pics/' + date + str(i) + 'img.png', img1)
                        #cv2.imwrite('pics/' + date + str(i) + 'blur.png', blur1)
                        #cv2.imwrite('pics/' + date + str(i) + 'img.png', img1)
                        #cv2.imwrite('pics/' + date + str(i) + 'blur.png', blur1)                        
                        if k == 99:
                            cv2.destroyAllWindows()
                            sys.exit()  
                    i+=1
                            
                #Return list of overlap intensities or empty list
            return overlap_intensity
        return []
    
    def _display_images(self, d1, d2, time_delay=1):
        images1 = self._get_paths(d1)
        images2 = self._get_paths(d2)
        images = zip(images1, images2)
        
        for fname1, fname2 in images:
            
            #print(fname1)
            #date_time1 = fname1.split("/")
            #if date_time1[-1][:-7] not in self.ignore_events:
            signal.signal(signal.SIGINT, self._sigint_handler)
            img1 = cv2.imread(fname1)
            #img2 = cv2.imread(fname2)
            cv2.imshow('frame1',img1)
            #cv2.imshow('frame2',img2)
        
            k = cv2.waitKey(time_delay)
            if k == 13:
                date_time1 = fname1.split("/")[-1][:-7]
                open_directory = os.getcwd()
                open_directory = open_directory.split('/')
                open_directory = open_directory[:len(open_directory)-1]
                open_directory = "/".join(open_directory) + '/data/fish_present'
                with open(open_directory+'/dates.txt', 'a+') as f:
                    f.write(date_time1+'\n')   
                    msg = 'Saved data! File: %s' % (date_time1)
                    logging.info(msg)
            if k == 99:
                cv2.destroyAllWindows()
                sys.exit()  
        
    
    def _sigint_handler(self, signum, frame):
        """
        Exit system if SIGINT
        """
        sys.exit()
        
class imageTransforms(object):
    """
    Class to help determine transformation between frames in two AMP3G cameras
    
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
    def __init__(self, images_path):
        """
        Args:
            images_path(str): Path pointing to location of images
        """
        self.images_path = images_path
        self.x1_points = []
        self.y1_points = []
        self.x2_points = []
        self.y2_points = []
        self.image1 = np.zeros([0,0])
        self.image2 = np.zeros([0,0])
        
        self.m1_subdirectories, self.m2_subdirectories = self._subdirs()
        
    def corresponding_image_points(self):
        """
        Determine coressponding image points between the frames
        
        Will display two AMP3G images. User must click on identical point
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
        """
        #Get corresponding points
        pnts1, pnts2 = self._image_points()
        #Get transform
        homography_transform = cv2.findHomography(pnts1, pnts2)
        
        if save:
            #Save data to text file
            np.savetxt(path+"homography_transform.txt", 
                       homography_transform.reshape(1,6), 
                       delimiter=',', fmt="%f")         
        
        return homography_transform   

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
        manta1_subdirs = sorted(glob.glob(self.images_path + "/Manta 1/*.jpg"))
        manta2_subdirs = sorted(glob.glob(self.images_path + "/Manta 2/*.jpg"))
        
        return manta1_subdirs, manta2_subdirs
    
    def _sigint_handler(self, signum, frame):
        """
        Exit system if SIGINT
        """
        sys.exit()                         
    
class calibration(imageTransforms):
    #TODO: Why can't I inherit mouse click functions?? 
    """
    Class to help with image intrinsic and stereo calibration using OpenCV. 
    Inherits from imageTransforms
    
    Methods:
        -intrinsic_calibration: Intrinsic callibration using checkerboard method
        -stereo_calibrate: Stereo calibration between two stereo cameras        
    """    
    def __init__(self):
        super(calibration, self).__init__(' ')
        
    def intrinsic_calibration(self, intrinsic_images_path = ' ', save = False, 
                              base_save_path = ' '):
        """
        Intrinsic calibration of a camera. User must point to location of 
        images containing a series of checkerboard rotations
        
        Args:
            [intrinsic_images_path(str)]: Path which points to location of 
                calibration images (should be in form /img1time and /img2time 
                see calibration_images_save.py)
            [save(bool)]: Whether or not to save intrinsic calibration
            [base_save_path(str)]: Location to save intrinsic values
        
        Returns:
            None
            
        """
        #Generate 3D world points (arbitrary)
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
        #Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                    30, 0.001)
        #Initalize image window
        cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img1', 800,800)
        cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('img2', 800,800)
        #Checkerboard points
        imgpoints1 = [] # 2d points in image plane.
        imgpoints2 = []
        i = 0
        #Load images
        imgs1 = sorted(glob.glob(intrinsic_images_path + '/img1*.png'))
        save_path1 = base_save_path + '/camera1/'
        imgs2 = sorted(glob.glob(intrinsic_images_path + '/img2*.png'))
        save_path2 = base_save_path + '/camera1/'
        
        images = zip(imgs1, imgs2)
        
        objpoints = [] # 3d point in real world space
        #Loop through all images to detect checkerboard corners
        for fname1, fname2 in images:
            i += 1
            #Load images
            img1 = cv2.imread(fname1)
            img2 = cv2.imread(fname2)
            cv2.imshow('img1', img1)
            cv2.imshow('img2', img2)
            if i == 1:
                print("Click ENTER if both checkerboards are" /
                      "visable to save checkerboards for calibration")
                #First loop, wait for a couple seconds
                cv2.waitKey(5000)
            k = cv2.waitKey(1000)
            #Pause for a second. If enter is pressed, use frame for calibtation
            if k == 10:   
                #Convert to grayscale for corner detection
                gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
                # Find the chess board corners
                ret1, corners_cam1 = cv2.findChessboardCorners(
                        gray1, (8,6), None)
                ret2, corners_cam2 = cv2.findChessboardCorners(
                        gray2, (8,6), None)
                #If checkerboards are seen in both frames (propably uneccessary...)
                #TODO: Remove this condition for more points or nah?
                if ret1 and ret2:
                    objpoints.append(objp)
                    #Find corners
                    corners_cam1_2 = cv2.cornerSubPix(
                            gray1,corners_cam1,(11,11),(-1,-1),criteria)       
                    corners_cam2_2 = cv2.cornerSubPix(
                            gray1,corners_cam2,(11,11),(-1,-1),criteria) 
                    #Append points
                    imgpoints1.append(corners_cam1_2)
                    imgpoints2.append(corners_cam2_2)
                    # Draw and display the corners
                    img1 = cv2.drawChessboardCorners(img1, (8,6), 
                                                     corners_cam1_2,ret1)
                    img2 = cv2.drawChessboardCorners(img2, (8,6), 
                                                     corners_cam2_2,ret2)
                    #Show images with checkerboards
                    cv2.imshow('img1',img1)
                    cv2.imshow('img2',img2)
                    cv2.waitKey(5) 
        #Find intrinsics and save                      
        ret1, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
                        objpoints, imgpoints1, gray1.shape[::-1], None, None)
                                            
        
        if save:
            np.savetxt(save_path1 + 'intrinsic_matrix.csv', mtx1, 
                       fmt='%1.3f', delimiter=",")
            np.savetxt(save_path1 + 'distortion_coeffs.csv', dist1, 
                       fmt='%1.3f', delimiter=",")  
        
        print('Intrinsic Matrix Camera 1', mtx1)
        print('Distortion Coefficients Camera 1', dist1)
        
        ret2, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, 
                                            imgpoints2, gray2.shape[::-1], 
                                            None, None)
        if save:
            np.savetxt(save_path2 + 'intrinsic_matrix.csv', mtx2, 
                       fmt='%1.3f', delimiter=",")
            np.savetxt(save_path2 + 'distortion_coeffs.csv', dist2, 
                       fmt='%1.3f', delimiter=",")   
        
        print('Intrinsic Matrix Camera 2', mtx1)
        print('Distortion Coefficients Camera 2', dist1)
        
        
    def find_essential(self, fname1, fname2, save = False, 
                         load_intrinsics = True, intrinsics_load_path = ' ', 
                         stereo_save_path = ' ', mtx1 = None, dist1 = None, 
                         mtx2 = None, dist2 = None):
        if load_intrinsics:
            #Try to load intrnsic values if specified, otherwise raise error
            try:
                mtx1, dist1 = self._load_intrinsics(
                        intrinsics_load_path + '/camera1')
                mtx2, dist2 = self._load_intrinsics(
                        intrinsics_load_path + '/camera2')
            except:
                raise AttributeError('No Intrinsics found')
        #Check that the matricies are of proper form
        else:
            if not type(mtx1) != type(np.array()):
                raise ValueError("R is not of type np.array")
            if not type(dist1) != type(np.array()):
                raise ValueError("R is not of type np.array")
            if not type(mtx2) != type(np.array()):
                raise ValueError("R is not of type np.array")
            if not type(dist2) != type(np.array()):
                raise ValueError("R is not of type np.array")  

        img_name = self._image_mouse_click_directory(fname1, fname2)
        #Return clicked points
        pnts1, pnts2 = self.get_points()
        img = cv2.imread(img_name)
        imshape = (img.shape[0], img.shape[1])    
        
        #print(pnts1)
        E = cv2.findEssentialMat(pnts1, pnts2)#, focal = mtx1[0,0], pp = (mtx1[0,2], mtx1[1,2]))
        R1 = np.zeros((3,3))
        R2 = np.zeros((3,3))
        t = np.zeros((3,1))
        
        cv2.decomposeEssentialMat(E[0], R1, R2, t)   
        print(R1, R2)
        
        if save:
            np.savetxt(stereo_save_path + '/rotation_matrix.csv', 
                       R2, fmt='%1.3f', delimiter=",")
            np.savetxt(stereo_save_path + '/translation_matrix.csv', 
                       t, fmt='%1.3f', delimiter=",")
            np.savetxt(stereo_save_path + '/essential_matrix.csv', 
                       E[0], fmt='%1.3f', delimiter=",")
        
        
    def stereo_calibrate(self, fname1, fname2, save = False, 
                         load_intrinsics = True, intrinsics_load_path = ' ', 
                         stereo_save_path = ' ', mtx1 = None, dist1 = None, 
                         mtx2 = None, dist2 = None):
        """
        Stereo calibration of two cameras. User must either load matricies from
        source (currently must be .csv file) or include intrinics for both 
        cameras
        
        Args:
            fname1(str): Calibration image path1
            fname2(str): Calibration image path2
            [save(bool)]: To save extrinsic calibration or not
            [load_intrinsics(bool)]: Load intrinics. If false, user must
                supply intrnic values
            [intrinsics_load_path(str)]: Path to intrinsic values.
            [stereo_save_path(bool)]: Location to save extrinsic values
            [mtx1 and mtx2 (np.array)]: Intrnisic matricies for both cameras
            [dist1 and dist2 (np.array)]: Distortion coefficients 
        
        Returns:
            None
            
        """        
        if load_intrinsics:
            #Try to load intrnsic values if specified, otherwise raise error
            try:
                mtx1, dist1 = self._load_intrinsics(
                        intrinsics_load_path + '/camera1')
                mtx2, dist2 = self._load_intrinsics(
                        intrinsics_load_path + '/camera2')
            except:
                raise AttributeError('No Intrinsics found')
        #Check that the matricies are of proper form
        else:
            if not type(mtx1) != type(np.array()):
                raise ValueError("R is not of type np.array")
            if not type(dist1) != type(np.array()):
                raise ValueError("R is not of type np.array")
            if not type(mtx2) != type(np.array()):
                raise ValueError("R is not of type np.array")
            if not type(dist2) != type(np.array()):
                raise ValueError("R is not of type np.array")   
        #Get corresponding images
        img_name = self._image_mouse_click_directory(fname1, fname2)
        #Return clicked points
        pnts1, pnts2 = self.get_points()
        img = cv2.imread(img_name)
        imshape = (img.shape[0], img.shape[1])
        #3d points, arbitrary
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        objpoints = []
        objpoints.append(objp[0:pnts1[0].shape[0]])
        #Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                    30, 0.001)
        #Stereo calibtation
        retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T,\
        E, F = cv2.stereoCalibrate(
                objpoints, pnts1, pnts2, mtx1, dist1, mtx2, dist2, imshape, 
                criteria = criteria, flags=cv2.CALIB_FIX_INTRINSIC)

            
        print('Rotation Matrix', R)
        print('Translation Vector', T)
        print('Essential Matrix', E)
        print('Fundemental Matrix', F)
        print('Transformation (RT)', np.dot(R,T))
    
        if save:
            np.savetxt(stereo_save_path + '/rotation_matrix.csv', 
                       R, fmt='%1.3f', delimiter=",")
            np.savetxt(stereo_save_path + '/translation_matrix.csv', 
                       T, fmt='%1.3f', delimiter=",")
            np.savetxt(stereo_save_path + '/essential_matrix.csv', 
                       E, fmt='%1.3f', delimiter=",")
            np.savetxt(stereo_save_path + '/fundemental_matrix.csv', 
                       F, fmt='%1.3f', delimiter=",")  
            
    
    
    def get_instrinsics(self, load_path):
            try:
                mtx1, dist1 = self._load_intrinsics(
                        load_path + '/camera1')
                mtx2, dist2 = self._load_intrinsics(
                        load_path + '/camera2')
            except:
                raise AttributeError('No Intrinsics found')
                
            return mtx1, dist1, mtx2, dist2
    
    def get_extrinsics(self, load_path):
        try:
            E,R,T = self._load_extrinsics(
                    load_path)
        except:
            raise AttributeError('No Intrinsics found')
        print(R)
        print(T)
        return E,R,T      
    
    def get_mouse_clicks(self, fname1, fname2):
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image1', 1800,1800)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image2', 1800,1800)
        #Define mouse callback functions
        cv2.setMouseCallback('image1',self._mouse_click1)
        cv2.setMouseCallback('image2',self._mouse_click2)        
        print("Click on the same point in both images")
        print("Press enter to move to next corresponding images")
        print("Press 'f' to finish")
        print("Press cntrl+c to quit")
        signal.signal(signal.SIGINT, self._sigint_handler)
        self.img1 = cv2.imread(fname1)
        self.img2 = cv2.imread(fname2)
        
        cv2.imshow('image1', self.img1)      
        cv2.imshow('image2', self.img2)
        k = cv2.waitKey(0)
        if k == 99:
            cv2.destroyAllWindows()
            sys.exit()  
        if k == 102:
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()   
        
        return self.x1_points, self.y1_points, self.x2_points, self.y2_points

    def _load_intrinsics(self, load_path):
        """
        Load camera intrinsics
        Args:
            load_path(str): Path to load intrinsics from
        Returns:
            mtx(np.array): Intrinsic matrix
            dist(np.array): Distortion matrix
        """
        mtx = np.array(np.loadtxt(load_path + '/intrinsic_matrix.csv', 
                                  dtype = float, delimiter=','))
        dist = np.array(np.loadtxt(load_path + '/distortion_coeffs.csv',
                                   dtype = float, delimiter=','))
        
        mtx.reshape((3,3))
        
        return mtx, dist   

    def _load_extrinsics(self, load_path):
        """
        Load camera intrinsics
        Args:
            load_path(str): Path to load intrinsics from
        Returns:
            mtx(np.array): Intrinsic matrix
            dist(np.array): Distortion matrix
        """
        E = np.array(np.loadtxt(load_path + '/essential_matrix.csv', 
                                  dtype = float, delimiter=','))
        R = np.array(np.loadtxt(load_path + '/rotation_matrix.csv',
                                   dtype = float, delimiter=','))      
        T = np.array(np.loadtxt(load_path + '/translation_matrix.csv',
                                   dtype = float, delimiter=','))          
        
        E.reshape((3,3))
        R.reshape((3,3))
        
        
        return E,R,T         

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
    
    def _image_mouse_click(self, fname1, fname2):
        """
        Determine coressponding image points between the frames
        
        Will display two images. User must click on identical point
        in two frames. x#_points, and y#_points will populate as the user 
        clicks on points  
        
        Args:
            fname1, fname2 (str): Two paths pointing to calibration image frame
        """
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image1', 1200,1200)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image2', 1200,1200)
        #Define mouse callback functions
        cv2.setMouseCallback('image1',self._mouse_click1)
        cv2.setMouseCallback('image2',self._mouse_click2)        
        print("Click on the same point in both images")
        print("Press enter to move to next corresponding images")
        print("Press 'f' to finish")
        print("Press cntrl+c to quit")
        signal.signal(signal.SIGINT, self._sigint_handler)
        self.img1 = cv2.imread(fname1)
        self.img2 = cv2.imread(fname2)
        
        cv2.imshow('image1', self.img1)      
        cv2.imshow('image2', self.img2)
        k = cv2.waitKey(0)
        if k == 99:
            cv2.destroyAllWindows()
            sys.exit()  
        if k == 102:
            cv2.destroyAllWindows()
        cv2.destroyAllWindows()    
        
    def _image_mouse_click_directory(self, fname1_path, fname2_path):
        """
        Determine coressponding image points between the frames
        
        Will display two images. User must click on identical point
        in two frames. x#_points, and y#_points will populate as the user 
        clicks on points  
        
        Args:
            fname1, fname2 (str): Two paths pointing to calibration image frame
        """
        cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image1', 1200,1200)
        cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image2', 1200,1200)
        #Define mouse callback functions
        cv2.setMouseCallback('image1',self._mouse_click1)
        cv2.setMouseCallback('image2',self._mouse_click2)        
        print("Click on the same point in both images")
        print("Press enter to move to next corresponding images")
        print("Press 'f' to finish")
        print("Press q to quit")
        imgs1 = sorted(glob.glob(fname1_path + "/*.jpg"))
        imgs2 = sorted(glob.glob(fname2_path + "/*.jpg"))
        images = zip(imgs1, imgs2)
        for fname1, fname2 in images:
            signal.signal(signal.SIGINT, self._sigint_handler)
            self.img1 = cv2.imread(fname1)
            self.img2 = cv2.imread(fname2)
            
            cv2.imshow('image1', self.img1)      
            cv2.imshow('image2', self.img2)
            k = cv2.waitKey(0)
            
            
            if k == 113:
                cv2.destroyAllWindows()
                sys.exit()    
            if k == 102:
                break
        
        return fname1
                

    
    
        