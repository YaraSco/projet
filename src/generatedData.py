# -*- coding: utf-8 -*-
##################################
#  
#  
##################################

import pandas as pd 
import numpy as np
import random 
import cv2
import math
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA


class GeneratedData:

    TRAIN_PATH = "train.csv"
    IMAGE_PATH = "images/"    
    R = random.randint(17, 27) # for random sampling in cross-validation

    _img_train = pd.DataFrame()
    _img_test = pd.DataFrame()
    _X_all_train = pd.DataFrame()
    _X_all_test = pd.DataFrame()
    _X_train = pd.DataFrame()
    _X_test = pd.DataFrame()
    _y_train = pd.DataFrame()
    _y_test = pd.DataFrame()
    _classes = []
    _id_img_train = []
    _id_img_test = []


    def __init__(self, nb_test=0.2, pca=False):
        """
        argument pca=True reduces dimensionality of data
        """
        self.nb_test = nb_test
        self.pca = pca

 # Definition of private methods          
    def generateData(self):
        """
        This method generates data to facilitate its use
        """
        train = pd.read_csv(self.TRAIN_PATH)
        
        s = LabelEncoder().fit(train.species)  
        self._classes = list(s.classes_)  
        class_labels = s.transform(train.species)
        train = train.drop(['species'], axis=1)

        if self.pca:
            trainX = train.drop(['id'], axis=1)
            pca = PCA(n_components=0.85 ,svd_solver='full')
            pca.fit(trainX)
            trainX = pca.transform(trainX)
            train_df = pd.DataFrame.from_records(trainX)
            train_df.insert(loc=0, column='id', value=train['id'])
            train = train_df

        sss = StratifiedShuffleSplit(n_splits=1, test_size=self.nb_test, random_state=self.R)
        for id_train, id_test in sss.split(train, class_labels):  
            X_train, X_test = train.values[id_train], train.values[id_test]  
            self._y_train, self._y_test = class_labels[id_train], class_labels[id_test]

        self._id_img_train =  list(np.int_( X_train[:,0]))
        self._id_img_test =  list(np.int_( X_test[:,0])) 
        self._X_train = np.delete(X_train, 0, 1)
        self._X_test = np.delete(X_test, 0, 1)

    def pixel_id_top (self,matrix):  
        """
        This method gets index of the first white pixel from top to bottom of the image
        matrix: matrix of the image
        return: id_row :row number
                id_col :column number
        """
        i=0   
        while max(matrix[i,:])!=float(1):
            i=i+1

        id_row=i

        j=0  
        while matrix[i,j]!=float(1):
            j=j+1

        id_col=j

        return id_row, id_col

    def pixel_id_bottom (self,matrix):
        """
        This method gets index of the first white pixel from bottom to top of the image
        matrix: matrix of the image
        return: id_row :row number
                id_col :column number
        """
        i=len(matrix[:,0])-1   
        while max(matrix[i,:])!=float(1):
            i=i-1

        id_row=i

        j=0   
        while matrix[i,j]!=float(1):
            j=j+1

        id_col=j

        return id_row, id_col

    def pixel_id_left (self,matrix):
        """
        This function gets index of the first white pixel from left to right of the image
        matrix: matrix of the image
        return: id_row :row number
                id_col :column number
        """
        j=0
        while max(matrix[:,j])!=float(1):
            j=j+1

        id_col=j

        i=0
        while matrix[i,j]!=float(1):
            i=i+1

        id_row=i

        return id_row, id_col

    def pixel_id_right (self,matrix):
        """
        This method gets index of the first white pixel from right to left of the image
        matrix: matrix of the image
        return: id_row :row number
                id_col :column number
        """
        j=len(matrix[0,:])-1   
        while max(matrix[:,j])!=float(1):
            j=j-1

        id_col=j

        i=0   
        while matrix[i,j]!=float(1):
            i=i+1

        id_row=i

        return id_row, id_col


    def removeBlackFrame(self,image):
        """
        This method removes black frame surrounding the leaf in the image
        image: image object
        return: result :image object
        """        
        image_array = np.asarray(image) 

        left_r, left_c = self.pixel_id_left (image_array)
        right_r, right_c = self.pixel_id_right (image_array)
        top_r, top_c = self.pixel_id_top (image_array)
        bottom_r, bottom_c = self.pixel_id_bottom (image_array)

        image_array = image_array[top_r:bottom_r+1, left_c:right_c+1]
        result = Image.fromarray(image_array) 

        return result
   
    def pixel_BW(self,image):
        """
        This method computes the percentage of withe and black pixels from the image
        image: image object
        return: p0: percentage of black pixels
                p1:percentage of white pixels
        """        
        h = image.histogram()
        nt, n0, n1 = sum(h), h[0], h[-1]
        p0 = round(100*n0/nt,2)  
        p1 = round(100*n1/nt,2)  
        return p0, p1
  
    def ratio_WL(self,image):
        """
        This function computes the ratio of width to length
        image: image object
        return: width/length: ratio
        """                
        width, length =image.size
        return width/length
  
    def contour_features(self,imagefile):   
        """
        This method computes different image features based on contour-detection 
        imagefile: image path
        return: peak: number of peaks of the contour
                eccentricity: eccentricity of the ellipse that fits the contour
                angle: deviation of the ellipse that fits the contour
                m: gradient of the line that fits the contour
                y0: image of the abscissa 0 by the equation of the line that fits the contour
        """
        original_color = cv2.imread(imagefile)
        original = cv2.cvtColor(original_color, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(original,(5,5),0)
        ret, thresh = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh,100,200)
        contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        LENGTH = len(contours)            
        big_cnt = np.vstack([contours[c] for c in range (0,LENGTH)])
        perimetre = cv2.arcLength(big_cnt,True)
        approx = cv2.approxPolyDP(big_cnt,0.01*perimetre,True)

        peak = len(approx)

        (x, y), (MA, ma), angle = cv2.fitEllipse(big_cnt)

        a = ma / 2
        b = MA / 2
        eccentricity = math.sqrt(pow(a, 2) - pow(b, 2))
        eccentricity = round(eccentricity / a, 2)

        [vx,vy,x,y] = cv2.fitLine(big_cnt, cv2.DIST_L2,0,0.01,0.01) #vx,vy are normalized vector collinear to the line and x0,y0 is a point on the line
        m=vy[0]/vx[0]
        y0=y[0]-m*x[0]

        return peak, eccentricity, angle, m, y0

    def getImagesCaracteristics(self,N):
        """
        This method computes all the features of an image and render a progress bar for the data
        N: list that contains indexes of the images
        return: a dataframe of the features
        """   
        image_data= [[0] * 8 for _ in range(len(N))]

        for i in tqdm(range(0,len(N))):
            imagefile=self.IMAGE_PATH+str(N[i])+".jpg"
            image = Image.open(imagefile)
            image = image.convert('1')
            image = self.removeBlackFrame(image)
            image_data[i][0], image_data[i][1] = self.pixel_BW(image) 
             
            image_data[i][2] = self.ratio_WL(image)

            peak, eccentricity, angle ,m ,y0 = self.contour_features(imagefile)
            
            image_data[i][3] = peak
            image_data[i][4] = eccentricity
            image_data[i][5] = angle
            image_data[i][6] = m
            image_data[i][7] = y0
        
        return pd.DataFrame(data=image_data, columns=['black_pxl%', 'white_pxl%', 'ratio_W/L','nb_peak','ellipse_eccentricity','ellipse_deviation','line_gradient','line_y0'])

    def getImageData(self):
        """
        This method gives the image's data
        """ 
        if len(self._id_img_train) == 0 or len(self._id_img_test) == 0:
            self.generateData()

        if len(self._img_train) == 0:
            self._img_train=self.getImagesCaracteristics(self._id_img_train).to_numpy()

        if len(self._img_test) == 0:
            self._img_test=self.getImagesCaracteristics(self._id_img_test).to_numpy()

# Definition of public methods

    def generated_data_train(self):
        """
        This function calls the private function _extractBasicData() to extract train data
        :return: _X_data_train: Train matrix
        """
        if len(self._X_train)==0:
            self.generateData()

        return self._X_train   
        
    def generated_data_test(self):          
        """
        This function calls the private function _extractBasicData() to extract test data
        :return: _X_data_test : Test matrix
        """
        if len(self._X_test)==0:
            self.generateData()

        return  self._X_test

    def generated_target_train(self):        
        """
        This function calls the private function _extractTrainTargets() to extract train Targets if they aren't already extracted
        :return: A vector of data classes
        """
        if len(self._y_train)==0:
            self.generateData()

        return self._y_train

    def generated_target_test(self):
        """
        This function calls the private function _extractTestTargets() to extract test Targets if they aren't already extracted
        :return: A vector of data classes
        """  
        if len(self._y_test)==0:
            self.generateData()

        return self._y_test

    def list_classes(self):
        """
        This function  lists all the classes
        :return: vector of all classes
        """ 
        if len(self._classes)==0:
            self.generateData()

        return self._classes

    def image_train_data(self):          
        """
        This function extract image features for train data
        :return: _X_img_train : Train image features matrix
        """
        if len(self._img_train)==0 :
            self.getImageData()

        return self._img_train
 
    def image_test_data(self):          
        """
        This function extract image features for test data
        :return: _X_img_test  : Test image features matrix
        """
        if len(self._img_test)==0:
            self.getImageData()

        return self._img_test

    def features_data_train(self):          
        """
        This function concatenates the data with the features for training
        :return: _X_train: Train matrix
        """
        if len(self._X_all_train)==0:
            if len(self._img_train)==0:
                self.getImageData()

            self._X_all_train=np.concatenate((self._X_train, self._img_train), axis=1)

        return self._X_all_train
    
    def features_data_test(self):          
        """
        This function concatenates the data wth the features for testing
        :return: _X_test: Test matrix
        """
        if len(self._X_all_test)==0:
            if len(self._img_test)==0:
                self.getImageData()

            self._X_all_test=np.concatenate((self._X_test, self._img_test), axis=1)

        return self._X_all_test
