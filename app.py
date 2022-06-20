import numpy as np
import pandas as pd
from flask import Flask, request, render_template,session
import os
import cv2 as cv
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)

@app.route("/",methods=['GET'])
def home_page():
    return render_template('index.html')

@app.route("/predict", methods=['GET','POST'])
def predict():

    def biggestContour(contours):
        biggest = np.array([])
        max_area = 0
        for i in contours:
            area = cv.contourArea(i)
            peri = cv.arcLength(i, True)
            if(area>0):
                approx = cv.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        return biggest, max_area
    
    def reorder(myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew
    
    def drawRectangle(img, biggest, thickness):
        cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        cv.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
        cv.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
        return img
    
    def optimization(img):
        widthImg=480
        heightImg=480
        img=cv.resize(img,(widthImg,heightImg)) #image resizing
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #converting image to grey scale image
        blur=cv.GaussianBlur(gray,(5,5),1) #blurring the image
        edges_detection=cv.Canny(blur,100,200) #Edge detection using Canny Edge Detector
        imgContours=img.copy()
        imgBigContour=img.copy()
        contours,hierarchy = cv.findContours(edges_detection, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(imgContours,contours,-1,(0,255,0),5)
        #print("Number of Contours found = " + str(len(contours)))
        biggest,maxArea=biggestContour(contours)
        #print(biggest)
        biggest=reorder(biggest)
        cv.drawContours(imgBigContour,biggest,-1,(0,255,0),20)
        imgBigContour=drawRectangle(imgBigContour,biggest,2)
        pts1=np.float32(biggest)
        pts2=np.float32([[0,0],[widthImg,0],[0,heightImg],[widthImg,heightImg]])
        matrix=cv.getPerspectiveTransform(pts1,pts2)
        imgWarpColored=cv.warpPerspective(img,matrix,(widthImg,heightImg))

        

        cv.imwrite('b.jpeg',img)
        
        #cv.imwrite('FINALIMAGE.jpeg',imgWarpColored)
        
        return imgWarpColored




        
    path=r"C:\Users\MRUDUN\Image Document Optimization\a.jpeg"
    path1=r"C:\Users\MRUDUN\Image Document Optimization\finale.jpeg"
    if request.method == 'POST':

        image=request.files.get('image', '')
        #image=request.form["image"]
        image.save('a.jpeg')
        ima=cv.imwrite('finale.jpeg',optimization(cv.imread(path)))
        im=Image.open(path1)
        p=im.save("image2.jpg",quality=95)
        cv.imwrite('image2.jpg',p)
        return render_template('result.html',op1=path1)

if __name__=='__main__':
    app.run(debug = True,host='0.0.0.0',port=80)
        
        