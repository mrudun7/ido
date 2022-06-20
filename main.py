import os
import numpy as np
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template,session
from werkzeug.utils import secure_filename
import os
import cv2 as cv
from werkzeug.utils import secure_filename
from PIL import Image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
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
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		print('uploads/' + os.path.join(app.config['UPLOAD_FOLDER'], filename))
		flash('Image successfully uploaded and displayed below')
		img=cv.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		imgWarpColored=optimization(img)
		cv.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'Done'+filename),imgWarpColored)
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/Done' +filename), code=301)




if __name__ == "__main__":
    app.run()