from tkinter import Y, Frame
import cv2

#需要VScode need VScode
#需要先安装OpenCV  need to install OpenCV first  "pip install opencv-python"
#载入事先训练好的OpenCV数据 正前方面部头像
#Load the pre-trained OpenCV data as cvData, front face
#https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
cvdata = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#输入图像 视频或者相机  
#Enter an image, video or camera

webcam = cv2.VideoCapture(0) #载入相机 Load the camera 
#img = cv2.imread('123.jpg') #载入图片源  Load the picture 
#webcam = cv2.VideoCapture('11111.mp4')  载入视频 Load the video

#读取相机帧数 Read the camera frames
while True:
    successfull_frame_read, Frame = webcam.read()    #读取"Frame"当前的图像帧 Reads the current frame

#successfull_frame_read可以换成read或者其他 但是不能是数字
#"successFULL_frame_read" can be replaced with "read" or something else but cannot be a number


#将图片黑白处理 Process the picture into monochrome 
    mono_img = cv2.cvtColor(Frame,cv2.COLOR_BGR2GRAY)

    #侦测脸部,让坐标在帧图中显示 detect the coordinates of the face,Display in an image frame
    face_coordinates = cvdata.detectMultiScale(mono_img)

    #使用上一步得到的面部坐标"face_coordinates",通过OpenCV的功能在召唤框框,for loop用于侦测多张脸
    #get the face coordinates from previous step"face_coordinates", 
    #use OpenCV function to call the box, for loop for multiface 
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(Frame,(x,y),(x+w, y+h),(0,255,0), 4)

    #这是OpenCV 用于窗口展示的功能  This is what OpenCV does for window display
    cv2.imshow('Window A1',Frame)  


    #设置窗口关闭键 Set the window close key
    key = cv2.waitKey(1)  
    if key==81 or key==113:
        break
    
webcam.release()
    


 

