from fileinput import filename
import streamlit as st
from PIL import Image
from sqlite3 import TimestampFromTicks
from cv2 import FONT_HERSHEY_COMPLEX
import numpy as np
import os
import cv2
import face_recognition
from datetime import datetime

st.title("Face Recognition System")
run=st.checkbox("Run")
FRAME_WINDOW = st.image([])


path = 'images'

images =[]
personName= []
myList= os.listdir(path)



for cu_img in myList:
    #read the current image and extract the name from the image name
    current_img=cv2.imread(f'{path}/{cu_img}')
    images.append(current_img)
    personName.append(os.path.splitext(cu_img)[0])
    

def faceEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


 
def attendance(name):
    with open('attendance.xls', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')



list=faceEncodings(images)
print("All encodings complete!!!")
        

    #use opencv library to load pre trained data

face_data= cv2.CascadeClassifier(r"C:\Users\dell\Desktop\attendance\MAIN\opencv-master\data\haarcascades_cuda\haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0) #captureDevice = camera


while run:

    #read the frame
    frame_read,frame = camera.read()    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces=cv2.resize(frame,(0,0),None,0.25,0.25)
    faces=cv2.cvtColor(faces,cv2.COLOR_BGR2RGB)
    
    
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodeFrame =face_recognition.face_encodings(faces, facesCurrentFrame)
        
    for encodeFace , faceloc in zip(encodeFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(list,encodeFace)
        faceDistance= face_recognition.face_distance(list,encodeFace)
        
        #to find the match we use argmin 
        matchIndex = np.argmin(faceDistance)
        
        if matches[matchIndex]:
            name= personName[matchIndex].upper()
            #to draw rectangle and resizing 
            y1,x2,y2,x1 = faceloc
            y1,x2,y2,x1 = y1 * 4,x2 * 4,y2 * 4,x1 * 4
            
            #making frames and adding text/Name
            cv2.rectangle(frame , (x1,y1) , (x2 , y2) , (0,255,0) , 5)
            cv2.rectangle (frame,(x1,y2-35) , (x2,y2) , (0,255,0) , cv2.FILLED)
            cv2.putText(frame,name , (x1+6,y2-6) , cv2.FONT_HERSHEY_COMPLEX, 1 ,(255,255,255) , 3)
            attendance(name)

    FRAME_WINDOW.image(frame)
        # cv2.imshow('Face Detector' , frame)
        # key = cv2.waitKey(1)
        # if key==32 or key==27:
        #     break   
        


    # if key== ord('c'):
    #     filename = 'faces/'+name+'.jpg'
    #     cv2.imwrite(filename,frame)
        
    # print ("Image saved - ",filename)






# print("code completed")
    
else:
    st.write('Stopped')
    #convert to grayscale
        #img_black= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #detecting face coordinates
        #face_coordinates = face_data.detectMultiScale(img_black)
        
        #going back to the colored image and drawing rectangles around the detected faces
        # for (x,y,w,h) in face_coordinates:
        #     
        # def detect_faces(our_image):
        