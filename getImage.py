#os, PIL 모듈을 이용한 filelist 불러오기 시도
#os, PIL 모듈을 이용해 불러온 filelist를 cv2로 출력하기
#Cascade를 사용해서 얼굴만 뽑아오기
#뽑아낸 얼굴만 잘라내서 resize하기
#resize한 파일 출력하기 cv2를 사용
#Version: 0.5.0
import cv2
import os
import numpy as np
from PIL import Image

def getImageFile(path):

    file_list = os.listdir(path)
    file_list.sort()

    for file in file_list:
        cv_file = path + file
        print(cv_file)
        img = cv2.imread(cv_file)

        processImage = faceDetect(img)
        saveFileName = "SAVEPATH" + file
        print(saveFileName)
        cv2.imwrite(saveFileName, processImage)

def faceDetect(frame):
    
    faceCascPath = "HAARCASCADE_PATH"
    faceCascade = cv2.CascadeClassifier(faceCascPath)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (50,50),
        flags = cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        frame = gray[y:y+h, x:x+w]
        frame = cv2.resize(frame, (64,64))

    return frame

if __name__ == "__main__":
    
    train_load_path ="IMAGEPATH"
    getImageFile(train_load_path)