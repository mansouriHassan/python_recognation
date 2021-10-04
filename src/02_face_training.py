import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
projectPath = os.getcwd().replace("src","").replace("\\","/")
datasetPath = projectPath + '/dataset'
trainerPath = projectPath + "/trainer"
recognizer = cv2.face.LBPHFaceRecognizer_create()
cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
detector = cv2.CascadeClassifier(cascPath)

# creating dataset dir if is not exists
if (not os.path.exists(trainerPath)):
    os.mkdir(trainerPath)

# function to get the images and label data
def getImagesAndLabels(datasetPath):

    imagePaths = [os.path.join(datasetPath,f) for f in os.listdir(datasetPath)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    return faceSamples,ids

print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(datasetPath)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))