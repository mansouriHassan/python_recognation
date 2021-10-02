import cv2
import os

# Get user supplied values
imagePath = "C:\\Users\\MANSOURO\\Desktop\\project\\images\\1.jpg"

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

'''
cascPath = "C:\\Users\\MANSOURO\\Desktop\\project\\haarcascades\\haarcascade_frontalface_default.xml"
imagePath = sys.argv[1]
cascPath = sys.argv[2]
'''

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

#print "Found {0} faces!".format(len(faces))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    
cv2.imshow("Faces found", image)
cv2.waitKey(0)

#python face_detect.py abba.png haarcascade_frontalface_default.xml