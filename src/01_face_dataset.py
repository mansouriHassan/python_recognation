import cv2
import os

projectPath = os.getcwd().replace("src","").replace("\\","/")
datasetPath = projectPath + "/dataset"
imagePath = projectPath + "/images/"
imageName = "10.jpg"

cascPath=os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(cascPath)

# For each person, enter one numeric face id
face_id = 10
#input('\n enter user id end press <return> ==>  ')

print("\n [INFO] Initializing face capture. wait ...")

# creating dataset dir if is not exists
if (not os.path.exists(datasetPath)):
    os.mkdir(datasetPath)

# Initialize individual sampling face count
count = 0
while(True):

    img = cv2.imread(imagePath + imageName)
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
#cam.release()
cv2.destroyAllWindows()