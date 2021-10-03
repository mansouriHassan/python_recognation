import os
from PIL import Image
import glob
import face_recognation as recognation
import utility

projectPath = os.getcwd()
imagePath = projectPath + "/images/"
imageName = "1.jpg"
filePath = projectPath + "/files/"
fileName = "1.json"

recognation.face_detector(imagePath + imageName)
#recognation.video_detector()
#utility.parsing_file(fileName)