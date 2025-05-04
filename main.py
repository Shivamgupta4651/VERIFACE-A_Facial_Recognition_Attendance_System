
import cv2
import os
import pickle
import numpy as np
import face_recognition
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import numpy as np

cred = credentials.Certificate("/Lenovo\Desktop\VERIFACE\Veriface123/.venv/veriface-9c99e-firebase-adminsdk-lkefp-a54bf03d00.json")
firebase_admin.initialize_app(cred,{
    "databaseURL": "https://veriface-9c99e-default-rtdb.firebaseio.com/",
    "storageBucket": "veriface-9c99e.appspot.com"
})

bucket = storage.bucket()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imageBackground = cv2.imread("/Users/Lenovo\Desktop\VERIFACE\Veriface123/.venv/Resources/WhatsApp Image 2024-04-18 at 23.26.02.jpeg")
# importing the mode images into a list
folderModePath = "/Users/rLenovo\Desktop\VERIFACE\Veriface123/.venv/Resources/Modes"
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))


# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded ...")

modeType = 0
counter = 0
id = -1
imgStudent = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img,(0,0), None, 0.25 , 0.25)
    imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

    imageBackground[162:162+480,55:55+640] = img
    imageBackground[44:44+633,808:808+414] = imgModeList[modeType]
    if faceCurFrame:
        for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        # print("matches", matches)
        # print("faceDis", faceDis)

            matchIndex = np.argmin(faceDis)
        # print("Match Index", matchIndex)

            if matches[matchIndex]:
            # print("known Face Detected")
            # print(studentIds[matchIndex])
                y1,x2,y2, x1, = faceLoc
                y1, x2, y2, x1, = y1 * 4, x2 * 4, y2 * 4, x1 * 4,
                bbox = 55+x1, 162+y1, x2-x1,y2-y1
                imageBackground = cvzone.cornerRect(imageBackground,bbox,rt=0)
                id = studentIds[matchIndex]
                # print(id)
                if counter == 0:
                    counter = 1
                    modeType = 1
        if counter!= 0:

            if counter ==1:
                # Get the Data
                studentsInfo = db.reference(f'Student/{id}').get()
                print(studentsInfo)
                # Get the Image From the storage
                blob = bucket.get_blob( 'C:\Users\Lenovo\Desktop\VERIFACE\Veriface123>/Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)



                imgStudent = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
                # Update data of attendance
                ref = db.reference(f'Student/{id}')
                studentsInfo['total_attendance'] +=1
                ref.child('total_attendance').set(studentsInfo['total_attendance'])

        if 10< counter <20:
               modeType = 2

        imageBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        if counter<=10:
            cv2.putText(imageBackground,str(studentsInfo['total_attendance']), (861,125),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)



            cv2.putText(imageBackground, str(studentsInfo['major']), (1006, 550),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(imageBackground, str(id), (1006, 493),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            cv2.putText(imageBackground, str(studentsInfo['standing']), (910, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

            cv2.putText(imageBackground, str(studentsInfo['year']), (1025, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

            cv2.putText(imageBackground, str(studentsInfo['starting_year']), (1125, 625),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

            (w, h), _ = cv2.getTextSize(studentsInfo['name'], cv2.FONT_HERSHEY_COMPLEX,1,1)
            offset = (414-w)//2
            cv2.putText(imageBackground, str(studentsInfo['name']), (808, 445),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

            imageBackground[175:175+216, 909:909+216] = imgStudent

            counter+=1

            if counter>=20:
                counter=0
                modeType=0
                studentsInfo=[]
                imgStudent = []
                imageBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

        cv2.imshow('Video', img)
        if imageBackground is not None and imageBackground.size > 0:
             cv2.imshow('Background', imageBackground)
        else:
             print("Error: Background image could not be loaded or is empty.")
        cv2.waitKey(1)
