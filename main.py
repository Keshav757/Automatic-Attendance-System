from datetime import datetime
import os
import pickle
import cv2
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : "https://faceattendancerealtime-405f1-default-rtdb.firebaseio.com/",
    'storageBucket' : "faceattendancerealtime-405f1.appspot.com"
})
bucket=storage.bucket()

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
imgBackground=cv2.imread("Resources/background.png")
folderMode="Resources/Modes"
modepathlist=os.listdir(folderMode)
imgmode=[]
imgStudent=[]
#print(modepathlist)
for path in modepathlist:
    imgmode.append(cv2.imread(os.path.join(folderMode,path)))
#print(imgmode)
#import encoding
file = open('Encoderfile.p','rb')
encodeListKnownWithIds=pickle.load(file)
file.close()
encodeListKnown,studentIds=encodeListKnownWithIds
print(studentIds)
print('encoder file loaded')
modeType=0
counter=0
id=0

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    faceCurrFrame=face_recognition.face_locations(imgS)
    encodeCurrFrame=face_recognition.face_encodings(imgS,faceCurrFrame)

    imgBackground[162:162+480,55:55+640]=img
    imgBackground[44:44+633,808:808+414]=imgmode[modeType]
    if faceCurrFrame:
        for encodeFace,faceLoc in zip(encodeCurrFrame,faceCurrFrame):
            matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDistance=face_recognition.face_distance(encodeListKnown,encodeFace)
            #print("matches",matches)
            #print("face Distance",faceDistance)

            matchIdx=np.argmin(faceDistance)
            #print('Match index: ',matchIdx)

            if matches[matchIdx]:
                # print("Known face detected")
                # print(studentIds[matchIdx])
                y1, x2, y2, x1= faceLoc
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
                bbox= 55+x1,162+y1,x2-x1,y2-y1
                imgBackground=cvzone.cornerRect(imgBackground,bbox,rt=0)
                id=studentIds[matchIdx]
                if counter==0:
                    cvzone.putTextRect(imgBackground,"Loading.....",(274,400))
                    cv2.imshow("Face Attendance",imgBackground)
                    cv2.waitKey(1)
                    counter=1
                    modeType=1

        if counter!=0:
            if counter==1:
                studentInfo = db.reference(f'Students/{id}').get()
                print(studentInfo)
                #get the Image from the storage
                blob = bucket.get_blob(f'Images/{id}.png')
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                imgStudent = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                dt=datetime.strptime(studentInfo['last_attendance_time'],"%Y-%m-%d %H:%M:%S")
                timelapsed=(datetime.now()-dt).total_seconds()
                print(timelapsed)
                if timelapsed> 30:
                    #update data
                    ref=db.reference(f'Students/{id}')
                    studentInfo['total_attendance']+=1
                    ref.child('total_attendance').set(studentInfo['total_attendance'])
                    ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    modeType=3
                    counter=0
                    imgBackground[44:44+633,808:808+414]=imgmode[modeType]

            if modeType!=3:         
                if 10<counter<20:
                    modeType=2
                imgBackground[44:44+633,808:808+414]=imgmode[modeType]

                if counter<=10:
                    cv2.putText(imgBackground,str(studentInfo['total_attendance']),(861,125),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
                    cv2.putText(imgBackground,str(studentInfo['major']),(1006,550),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    cv2.putText(imgBackground,str(id),(1006,493),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
                    cv2.putText(imgBackground,str(studentInfo['standing']),(910,625),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)
                    cv2.putText(imgBackground,str(studentInfo['year']),(1025,625),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)
                    cv2.putText(imgBackground,str(studentInfo['starting_year']),(1125,625),cv2.FONT_HERSHEY_SIMPLEX,0.6,(100,100,100),1)
                    (w,h),_=cv2.getTextSize(studentInfo['name'],cv2.FONT_HERSHEY_COMPLEX,1,1)
                    offset=(414-w)//2
                    cv2.putText(imgBackground,str(studentInfo['name']),(808+offset,445),cv2.FONT_HERSHEY_SIMPLEX,1,(50,50,50),1)
                    imgBackground[175:175+216,909:909+216]=imgStudent
                counter+=1

                if counter>=20:
                    counter=0
                    modeType=1
                    studentInfo=[]
                    imgStudent=[]
                    imgBackground[44:44+633,808:808+414]=imgmode[modeType]
    else:
        modeType=0
        counter=0
    #cv2.imshow("Face attendance",img)
    cv2.imshow("Face Attendance",imgBackground)
    cv2.waitKey(1)