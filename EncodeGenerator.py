import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred,{
    'databaseURL' : "https://faceattendancerealtime-405f1-default-rtdb.firebaseio.com/",
    'storageBucket' : "faceattendancerealtime-405f1.appspot.com"
})

folderMode="Images"
pathlist=os.listdir(folderMode)
imgList=[]
studentIds=[]
print(pathlist)
for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderMode,path)))
    studentIds.append(os.path.splitext(path)[0])

    fileName=f'{folderMode}/{path}'
    bucket=storage.bucket()
    blob=bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(studentIds)

def findEncodings(imglist):
    encodeList= []
    for img in imglist:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding started........ ")
encodeListKnown= findEncodings(imgList)
encodeListKnownWithIds=[encodeListKnown,studentIds]
print(encodeListKnown)
print("Encoding complete")

file=open("Encoderfile.p",'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File saved")