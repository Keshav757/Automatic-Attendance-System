## Automatic Face Recognition Attendance System
# Introduction
This project is an Automatic Face Recognition Attendance System built using Python, OpenCV, and Firebase. The system captures real-time video from a webcam, detects faces, and matches them against a database of known faces to record attendance. The attendance data is stored in Firebase Realtime Database and Firebase Storage.
# Features
Real-time face detection and recognition
Automatic attendance logging
Storage of attendance data in Firebase Realtime Database
Storage of student images in Firebase Storage

# Step 1: Encode Student Images
Place student images in the Images folder. Each image file name should be the student ID (e.g., 12345.jpg). Run the encoding script to generate face encodings and upload images to Firebase Storage.

# Step 2: Run the Attendance System

The system will start capturing video from the webcam. When a face is detected, it will be matched against the known faces, and attendance will be logged in Firebase.
