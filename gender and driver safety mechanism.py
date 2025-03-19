import cv2
import math
import numpy as np
import tensorflow as tf
import winsound
from deepface import DeepFace

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def beep_alert():
    duration = 1000  # milliseconds
    freq = 1000  # Hz
    winsound.Beep(freq, duration)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frameOpencvDnn, faceBoxes

def monitor_driver():
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = [f'({i}-{i+5})' for i in range(0, 100, 6)]
    genderList = ['Male', 'Female']
    
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    cap = cv2.VideoCapture(0)

    # Set resolution to 360p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            cv2.putText(frame, "FACE NOT FOUND!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            beep_alert()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        
        try:
            result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
            padding = 20
            for i, faceBox in enumerate(faceBoxes):
                x1, y1, x2, y2 = faceBox
                
                face = frame[max(0, y1 - padding): min(y2 + padding, frame.shape[0] - 1),
                             max(0, x1 - padding): min(x2 + padding, frame.shape[1] - 1)]
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                cv2.putText(resultImg, f'{gender}, {age}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                if i < len(result) and 'dominant_emotion' in result[i]:
                    dominant_emotion = result[i]['dominant_emotion']
                    confidence = result[i].get('emotion', {}).get(dominant_emotion, 0)
                    
                    cv2.putText(resultImg, f"{dominant_emotion} ({confidence:.1f}%)", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    drowsy_emotions = ['tired', 'sad', 'angry', 'fear', 'disgust']
                    if dominant_emotion in drowsy_emotions and confidence > 50:
                        cv2.putText(resultImg, "DROWSINESS ALERT!", (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        beep_alert()
            
        except Exception as e:
            print("Face not detected, skipping frame.")
        
        cv2.imshow("Driver Safety Monitoring (360p)", resultImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_driver()
