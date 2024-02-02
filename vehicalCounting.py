import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import*

model = YOLO("yolov8s.pt")

# read the classes from coco.txt file
myFile = open("coco.txt",'r')
data = myFile.read()
classList = data.split("\n")
# print(classList)

def VC(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('VehicalCounting')
cv2.setMouseCallback('VehicalCounting', VC)
cam = cv2.VideoCapture("trafficVid.mp4")

cy1 = 375
cy2 = 430

offset = 6

count = 0
carDown = {}
tracker = Tracker()
counter1 = []

carUp = {}
counter2 = []

while True:
    success, frame = cam.read()
    if not success:
        break

    # skipping every third frame
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1300,700))

    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    # print(px)

    list = []

    for index, row in px.iterrows():
        
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = classList[d]
        if 'car' or 'truck' in c:
            list.append([x1,y1,x2,y2])

    bboxId = tracker.update(list)
    for bbox in bboxId:
        x3,y3,x4,y4,id = bbox

        cx = int(x3+x4)//2
        cy = int(y3+y4)//2

        cv2.circle(frame, (cx,cy), 4, (0,255,255), cv2.FILLED)
        if cy1<(cy+offset) and cy1>(cy-offset):
            cv2.rectangle(frame, (x3,y3),(x4,y4), (0,255,0),2,cv2.FILLED)
            cv2.putText(frame,f"id:{id}",(x3-5,y3),cv2.FONT_HERSHEY_PLAIN,1,(255,70,255),2)
            # cv2.putText(emptyFrame,f"id:{id}",(100,250),cv2.FONT_HERSHEY_PLAIN,1,(255,50,255),2)
            carDown[id] = (cx, cy)
        
        if id in carDown:
            if cy2<(cy+offset) and cy2>(cy-offset):
                
                if counter1.count(id)==0:
                    counter1.append(id)
        
        if cy2<(cy+offset) and cy2>(cy-offset):
            cv2.rectangle(frame, (x3,y3),(x4,y4), (0,255,0),2,cv2.FILLED)
            cv2.putText(frame,f"id:{id}",(x3-5,y3),cv2.FONT_HERSHEY_PLAIN,1,(255,70,255),2)
            # cv2.putText(emptyFrame,f"id:{id}",(100,250),cv2.FONT_HERSHEY_PLAIN,1,(255,50,255),2)
            carUp[id] = (cx, cy)
        
        if id in carUp:
            if cy1<(cy+offset) and cy1>(cy-offset):
                cv2.rectangle(frame, (x3,y3),(x4,y4), (0,255,0),2,cv2.FILLED)
                cv2.putText(frame,f"id:{id}",(x3-5,y3),cv2.FONT_HERSHEY_PLAIN,1,(255,70,255),2)
                if counter2.count(id)==0:
                    counter2.append(id)

    cv2.line(frame, (363,375),(1150,375), (50,0,100),2)
    cv2.putText(frame, "L1", (358, 370), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,200),2)
    cv2.putText(frame, f"Down: {len(counter1)}", (80, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,200),2)


    cv2.line(frame, (268,430),(1217, 430), (50,250,100),2)
    cv2.putText(frame, "L2", (263, 425), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,200),2)
    cv2.putText(frame, f"UP: {len(counter2)}", (80, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,200),2)

    cv2.imshow("VehicalCounting", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
