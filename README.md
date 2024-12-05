# WebCameraでリアルタイム物体検出AI（YOLO）を動かしてみよう

## 学習済み汎用モデル
[ここからダウンロード](https://huggingface.co/Ultralytics/YOLOv8/tree/main)しましょう。
- yolov8n.pt
- yolov8s.pt
- yolov8m.pt
- yolov8l.pt
- yolov8x.pt
※下のモデルほど精度が高いですが、データサイズが大きく、処理速度が遅くなります。

## 標準コード
ダウンロードしたモデルファイルに合わせて、load modelの文字列を変更しましょう。
標準では、yolov8n.ptを読み込む設定にしています。
```
from ultralytics import YOLO
import cv2
import math 
import csv 
import pprint 

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# load model
model = YOLO("yolov8n.pt")

# classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # class name
            cls = int(box.cls[0])

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 判定枠を人物のみに限定し、検出数を画面左上に表示

---
from ultralytics import YOLO
import cv2
import math 
import csv 
import pprint 

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# load model
model = YOLO("yolov8s.pt")

# classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes
        counter = 0

        for box in boxes:
            # class name
            cls = int(box.cls[0])

            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            if cls == 0:
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->", confidence)
            print("Class name -->", classNames[cls])

            # person only
            if cls == 0:
                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                counter+=1
                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        #count view
        p_up = [15, 30]
        font_up = cv2.FONT_HERSHEY_SIMPLEX
        fontScale_up = 1
        color_up = (255, 0, 0)
        thickness_up = 2
        cv2.putText(img, str(counter), p_up, cv2.FONT_HERSHEY_SIMPLEX, fontScale_up, color_up, thickness_up)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
---
