import cv2
import numpy as np
import os

labels_path = r"C:\Users\Varun\Desktop\darknet\darknet-master\data\coco.names"
LABELS = open(labels_path).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(r"C:\Users\Varun\Desktop\darknet\darknet-master\cfg\yolov3.cfg" , r"C:\Users\Varun\Desktop\yolov3.weights")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                box = detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    # ensure at least one detection exists
    if len(idxs) > 0 :
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the frame
            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{LABELS[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # store the person's face in a variable
             
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if LABELS[class_ids[i]] == "person":
                face = frame[y:y+h, x:x+w]
                # check if the face region is valid
                if not face.size == 0:
                    # save the face to a file
                    cv2.imwrite(r"Faces/face.jpg", face)
                    if key == ord("q"):
                        break
            


                
               


    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
