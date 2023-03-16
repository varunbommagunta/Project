import cv2
import numpy as np
import threading
import cv2
import pyzbar.pyzbar as pyzbar
import cv2
from torchvision.transforms import transforms
import torchvision.models as models
import torch
import numpy as np
from PIL import Image
import numpy as np
import librosa
import sounddevice as sd
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np
import os
import cv2
import os
import face_recognition
from w_message import send_alert
from shutil import copy
from speech import run_speech
tf.compat.v1.enable_eager_execution()
from flask import Flask, render_template, jsonify, request
import json
from flask import Flask, render_template
import threading
import speech_recognition as sr
import tensorflow as tf
import pyttsx3
import cv2
import pyzbar.pyzbar as pyzbar

# Initialize the TTS engine
  # Set the speaking rate (words per minute)

face_rec_condition = threading.Event()

app = Flask(__name__)

fire_output = ""
face_output = ""
glass_output = ""
face_rec_output = ""



labels_path = r"C:\Users\Varun\Desktop\darknet\darknet-master\data\coco.names"
LABELS = open(labels_path).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
dummy_face_loc = r"E:\project\utils\istockphoto-1013725314-612x612.jpg"
dummy_face = cv2.imread(dummy_face_loc)
net = cv2.dnn.readNetFromDarknet(r"C:\Users\Varun\Desktop\darknet\darknet-master\cfg\yolov3.cfg" , r"C:\Users\Varun\Desktop\yolov3.weights")



# Create video capture object for default camera
cap = cv2.VideoCapture(0)

# Initialize face detection classifier
#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



def find_matching_face():
    # Load the faces of the people in the folder "Friends"
    global face_output
    face_output = "No face found in the extracted image"
    friends = []
    for friend_image in os.listdir("Friends"):
        friend = face_recognition.load_image_file(f"Friends/{friend_image}")
        friends.append(friend)

    # Load the extracted face
    while True:
        extracted_face = face_recognition.load_image_file(r"E:\project\demo\camera\Faces\face.jpg")

        # Find the encoding of the extracted face
        extracted_face_encodings = face_recognition.face_encodings(extracted_face)
        if len(extracted_face_encodings) > 0:
            extracted_face_encoding = extracted_face_encodings[0]

            # Compare the encoding of the extracted face with the encodings of the friends
            match = False
            for friend in friends:
                friend_encodings = face_recognition.face_encodings(friend)
                if len(friend_encodings) > 0:
                    friend_encoding = friend_encodings[0]
                    result = face_recognition.compare_faces([friend_encoding], extracted_face_encoding)
                    if result[0]:
                        match = True
                        break

            if match:
                face_output = "Match found with a face in the folder 'Friends'"
            else:
                face_output = "No match found with any face in the folder 'Friends'"
                copy(r"E:\project\demo\camera\Faces\face.jpg",r"E:\project\demo\camera\intruders\face.jpg")
                #send_alert(r"E:\project\intruders\face.jpg")
                face_rec_condition.set()
                #exit()
        else:
            face_output = "No face found in the extracted image"


def face_processor(frame):
    global face_rec_output
    face_rec_thread = "No Face"
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
                        cv2.imwrite(r"E:\project\demo\camera\Faces\face.jpg", face)
                        face_rec_output = "Face saved"
                        
                    else:
                        cv2.imwrite(r"E:\project\demo\camera\Faces\face.jpg", dummy_face)
                        face_rec_output = "No Face"
                
                else:
                    face_rec_thread = "No Face"
                                    

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

# Define function to process frames with fire.py
def fire_processor(frame):
    global fire_output
    # Load the model
    model = torch.load(r"E:\project\Fire-Smoke-Detection-master (1)\Fire-Smoke-Detection-master\trained-models\model_final.pth", map_location=torch.device('cpu'))

    # Define the class names
    class_names = ['Fire', 'Neutral', 'Smoke']

    # Define the prediction transform
    prediction_transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Start the video capture

    # Loop over the frames of the video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break
        # Read the next frame

        # Resize the frame and convert it from BGR to RGB
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the NumPy array to a PIL image
        pil_image = Image.fromarray(frame_rgb)

        # Apply the prediction transform
        image = prediction_transform(pil_image)[:3, :, :].unsqueeze(0)

        # Make the prediction
        pred = model(image)
        idx = torch.argmax(pred)
        prob = pred[0][idx].item() * 100
        prediction = class_names[idx]

        # Overlay the prediction on the frame
        if prediction == 'Neutral':
            color = (0, 255, 0)
            fire_output = "Clear"
        else:
            color = (0, 0, 255)
            fire_output = "Emergency Detected"
        cv2.putText(frame, f'{prediction} {prob:.2f}%', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show the frame
        #cv2.imshow('Frame', frame)

        # Check if the user pressed the 'q' key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


def detect_glass_break():
    global glass_output
    glass_output = "Clear"
    # Load the trained model
    model_path = r"E:\project\Sounds\glass_sound_detection.h5"
    model = load_model(model_path)
    n_mels=128
    n_steps=128
    threshold=70

    # Define a function to convert an audio array to a mel-spectrogram
    def array_to_melspec(audio):
        # Convert to mel-spectrogram
        spec = librosa.feature.melspectrogram(audio, sr=22050, n_mels=n_mels)
        # Resize the spectrogram to n_steps x n_mels
        spec = librosa.util.fix_length(spec, n_steps, axis=1)
        # Convert to decibel scale
        spec = librosa.power_to_db(spec, ref=np.max)
        return spec

    # Define a function to record audio from the microphone
    def record(duration):
        # Set the sample rate and number of channels
        sr = 22050
        channels = 1
        # Record the audio
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=channels)
        sd.wait()
        # Convert to mono if necessary
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        return audio

    # Define a function to predict the class label of a mel-spectrogram
    def predict_class(spec):
        spec = np.expand_dims(spec, axis=0)
        prediction = model.predict(spec)
        label_map = {1: 'glass_breaking', 0: 'other'}
        predicted_label = np.argmax(prediction)
        return label_map[predicted_label], prediction[0][predicted_label] * 100

    # Listen for glass breaking sound
    while True:
        print('Listening for glass breaking sound...')
        audio = record(duration=5)  # Record 5 seconds of audio
        spec = array_to_melspec(audio)  # Convert audio to mel-spectrogram
        predicted_label, predicted_prob = predict_class(spec)  # Make a prediction
        print(f"{predicted_label} probability: {predicted_prob:.2f}%")
        if predicted_label == 'glass_breaking' and predicted_prob >= threshold:
            glass_output =  "Glass Breaking Sound Detected"
            return True

def speech_function(frame):

    face_rec_condition.wait()
    # Initialize the recognizer
    r = sr.Recognizer()
    engine = pyttsx3.init()

    # Set the properties of the voice
    engine.setProperty('rate',250) 

    while True:
        # Use the microphone as the audio source
        with sr.Microphone() as source:
            # Ask the user for input
            text = "Hello! How can I help you?"
            print(text)
            engine.say(text)
            #engine.runAndWait()
            audio = r.listen(source)
            
            # Use the Google speech recognition API to convert speech to text
            try:
                text = r.recognize_google(audio)
                print("You said: ", text)
                # Check for the user's response
                if "hello" in text.lower():
                    #qr detection
                    while True:
                        # grab the current frame
                        ret, frame = cap.read()
                        engine.say("Show QR code")

                        # find and decode the QR codes in the frame
                        decoded_objs = pyzbar.decode(frame)

                        # loop over the detected QR codes
                        for obj in decoded_objs:
                            # extract the QR code data
                            data = obj.data.decode("utf-8")

                            # save the QR code data in a file
                            with open("qrcode_data.txt", "w") as f:
                                f.write(data)
                            
                            # break the loop if data is found
                            break

                        # show the output frame
                        cv2.imshow("QR Code Scanner", frame)

                        if 'data' in locals():
                            print("Successfully read the data")
                            break
                    break  # exit the loop once the QR code has been detected
                else:
                    print("I'm sorry, I didn't understand what you said.")
            except:
                # If there is a problem with the audio or recognition, ask the user again
                print("I'm sorry, I didn't catch that. Could you please repeat yourself?")



# Define function to read frames from camera and process them
def process_frames():
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            break

        # Create separate threads for each program to process the frame concurrently
        fire_thread = threading.Thread(target=fire_processor, args=(frame,))
        face_thread = threading.Thread(target=face_processor, args=(frame,))
        glass_thread = threading.Thread(target= detect_glass_break)
        face_rec_thread = threading.Thread(target = find_matching_face)
        speech_thread = threading.Thread(target = speech_function, args = (frame,))


        # Start the threads
        fire_thread.start()
        face_thread.start()
        glass_thread.start()
        face_rec_thread.start()
        speech_thread.start()

        # Wait for all threads to finish before processing the next frame
        fire_thread.join()
        face_thread.join()
        glass_thread.join()
        face_rec_thread.join()
        speech_thread.join()

        '''# Display the original frame and the processed frame side by side
        cv2.imshow('Original Frame', frame)
        #cv2.imshow('Processed Frame', processed_frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break'''

    # Release video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

@app.route("/ajax")
def ajax():
    return jsonify(fire_output=fire_output, face_output=face_output, glass_output=glass_output, face_rec_output=face_rec_output)

@app.route("/")
def index():
    return render_template("index.html")

# create a thread for the Flask app
flask_thread = threading.Thread(target=app.run)

# create a thread for the process_frames function
processing_thread = threading.Thread(target=process_frames)

# start both threads
flask_thread.start()
processing_thread.start()
