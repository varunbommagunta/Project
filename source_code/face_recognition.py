import cv2
import os
import face_recognition
from w_message import send_alert
from shutil import copy


# Load the faces of the people in the folder "Friends"
friends = []
for friend_image in os.listdir("Friends"):
    friend = face_recognition.load_image_file(f"Friends/{friend_image}")
    friends.append(friend)

# Load the extracted face
while True:
    extracted_face = face_recognition.load_image_file("Faces/face.jpg")

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
            print("Match found with a face in the folder 'Friends'")
        else:
            print("No match found with any face in the folder 'Friends'")
            copy("Faces/face.jpg",r"E:\project\intruders\face.jpg")
            send_alert(r"E:\project\intruders\face.jpg")
    else:
        print("No face found in the extracted image")
  



