import cv2
import face_recognition
import os

# Load the faces of the people in the folder "Friends"
friends = []
for friend_image in os.listdir("Friends"):
    friend = face_recognition.load_image_file(f"Friends/{friend_image}")
    friends.append(friend)

# Initialize the video capture object
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame from the camera
    ret, frame = video_capture.read()

    # Convert the frame to RGB format
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Iterate through each face in the current frame
    for face_encoding in face_encodings:
        # Compare the encoding of the current face with the encodings of the friends
        match = False
        for friend in friends:
            friend_encoding = face_recognition.face_encodings(friend)[0]
            result = face_recognition.compare_faces([friend_encoding], face_encoding)
            if result[0]:
                match = True
                break

        # Display the result of the face recognition
        if match:
            cv2.putText(frame, "Match found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No match found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the current frame
    cv2.imshow('Video', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
