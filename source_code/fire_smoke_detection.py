import cv2
from torchvision.transforms import transforms
import torchvision.models as models
import torch
import imutils
from imutils.video import VideoStream
import numpy as np
from PIL import Image


# Load the model
model = torch.load(r"E:\project\Fire and Smoke\Fire-Smoke-Detection-master\trained-models\model_final.pth", map_location=torch.device('cpu'))

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

# Start the video stream
vs = VideoStream(src=0).start()

# Loop over the frames of the video
while True:
    # Read the next frame
    frame = vs.read()

    # Resize the frame and convert it from BGR to RGB
    frame = imutils.resize(frame, width=640)
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
    else:
        color = (0, 0, 255)
    cv2.putText(frame, f'{prediction} {prob:.2f}%', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Show the frame
    cv2.imshow('Frame', frame)

    # Check if the user pressed the 'q' key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up
vs.stop()
cv2.destroyAllWindows()
