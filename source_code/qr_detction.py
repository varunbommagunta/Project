import cv2
import pyzbar.pyzbar as pyzbar

# initialize the video capture object
cap = cv2.VideoCapture(0)

# loop over frames from the video stream
while True:
    # grab the current frame
    ret, frame = cap.read()

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

    # if the 'q' key was pressed, break from the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# release the video capture object
cap.release()

# close all windows
cv2.destroyAllWindows()
