import pywhatkit as kit
from datetime import datetime
	
def send_alert(image_path):
	phone_number = "+999999999999"
	kit.sendwhats_image(phone_number,img_path = image_path,caption = "Check this",wait_time = 7)
