import subprocess
import threading
import os


def capture_face():
    os.system("python b.py")
    stop_event.set()

def detect_face():
    while not stop_event.is_set():
        os.system("python c.py")

if __name__ == '__main__':
    stop_event = threading.Event()
    t1 = threading.Thread(target=capture_face)
    t2 = threading.Thread(target=detect_face)
    t2.start()
    t1.start()
    t1.join()
    stop_event.set()
    t2.join()
