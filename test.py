from subprocess import Popen
from detect import detect

# process = Popen(["python3", "detect.py"])
process = Popen(["python3", "detect.py", "--source", "uploads/original-801366f61633290e5387331e74e0753f.jpg", "--weights","yolov7.pt"])
process.wait()