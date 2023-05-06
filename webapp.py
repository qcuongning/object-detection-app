"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse

import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from werkzeug.utils import send_from_directory
import os
import re
import requests
import time
from models.experimental import attempt_load
from utils.general import  non_max_suppression, scale_coords
from utils.plots import plot_one_box

app = Flask(__name__)


def detect(path):
    img = cv2.imread(path)
    img_or = img.copy()
    img = cv2.resize(img, (320, 320))
    img = img/255.
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img).float()
    pred = model(img)[0]
    pred = non_max_suppression(pred, 0.25, 0.45, agnostic=False)

    det = pred[0]
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_or.shape).round()
    for *xyxy, conf, cls in reversed(det):
        label = f'{names[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img_or, label=label, color=colors[int(cls)], line_thickness=1)
        cv2.imwrite(path, img_or)





@app.route("/")
def hello_world():
    return render_template('object-detect.html')

def get_frame():
    # folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    filename = imgpath    
    image_path = 'upload/'+filename    
    video = cv2.VideoCapture(image_path)  # detected video path
    #video = cv2.VideoCapture("video.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.jpg', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    # folder_path = 'runs/detect'
    # subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    # latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    # directory = folder_path+'/'+latest_subfolder
    directory = "uploads"
    print("printing directory: ",directory)  
    filename = imgpath
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'jpg':      
        return send_from_directory(directory,filename,environ)

    elif file_extension == 'mp4':
        return render_template('object-detect.html')

    else:
        return "Invalid file format"

    
@app.route("/", methods=["GET", "POST"])
def predict_img():
    print("submit")
    # global imgpath
    # if request.method == "POST":
    #     if 'file' in request.files:
    #         f = request.files['file']
    #         basepath = os.path.dirname(__file__)
    #         filepath = os.path.join(basepath,'uploads',f.filename)
    #         print("upload folder is ", filepath)
    #         f.save(filepath)
            
    #         imgpath = f.filename
    #         print("printing predict_img :::::: ", predict_img)

    #         file_extension = f.filename.rsplit('.', 1)[1].lower()    
    #         detect(filepath)

            
    folder_path = 'runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename 
    return render_template('index.html', image_path=image_path)
    #return "done"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    imgpath = ""
    weights = "./yolov7.pt"
    device = "cpu"
    model = attempt_load(weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in names]

    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat

