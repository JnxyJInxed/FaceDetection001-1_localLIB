# import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt

#import darknet-post make(?)
import sys
sys.path.insert(1, './lib/noGPU/darknet')

from darknet import *

# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio

# run test on person.jpg image that comes with repository
def test_function():
  # run test on person.jpg image that comes with repository
  image = cv2.imread("data/person.jpg")
  detections, width_ratio, height_ratio = darknet_helper(image, width, height)

  for label, confidence, bbox in detections:
    left, top, right, bottom = bbox2points(bbox)
    left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(bottom * height_ratio)
    cv2.rectangle(image, (left, top), (right, bottom), class_colors[label], 2)
    cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                      (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      class_colors[label], 2)
  # cv2_imshow(image)
  return detections

from fastapi import FastAPI
import pickle
import json

app = FastAPI()

@app.on_event("startup")
# def load_model():
#     # global model
#     # model = pickle.load(open("model_tree.pkl", "rb"))

@app.get('/')
def index():
    from datetime import datetime
    
    time = datetime.now()
    testCase = test_function()
    message = 'This is the homepage of the API at ' + str(time) + str(testCase)
    return {'message': message}


# @app.post('/detect')
# def detect_image(data: InputImage):
#     received = data.dict()
#     image_Base64 = received['ImageBase64']

#     detect_result, predictions = detectImage(image_Base64) 
#     jsonString = json.dumps(predictions)
#     return {jsonString}
