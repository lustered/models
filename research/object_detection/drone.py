#!/usr/bin/python3
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
cap = cv2.VideoCapture('footage2.MOV')
# cap = cv2.VideoCapture(0)


from utils import label_map_util
from utils import visualization_utils as vis_util

MODEL_NAME = 'sunflower_graph'
MODEL_FILE = MODEL_NAME + '.tar.gz'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1 

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# file = open('output.txt', 'w')

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      ret, image_np = cap.read()
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=4)

      # distance detection ***
      for i, b in enumerate(boxes[0]):
          if classes[0][i] == 1: 
                  if scores[0][i] > 0.9:
                         mid_x = (boxes[0][i][3] + boxes[0][i][1] ) /2
                         mid_y = (boxes[0][i][2] + boxes[0][i][0] ) /2
                         apx_distance = round((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4 , 1)
                         cv2.putText(image_np, '{}'.format(apx_distance),(int(mid_x*800), int(mid_y*600)),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2) 
                         if apx_distance <= 0.1:
                             if mid_x > 0.3 and mid_x < 0.7:
                                 cv2.putText(image_np, 'IN RANGE',(int(mid_x*800) - 100, int(mid_y*600)+ 100),cv2.FONT_HERSHEY_SIMPLEX, 1.0 , (0 , 0 , 255), 3) 
                             # file.write('1')
                             # file.flush()
                         # else:
                             # file.write('0')
                             # file.flush()
                             
          
      cv2.imshow('sunflower detection', cv2.resize(image_np, (800,600)))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
