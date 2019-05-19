import argparse
import numpy as np
import time
import cv2
import traceback
 
# initialize the list of class labels our network was trained to
# detect, then generate a set of bounding box colors for each class
 
# frame dimensions should be square
PREPROCESS_DIMS = (300, 300)

def read_labels(label_map_path = 'labelmap.prototxt'):
    CLASSES = []
    with open(label_map_path) as f:
        lines = f.readlines()
    for x in range(3, len(lines),5):
        CLASSES.append(((lines[x].split(": "))[1]).replace("\"","").replace("\n",""))

 
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    
    return COLORS, CLASSES

COLORS, CLASSES = read_labels()

def preprocess(input_image):
	# preprocess the image
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float32)
	preprocessed = preprocessed[np.newaxis,:,:,:]
	preprocessed = preprocessed.transpose((0,3,1,2))
	# return the image to the calling function
	return preprocessed
 
def postprocess(img, out, threshold):   
    h = img.shape[0]
    w = img.shape[1]
    box = out[0,0,:,3:7] * np.array([w, h, w, h])
    box = box.astype(np.int32)
    #class
    cls = out[0,0,:,1]
    conf = out[0,0,:,2]
    # detections for the classes schraube, duebel, montiert
    detections = [0, 0, 0]
    result = []
    for i in range(len(box)):
        # filter out background class
        if cls[i] > 0 and conf[i] >= threshold:
           # -1 necesssary for background class
           detections[int(cls[i]-1)] += 1
           #title = '{} ({}%)'.format(CLASSES[int(cls[i])], round(conf[i] *100, 2))
           title = CLASSES[int(cls[i])]
	   
           result.append([int(box[i][0]), int(box[i][1]), int(box[i][2]), int(box[i][3]), title, str(round(conf[i],2))])
	   
    # testing only
    # test_write_image(img, conf, box, cls)
    
    return (detections, result)


# def test_write_image(origimg, conf, box, cls):

#     for i in range(len(box)):
# 	       # x,y
# 	       p1 = (box[i][0], box[i][1])
# 	       # x+wdith, y+width
# 	       p2 = (box[i][2], box[i][3])
# 	       cv2.rectangle(origimg, p1, p2, (0,255,0))
# 	       p3 = (max(p1[0], 15), max(p1[1], 15))
# 	       title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
# 	       cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
	       
#     cv2.imwrite("SSD.jpg", origimg)
 


