import numpy as np
import cv2
 
# frame dimensions should be square
PREPROCESS_DIMS = (300, 300)

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

