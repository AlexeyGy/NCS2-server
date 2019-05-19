import cv2
import numpy as np
PREPROCESS_DIMS = (300, 300)

def preprocess_image(input_image):
	# preprocess the image
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float32)
	preprocessed = preprocessed[np.newaxis,:,:,:]
	preprocessed = preprocessed.transpose((0,3,1,2))
	# return the image to the calling function
	return preprocessed
 
def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out[0,0,:,3:7] * np.array([w, h, w, h])
 
    cls = out[0,0,:,1]
    conf = out[0,0,:,2]
    return (box.astype(np.int32), conf, cls)
 
