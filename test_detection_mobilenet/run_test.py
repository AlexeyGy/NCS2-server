import numpy as np
import cv2
import traceback
import util
 
def test():
	CLASSES = []
	label_map_path = '../models/labelmap.prototxt'
	
	with open(label_map_path) as f:
		lines = f.readlines()
	for x in range(3, len(lines),5):
		CLASSES.append(((lines[x].split(": "))[1]).replace("\"","").replace("\n",""))
	print (CLASSES)
	
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	
	# frame dimensions should be sqaure
	PREPROCESS_DIMS = (300, 300)
	DISPLAY_DIMS = (900, 900)
	
	# calculate the multiplier needed to scale the bounding boxes
	DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]
	
	
	# Load the model 
	net = cv2.dnn.readNet('../models/no_bn.xml', '../models/no_bn.bin') 

	# Specify target device 
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)


	for filename in ["test1.png", "test2.jpg", "test3.jpg"]:

		image = cv2.imread(filename)
		origimg = np.copy(image)
		image = util.preprocess_image(image)

		#image = image.transpose((2, 0, 1))
		#blob = cv2.dnn.blobFromImage(image, size=PREPROCESS_DIMS) 

		net.setInput(image)
		outputs = net.forward()

		box, conf, cls = util.postprocess(origimg, outputs)

		for i in range(len(box)):
				p1 = (box[i][0], box[i][1])
				p2 = (box[i][2], box[i][3])
				cv2.rectangle(origimg, p1, p2, (0,255,0))
				p3 = (max(p1[0], 15), max(p1[1], 15))
				title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
				cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
				
		cv2.imwrite(filename.split(".")[0] + "SSD.jpg", origimg)

if __name__ =='__main__':
	"""test the image recognition via three test images"""
	test()