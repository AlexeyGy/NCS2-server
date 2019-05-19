import os
from flask import Flask, flash, request, redirect, url_for, jsonify
import util_mobilnet
import numpy as np
import os, cv2
import base64
from flask_cors import CORS

# settings
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
net = cv2.dnn.readNet('models/no_bn.xml', 'models/no_bn.bin') 

# set the target for the computation to the NCS2
net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)

# flask configuration
app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# wrapper needed to prevent cross-scripting permission issues
CORS(app)

# file name filter
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#processing of picture
@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        image = request.files['file']

        threshold = float(request.form.getlist('text')[0]);
        
        #image_string = base64.b64encode(image.read())
        nparr = np.fromstring(image.read(), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        res = process(img_np, net, threshold)
        return jsonify(res)
           

def process(image, net, threshold):
	image_processed = util_mobilnet.preprocess(image)
	net.setInput(image_processed)
	outputs = net.forward()
	return util_mobilnet.postprocess(image, outputs, threshold)
    
