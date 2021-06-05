# Image-Detection-with-Neural-Compute-Stick-RaspberryPi

The attached files are designed to be run on a Raspberry Pi with the Pi camera module using Intel's Neural Compute Stick 2.

Dependencies include standard NCS2 configurations (setupvars.sh, environment variables) as well as python packages: picamera, cv2, and smtplib.  The underlying inference model deploys MobileNet CNN converted to Intermediate Representation (graph format) for faster processing on the NCS2. 

To deploy model, openvino environment must be initialized on the local Pi device, and from the terminal command line:
python Inference_Dogs --prototxt mobilenet.prototxt --model mobilnet.caffemodel

This command will initiate streaming Pi camera, apply bounding boxes to images detected within frame (using OpenCV) and run inference using the IR MobileNet model on the compute stick.  Inferences that exceed the pre-set threshold and class criteria ('dog') will trigger the Python SMTP protocol client for email and/or SMS to local device.
