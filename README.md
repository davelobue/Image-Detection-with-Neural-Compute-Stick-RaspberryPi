# Image-Detection-with-Neural-Compute-Stick-RaspberryPi

The attached files are designed to be run on a Raspberry Pi with the Pi camera module using Intel's Neural Compute Stick 2.

Dependencies include standard NCS2 configurations (setupvars.sh, environment variables) as well as python packages: picamera, cv2, and smtplib.  The underlying inference model deploys MobileNet CNN converted to Intermediate Representation (graph format) for faster processing on the NCS2. 

To deploy model, openvino environment must be initialized on the local Pi device, and from the terminal command line:
python Inference_Dogs --prototxt mobilenet.prototxt --model mobilnet.caffemodel

This command will initiate streaming Pi camera, apply bounding boxes to images detected within frame (using OpenCV) and run inference using the IR MobileNet model on the compute stick.  Inferences that exceed the pre-set threshold and class criteria ('dog') will trigger the Python SMTP protocol client for email and/or SMS to local device.


Summary:

The core python dependency is OpenCV library for image streaming and bounding box creation.  A separate library (imutils) provides efficient packaging of many OpenCV functions and capabilities.  The baseline model used for image detection inference is MobileNet, which was designed for efficient convolutional neural network implementation on mobile vision applications.  Models deployed on the NCS2 with OpenVINO are required to be in Intermediate Representation (IR), which is a form of graph representation that includes an XML file and a binary file.  Converting files from common platforms like Tensorflow, PyTorch, Caffe is a prerequisite for deployment and requires a multi-step process of configuring model optimizer, freezing the model, and converting to IR prior to being deployed on Intel’s inference engine.

When the application is deployed, the live stream managed by the OpenCV package runs on RPi RAM to identify objects within each frame and applies bounding boxes for image localization and centering.  Detected images from bounding boxes are then processed for inference calculations using the added speed of NCS2 and IR of MobileNet pre-trained model.  If inference confidence exceeds the pre-configured threshold (e.g., 40%) a function is triggered that calls Python’s built in SMTP protocol client to send an email or text notifying end users the desired object has been detected and is frame. 


References:

MobileNets:  Efficient Convolutional Neural Networks for Mobile Vision Applications:  https://arxiv.org/abs/1704.04861
OpenCV: https://opencv.org/
OpenVino:  https://docs.openvinotoolkit.org/latest/index.html
imutils:  https://www.pyimagesearch.com/2019/04/08/openvino-opencv-and-movidius-ncs-on-the-raspberry-pi/
SMTPlib:  https://docs.python.org/3/library/smtplib.html
