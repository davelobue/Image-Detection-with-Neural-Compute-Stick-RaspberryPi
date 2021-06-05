# Image-Detection-with-Neural-Compute-Stick-RaspberryPi

The attached files are designed to be run on a Raspberry Pi with the Pi camera module using Intel's Neural Compute Stick 2.

Dependencies include standard NCS2 configurations (setupvars.sh, environment variables) as well as python packages: picamera, cv2, and smtplib.  The underlying inference model deploys MobileNet CNN converted to Intermediate Representation (graph format) for faster processing on the NCS2. 
