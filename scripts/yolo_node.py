#! /usr/bin/env python3


import torch
from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge
import rospy
from geometry_msgs.msg import Pose
import math
from std_msgs.msg import Float32MultiArray, Float64
from sensor_msgs.msg import Image

class YoloNode:
    def __init__(self):
        rospy.init_node('yolo_inference_node', anonymous=True)
        self.cam_sub = rospy.Subscriber('/camera/usb_cam/image_raw', Image, self.cam_callback)
        self.pub = rospy.Publisher('robot/tension_value', Float64, queue_size=10)
        self.image = None
        self.bridge = CvBridge()
        self.model_door = YOLO("../best_door.pt")
        self.model_knob = YOLO("../best_knob.pt")
        self.results=[]
        self.results_knob=[]
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 1
        self.color = (255, 0, 0)
        self.thickness = 2
        print("models loaded: OK")

    def cam_callback(self, msg):
        
        try:
           self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")  # Asumiendo que el formato es BGR8
        except Exception as e:
            rospy.logerr(f"Error al convertir la imagen: {e}")
            return

        self.door_detection_publish()

    def door_detection_publish(self):
        classNames = ["door","handle","hinge","knob"]
        self.results = self.model_door(self.image, stream=True)
        img = self.image

        # coordinates
        for r in self.results:
            annotated_img = r.plot()
            boxes = r.boxes
            for box in boxes:
                # Extraer las coordenadas
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


                # Extraer la imagen de la puerta (ROI)
                door_img = img[y1:y2, x1:x2].copy()
                self.results_knob = self.model_knob(door_img, stream=True)
                if self.results_knob:
                    for result in self.results_knob:
                        door_annotated = result.plot()

                        annotated_img[y1:y2, x1:x2] = door_annotated 
                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])



        cv2.imshow('Webcam', annotated_img)
        cv2.waitKey(1)
        #if cv2.waitKey(1) == ord('q'):
        #    break

        #cap.release()
        #cv2.destroyAllWindows()
    	#self.pub.publish(output_msg)

if __name__ == '__main__':
    try:
        node = YoloNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass



