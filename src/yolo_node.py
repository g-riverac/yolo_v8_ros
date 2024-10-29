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


# model
#model = YOLO("/home/nathy/Documents/Door_detectio_2024/best_door.pt")
#model_knob = YOLO("/home/nathy/Documents/Door_detectio_2024/best_knob.pt")

class YoloNode:
    def __init__(self):
        rospy.init_node('yolo_inference_node', anonymous=True)
        self.cam_sub = rospy.Subscriber('/camera/usb_cam/image_raw', Image, self.cam_callback)
        self.pub = rospy.Publisher('robot/tension_value', Float64, queue_size=10)
        self.image = None
        self.model_door = YOLO("/home/nathy/Documents/Door_detectio_2024/best_door.pt")
        self.model_knob = YOLO("/home/nathy/Documents/Door_detectio_2024/best_knob.pt")
        print("OK")

    def cam_callback(self, msg):
        bridge = CvBridge()
        try:
           self.image = bridge.imgmsg_to_cv2(msg, "bgr8")  # Asumiendo que el formato es BGR8
        except Exception as e:
            rospy.logerr(f"Error al convertir la imagen: {e}")
            return
        #cv2.imshow("Image from Camera", cv_image)
        #cv2.waitKey(1)
        self.door_detection_publish()

    def door_detection_publish(self):
        classNames = ["door","handle","hinge","knob"]
        results = self.model_door(self.image, stream=True)
        img = self.image

        # coordinates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extraer las coordenadas
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #x1, y1, x2, y2 = map(int, box)
                # Extraer la imagen de la puerta (ROI)
                door_img = img[y1:y2, x1:x2]
                results_knob = self.model_knob(door_img, stream=True)
                for result in results_knob:
                    boxes_knob = result.boxes
                    for b in boxes_knob:
                        xk1, yk1, xk2, yk2 = b.xyxy[0]
                        xk1, yk1, xk2, yk2 = int(xk1), int(yk1), int(xk2), int(yk2)
                        cv2.rectangle(door_img, (xk1, yk1), (xk2, yk2), (0, 0, 255), 3)
                        cls_name = int(b.cls[0])+1
                        print("Class name -->", classNames[cls_name])
                        # object details
                        org = [xk1, yk1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(door_img, classNames[cls_name], org, font, fontScale, color, thickness)
                
                img[y1:y2, x1:x2] = door_img
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->",confidence)

                # class name
                cls = int(box.cls[0])
                print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
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



