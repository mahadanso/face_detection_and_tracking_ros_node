#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from facenet_pytorch import MTCNN
import numpy as np
import mediapipe as mp
from collections import OrderedDict
from geometry_msgs.msg import Point
from face_detection.msg import facePose

class FaceDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.detector = MTCNN(keep_all=True)

        # Initialize ROS node
        rospy.init_node('face_detection_node', anonymous=True)

        # Subscribe to the /camera topic
        self.image_sub = rospy.Subscriber('/naoqi_driver/camera/front/image_raw', Image, self.image_callback)

        # Create a publisher to publish processed images
        self.data_pub = rospy.Publisher('faceDetection/data', facePose, queue_size=10)

        # Initialize face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=10, min_detection_confidence=0.5,min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils

        self.drawing_spec = self.mp_drawing.DrawingSpec(color=(128,128,128),thickness=1,circle_radius=1)

        # Initialize tracker
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.face_detection_data_msg = facePose()

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > 0:  # Disappeared threshold
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = self.distance_matrix(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > 0:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

    @staticmethod
    def distance_matrix(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a[:, np.newaxis] - b, axis=2)

    def image_callback(self, data):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        image = cv2.cvtColor(cv2.flip(cv_image,1),cv2.COLOR_BGR2RGB) #flipped for selfie view

        image.flags.writeable = False

        results = self.face_mesh.process(image)

        image.flags.writeable = True

        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        img_h , img_w, img_c = image.shape
        
        mutualGaze_list = []

        if results.multi_face_landmarks:
            for face_id, face_landmarks in enumerate(results.multi_face_landmarks):
                # Initialize face_2d and face_3d as lists for each detected face
                face_2d = []
                face_3d = []
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx ==1 or idx == 61 or idx == 291 or idx==199:
                        if idx ==1:
                            nose_2d = (lm.x * img_w,lm.y * img_h)
                            nose_3d = (lm.x * img_w,lm.y * img_h,lm.z * 3000)
                        x,y = int(lm.x * img_w),int(lm.y * img_h)

                        face_2d.append([x,y])
                        face_3d.append(([x,y,lm.z]))

                #Get 2d Coord
                face_2d = np.array(face_2d,dtype=np.float64)

                face_3d = np.array(face_3d,dtype=np.float64)

                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length,0,img_h/2],
                                    [0,focal_length,img_w/2],
                                    [0,0,1]])
                distortion_matrix = np.zeros((4,1),dtype=np.float64)

                success,rotation_vec,translation_vec = cv2.solvePnP(face_3d,face_2d,cam_matrix,distortion_matrix)


                #getting rotational of face
                rmat,jac = cv2.Rodrigues(rotation_vec)

                angles,mtxR,mtxQ,Qx,Qy,Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                # Determine if head is facing forward
                mutualGaze = abs(x) <= 5 and abs(y) <= 5
                mutualGaze_list.append(mutualGaze)

                # Display text (forward or not forward)
                text = "Forward" if mutualGaze else "Not Forward"
                label = f"Face {face_id + 1}: {text}"

                nose_3d_projection,jacobian = cv2.projectPoints(nose_3d,rotation_vec,translation_vec,cam_matrix,distortion_matrix)

                p1 = (int(nose_2d[0]),int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y*10), int(nose_2d[1] -x *10))

                cv2.line(image,p1,p2,(255,0,0),3)

                cv2.putText(image,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)

                self.mp_drawing.draw_landmarks(image=image,
                                        landmark_list=face_landmarks,
                                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                                        landmark_drawing_spec=self.drawing_spec,
                                        connection_drawing_spec=self.drawing_spec)

        cv2.imshow('Head Pose Detection',image)

        # Perform face detection
        boxes, probs, landmarks = self.detector.detect(cv_image, landmarks=True)

        # Prepare centroids for tracking
        input_centroids = []

        left_eyes = []
        right_eyes = []
        # Draw bounding boxes and facial landmarks
        if boxes is not None:
            for box, landmark, prob in zip(boxes, landmarks, probs):
                if prob > 0.7:
                    # Draw the bounding box
                    cv2.rectangle(cv_image,
                                (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])),
                                (0, 155, 255),
                                2)

                    # Draw the landmarks (eyes)
                    cv2.circle(cv_image, (int(landmark[0][0]), int(landmark[0][1])), 2, (0, 255, 0), 2)
                    cv2.circle(cv_image, (int(landmark[1][0]), int(landmark[1][1])), 2, (0, 255, 0), 2)

                    left_eyes.append(int(landmark[0][0]))
                    left_eyes.append(int(landmark[0][1]))
                    right_eyes.append(int(landmark[1][0]))
                    right_eyes.append(int(landmark[1][1]))

                    # Compute the centroid of the bounding box
                    centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                    input_centroids.append(centroid)

        # Update the tracker with the new centroids
        objects = self.update(input_centroids)

        # Draw the centroids and object IDs
        centroids = []
        for (object_id, c) in objects.items():
            centroid = Point(x=c[0], y=c[1], z=0)  # z = 0 for 2D image coordinates
            centroids.append(centroid)

            text = f"ID {object_id}"
            cv2.putText(cv_image, text, (c[0] - 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(cv_image, (c[0], c[1]), 4, (0, 255, 0), -1)

        # Display the image with bounding boxes, landmarks, and centroids
        cv2.imshow('Face Detection', cv_image)
        cv2.waitKey(1)

        # Publish the processed image
        self.face_detection_data_msg.centroids = centroids
        self.face_detection_data_msg.mutualGaze = any(p for p in mutualGaze_list)
        self.data_pub.publish(self.face_detection_data_msg)

if __name__ == '__main__':
    try:
        print("Starting face detection node")
        FaceDetectionNode()
        print("Face detection node started successfully!!")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
