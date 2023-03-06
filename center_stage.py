import cv2
import mediapipe as mp
import numpy as np
from vidstab import VidStab
import statistics


camera_width= 640
camera_height = 480

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
gb_zoom = 1.2

def zoom_image(img,coord=None,zoom=None):
  global gb_zoom

  if zoom==1 and gb_zoom<3.0:
    gb_zoom = gb_zoom + 0.1
    print("Zoom in") 
  if zoom ==0 and gb_zoom>1.2:
    gb_zoom = gb_zoom - 0.1
    print("Zoom out") 
  rot_mat = cv2.getRotationMatrix2D(coord, 0, gb_zoom)
  
  # Use warpAffine to make sure that  lines remain parallel
  result = cv2.warpAffine(image, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

  return result



def makezoomandcenter(right_ear_x,left_ear_x,coordinates):
  if not coordinates:
    return (image.shape[0]//2,image.shape[1]//2),0
  max_dist = 0 
  zoom = 0
  for index_r,right_ear in enumerate(right_ear_x):
    for index_l,left_ear in enumerate(left_ear_x):
      curr_dist=abs(right_ear-left_ear)
      if curr_dist >max_dist:
        max_r = index_r
        max_l = index_l
        max_dist = curr_dist
  if max_dist < image.shape[0]//6:
    zoom = 1
  if max_l==max_r:
    return coordinates[max_l],zoom
  fin_coord = tuple((x+y)//2 for x,y in zip(coordinates[max_l],coordinates[max_r]))
  return fin_coord,zoom



def frame_adjuster(image):
  detection=None
  coordinates = []
  mp_face_detection=mp.solutions.face_detection

  right_ear_x = []
  left_ear_x = []
  with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3) as face_detection:
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    # Draw the face detection annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detections:
      for detection in results.detections:
        height, width, channels = image.shape
        nose = detection.location_data.relative_keypoints[2]
        bounding_box_width = int(detection.location_data.relative_bounding_box.width*width)
        bounding_box_height = int(detection.location_data.relative_bounding_box.height *height )
        right_ear = detection.location_data.relative_keypoints[4]
        left_ear = detection.location_data.relative_keypoints[5]

        #  get coordinates for right ear and left ear
        right_ear_x.append(int(right_ear.x * width))
        left_ear_x.append(int(left_ear.x * width))
        # Fetch coordinates for the nose and set as center
        center_x = int(nose.x * width)
        center_y = int(nose.y * height)
        coordinates.append((center_x, center_y))
  coordinates,zoom =makezoomandcenter(right_ear_x,left_ear_x,coordinates)
  #print(zoom)
  return coordinates,zoom

# define a video capture object
cap = cv2.VideoCapture(0,cv2.CAP_ANY)
#cap = cv2.VideoCapture("stock.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)  # set new dimensions to cam object (not cap)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
cap.set(cv2.CAP_PROP_FPS, 120)
stabilizer = VidStab()


fin_coord = (camera_width/2,camera_height/2)
fin_zoom= 0
i=0
while cap.isOpened():
  success, image = cap.read()
  if not success:
    print("Ignoring empty camera frame.")
    # If loading a video, use 'break' instead of 'continue'.
    continue
  # To improve performance, optionally mark the image as not writeable to
  # pass by reference. 
  curr_coordinate,curr_zoom=frame_adjuster(image)
  if curr_coordinate== (image.shape[0]//2,image.shape[1]//2):
     i=i+1
  if i%5==0 or curr_coordinate!= (image.shape[0]//2,image.shape[1]//2):
    fin_coord= curr_coordinate
    fin_zoom = curr_zoom
    image=zoom_image(image,fin_coord,fin_zoom)
    #image = stabilizer.stabilize_frame(input_frame=image,
                                            #smoothing_window=2, border_size=-20)
    image = cv2.resize(image,image.shape[1::-1],
                            interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Center_stage', cv2.flip(image, 1))
    i=1
  else:
    image=zoom_image(image,fin_coord,fin_zoom)
    image = cv2.resize(image,image.shape[1::-1],
                            interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Center_stage', cv2.flip(image, 1))
  if cv2.waitKey(5) & 0xFF == 27:
    break
cap.release()