import cv2
import mediapipe as mp
import math
import numpy as np
from pynput.keyboard import Key, Controller
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = []
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape
    annotated_image = image.copy()
    for hand_landmarks in results.multi_hand_landmarks:
      print('hand_landmarks:', hand_landmarks)
      print(
          f'Index finger tip coordinates: (',
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
          f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
      )
      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())
    cv2.imwrite(
        '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    # Draw hand world landmarks.
    if not results.multi_hand_world_landmarks:
      continue
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
      

import numpy as np

def rotate_image(img, angle):
  img_center = tuple(np.array(img.shape[1::-1])/2)
  rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags = cv2.INTER_LINEAR)
  return result


def overlay_images(background, foreground, offset=(0, 0)):
  
    # Ensure offset is within bounds
    y_offset, x_offset = offset
    y1, y2 = max(0, y_offset), min(background.shape[0], y_offset + foreground.shape[0])
    x1, x2 = max(0, x_offset), min(background.shape[1], x_offset + foreground.shape[1])

    # Handle the case when the offset is out of bounds
    fg_y1, fg_y2 = max(0, -y_offset), min(foreground.shape[0], background.shape[0] - y_offset)
    fg_x1, fg_x2 = max(0, -x_offset), min(foreground.shape[1], background.shape[1] - x_offset)

    # Extract the region of interest from the background
    background_roi = background[y1:y2, x1:x2]

    # Extract the corresponding part of the foreground image
    foreground_roi = foreground[fg_y1:fg_y2, fg_x1:fg_x2]

    # Separate the alpha channel and the RGB channels
    alpha = foreground_roi[:, :, 3] / 255.0
    alpha = np.stack([alpha, alpha, alpha], axis=-1)  # Convert to (height, width, 3)

    # Composite the images
    composite = background_roi * (1 - alpha) + foreground_roi[:, :, :3] * alpha

    # Replace the region in the background with the composite image
    background[y1:y2, x1:x2] = composite

    return background

  

wheel_image = cv2.imread(r"c2.png", cv2.IMREAD_UNCHANGED)
mouse = Controller()
# For webcam input:
pre1 = -1
pre2 = -1
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    if results.multi_hand_landmarks:
      hand_center = []
      thumb = []
      for hand_landmarks in results.multi_hand_landmarks:
        hand_center.append(
          [int(hand_landmarks.landmark[9].x*width), int(hand_landmarks.landmark[9].y*height)]) 
        for j in range(2,5):
          thumb.append(int(hand_landmarks.landmark[j].y*height))
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      
      if len(hand_center) == 2:
        center_x = (hand_center[0][0] + hand_center[1][0])//2
        center_y = (hand_center[0][1] + hand_center[1][1])//2
        radius = int(math.sqrt((hand_center[0][0]-hand_center[1][0])**2+
                               (hand_center[0][1]-hand_center[1][1])**2)/2)
        if hand_center[0][0] < hand_center[1][0]:
          angle =  np.degrees(np.arctan2(hand_center[1][1]-hand_center[0][1], hand_center[1][0]-hand_center[0][0]))
        else:
          angle =  np.degrees(np.arctan2(hand_center[0][1]-hand_center[1][1], hand_center[0][0]-hand_center[1][0]))

        rotate_wheel = rotate_image(wheel_image, 180 - angle)        
        try:
          overlay_images(image, cv2.resize(rotate_wheel, (2*radius, 2*radius)), (center_y-radius, center_x-radius))
        except cv2.error as e:
          print('error')
          continue

        if angle < -10 and (pre1 != 0):
          print("left")
          pre1 = 0
          mouse.release(Key.right)
          mouse.press(Key.left)
        elif angle > 10 and (pre1 != 1):
          print('right') 
          pre1 = 1
          mouse.release(Key.left)
          mouse.press(Key.right)
        elif -10<=angle<=10:
            mouse.release(Key.left)
            mouse.release(Key.right)
            pre1 = -1

        boost = 0
        if (thumb[3]-thumb[4]>20) and (thumb[4]-thumb[5]>20): 
          boost+=1
        if (thumb[0]-thumb[1]>20) and (thumb[1]-thumb[2]>20):
          boost+=1
        if boost==0 and pre2!=0:
          print('brake')
          mouse.release(Key.up)
          mouse.press(Key.down)
        elif boost==2 and pre2!=1:
          print('boost')
          mouse.release(Key.down)
          mouse.press(Key.up)
        elif boost==1:
          print('go')
          pre2 = -1
          mouse.release(Key.down)
          mouse.release(Key.up)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()