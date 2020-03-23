from infering_tflite import Key_Point_Generator

import cv2
import numpy as np

cap = cv2.VideoCapture('test_video.mp4')

while True:
	_, frame = cap.read()
	frame = np.array(frame, dtype=np.uint8)
	image, map_cords_to_part = Key_Point_Generator(frame)
	print('No of body parts detetcted : {}'.format(len(map_cords_to_part)))
	for element in map_cords_to_part:
		print(element[0],element[1], element[2])
	print('#################')
	cv2.imshow('Preview', image)
	k = cv2.waitKey(27)
	if k == ord('q'):
		break