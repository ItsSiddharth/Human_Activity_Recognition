from infering_tflite import Key_Point_Generator

import cv2
import numpy as np
import time
import pandas as pd
import os
import collections

list_of_parts = ['NOSE','LEFT_EYE','RIGHT_EYE','LEFT_EAR','RIGHT_EAR','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST','LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE']
list_of_indexes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q']
def video_sampler(path_to_video):
	no_of_frames = 0
	list_of_sampled_keypoints = []
	cap = cv2.VideoCapture(path_to_video)
	while True:
		try:
			_, frame = cap.read()
			if no_of_frames%30 == 0:
				frame = np.array(frame, dtype=np.uint8)
				image, map_cords_to_part, heatmap = Key_Point_Generator(frame)
				list_of_sampled_keypoints.append(map_cords_to_part)
				# x, y, part
				# You can visualise how a certain part is being observed by visualising the heat map.
				# Just cv2.imshow('<frame_name>', heatmap[index_of_body_part])
				no_of_frames = no_of_frames + 1
				cv2.imshow('Preview', image)
				k = cv2.waitKey(27)
				if k == ord('q'):
					break
			else:
				no_of_frames = no_of_frames + 1
				continue
		except:
			break

	return list_of_sampled_keypoints
def list_of_dictionaries_of_keypoints(list_of_sampled_keypoints):
	map_of_points_to_part={}
	list_of_maps = []
	for element in list_of_sampled_keypoints:
		for unit in element:
			map_of_points_to_part['{}'.format(list_of_indexes[list_of_parts.index(unit[2])])] = [unit[0],unit[1]]
		before_preprocessing=list(map_of_points_to_part.keys())
		for element in list_of_indexes:
			if element not in before_preprocessing:
				map_of_points_to_part['{}'.format(element)] = [0,0]
		list_of_maps.append(map_of_points_to_part)
		# print(map_of_points_to_part)
	return list_of_maps
# example usage of functions:
'''list_of_sampled_keypoints=video_sampler('test_video.mp4')
training_data = list_of_dictionaries_of_keypoints(list_of_sampled_keypoints)
for element in training_data:
	print(list(element.keys()))'''
# training_data is a list of dictionaries which has the coordinates of all points in 
def sort_dict(dictionary):
	list_of_keys = []
	new_dict = []
	for key in dictionary.keys():
		list_of_keys.append(int(key))
	list_of_keys = list_of_keys.sort()
	for element in list_of_keys:
		new_dict['{}'.format(element)] = dictionary['{}'.format(element)]
	return new_dict


def generate_training_data(path_of_folder):
	videos = [video for video in os.listdir(path_of_folder) if video.endswith('.mp4')]
	print(len(videos))
	for video in videos:
		list_of_sampled_keypoints=video_sampler(video)
		training_data = list_of_dictionaries_of_keypoints(list_of_sampled_keypoints)
		for element in training_data:
			print(element)
			df = pd.DataFrame(dict(element))
			df = df.T
			df.to_csv('training_data.csv', mode='a', header=False)

generate_training_data('/home/ubuntu/Human_activity_recognition/mock_dataset/')







