import os
import subprocess
import cv2
import numpy as np
import time
import pandas as pd
import os
import collections
import subprocess
from functions.train_datagen_functions import video_sampler, list_of_dictionaries_of_keypoints

import tensorflow as tf

def preprocess_video(path_to_video):
	no_of_videos_skipped = 0
	list_of_sampled_keypoints_for_single_video, no_of_frames_extracted=video_sampler(path_to_video)
	if no_of_frames_extracted < 9:
		print("ERROR: Video too Short -> No.no_of_frames_extracted = {} ".format(no_of_frames_extracted))
		no_of_videos_skipped = no_of_videos_skipped + 1
	elif no_of_frames_extracted >=9:
		list_of_sampled_keypoints_for_single_video = list_of_sampled_keypoints_for_single_video[:9]
		sampled_frames_of_video = list_of_dictionaries_of_keypoints(list_of_sampled_keypoints_for_single_video)
		for frame in sampled_frames_of_video:
			df = pd.DataFrame(dict(frame))
			df = df.T
			df.to_csv('testing_random1.csv', mode='a', header=False)

	concatinated_frames = []
	intermediate_list = []
	X_test = []

	df_test = pd.read_csv('testing_random1.csv', header=None)
	df_test.columns = ["keypoints", "x", "y"]
	os.remove("testing_random1.csv")

	x_cords = list(df_test['x'])
	y_cords = list(df_test['y'])
	x_cords = x_cords + x_cords
	y_cords = y_cords + y_cords
	print(len(x_cords)) # ==> 153
	for i in range(0, len(x_cords)):
	  if i%17 ==0 and i != 0:
	    concatinated_frames.append(intermediate_list)
	    intermediate_list = []
	    intermediate_list.append(x_cords[i])
	    intermediate_list.append(y_cords[i])
	  else :
	    intermediate_list.append(x_cords[i])
	    intermediate_list.append(y_cords[i])
	print(len(concatinated_frames))

	intermediate_list = []
	for i in range(0, len(concatinated_frames)):
	  if i%9 ==0 and i != 0:
	    X_test.append(intermediate_list)
	    intermediate_list = []
	    intermediate_list.append(concatinated_frames[i])
	  else :
	    intermediate_list.append(concatinated_frames[i])
	return X_test
