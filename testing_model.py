import os
import subprocess
import cv2
import numpy as np
import time
import pandas as pd
import os
import collections
import subprocess
from train_datagen import video_sampler, list_of_dictionaries_of_keypoints

import tensorflow as tf

def generate_training_data(path_of_folder):
	no_of_videos_skipped = 0
	videos = [video for video in os.listdir(path_of_folder) if video.endswith('.mp4')]
	print(videos)
	# videos = videos[:300]
	print(len(videos))
	for video in videos:
		duration = subprocess.check_output(['ffprobe', '-i', '{}'.format(os.path.join(path_of_folder,video)), '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")])
		if int(float((str(duration)[2:-3]))) != 10:
			# print(video)
			list_of_sampled_keypoints_for_single_video, no_of_frames_extracted=video_sampler(os.path.join(path_of_folder,video))
			if no_of_frames_extracted < 9:
				print("Imprefection in extraction ===>>> {} ===>>> {}".format(no_of_frames_extracted, video))
				no_of_videos_skipped = no_of_videos_skipped + 1
			elif no_of_frames_extracted >=9:
				list_of_sampled_keypoints_for_single_video = list_of_sampled_keypoints_for_single_video[:9]
				sampled_frames_of_video = list_of_dictionaries_of_keypoints(list_of_sampled_keypoints_for_single_video)
				for frame in sampled_frames_of_video:
					# print(element)
					df = pd.DataFrame(dict(frame))
					df = df.T
					df.to_csv('testing_random1.csv', mode='a', header=False)

# video_sampler('/home/ubuntu/Human_activity_recognition/test_vid/test.mp4')

# generate_training_data('/home/ubuntu/Human_activity_recognition/test_vid/')



concatinated_frames = []
intermediate_list = []
X_test = []
df_test = pd.read_csv('testing_random1.csv')
print(df_test['x'].head(17))
x_cords = list(df_test['x'])
y_cords = list(df_test['y'])
print(len(x_cords)) # ==> 64107
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

print(X_test)

X_test = np.array(X_test)
print(X_test.shape)

model = tf.keras.models.load_model('/home/ubuntu/Human_activity_recognition/INITIAL MODEL 200-64.model')

y = model.predict(X_test)

print(y)