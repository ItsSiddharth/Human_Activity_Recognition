from functions.testing_model_functions import preprocess_video
import tensorflow as tf
import numpy as np

X_test = preprocess_video('/home/ubuntu/Human_activity_recognition/sample_videos_for_inference/test.mp4')
X_test = np.array(X_test)
model = tf.keras.models.load_model('/home/ubuntu/Human_activity_recognition/models/wrestling_vs_guitar.model')

# print(model.summary())

y = model.predict(X_test)
# For MODEL 200-64.model Action1 is Yoga and Action2 is Wrestling.
if y[0][0] > 0.5:
	print("Action is Playing Guitar")
else:
	print("Action is Wrestling")