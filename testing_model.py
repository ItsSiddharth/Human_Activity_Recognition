from testing_model_functions import preprocess_video
import tensorflow as tf
import numpy as np

X_test = preprocess_video('/home/ubuntu/Human_activity_recognition/test.mp4')
X_test = np.array(X_test)
model = tf.keras.models.load_model('/home/ubuntu/Human_activity_recognition/INITIAL MODEL 200-64.model')

y = model.predict(X_test)

if y[0][0] > 0.5:
	print("Action is Yoga")
else:
	print("Action is Playing Guitar")