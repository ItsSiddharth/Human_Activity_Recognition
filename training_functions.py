import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import numpy as np
import random
from sklearn.utils import shuffle



def preprocessing_csv(path_to_csv, no_of_samples_in_test):
	concatinated_x_y = []
	concatinated_frames = []
	intermediate_list = []
	X_train = []
	df_x = pd.read_csv(path_to_csv)
	x_cords = list(df_x['x'])
	y_cords = list(df_x['y'])
	for i in range(0, len(x_cords)):
	  if i%17 ==0 and i != 0:
	    concatinated_frames.append(intermediate_list)
	    intermediate_list = []
	    intermediate_list.append(x_cords[i])
	    intermediate_list.append(y_cords[i])
	  else :
	    intermediate_list.append(x_cords[i])
	    intermediate_list.append(y_cords[i])

	intermediate_list = []
	for i in range(0, len(concatinated_frames)):
	  if i%9 ==0 and i != 0:
	    X_train.append(intermediate_list)
	    intermediate_list = []
	    intermediate_list.append(concatinated_frames[i])
	  else :
	    intermediate_list.append(concatinated_frames[i])

	X_test = X_train[-no_of_samples_in_test:]
	X_train = X_train[:-no_of_samples_in_test]
	return X_train, X_test


def get_final_data_for_model(X_action2_train, X_action2_test, X_action1_train, X_action1_test):

	if len(X_action1_train) > len(X_action2_train):
		X_action1_train = X_action1_train[:len(X_action2_train)]
	else:
		X_action2_train = X_action2_train[:len(X_action1_train)]


	Y_action2_train = list(np.zeros((len(X_action2_train))))
	Y_action2_test = list(np.zeros((len(X_action2_test))))

	Y_action1_train = list(np.ones((len(X_action1_train))))
	Y_action1_test = list(np.ones((len(X_action1_test))))



	X_train = X_action2_train + X_action1_train
	Y_train = Y_action2_train + Y_action1_train
	# print(len(X_train), len(Y_train))
	X_test = X_action2_test + X_action1_test
	Y_test = Y_action2_test + Y_action1_test

	return X_train, X_test, Y_train, Y_test


def shuffler(X_train, Y_train):
	X_train, Y_train = shuffle(np.array(X_train), np.array(Y_train))
	return X_train, Y_train





def train_LSTM(X_train, Y_train, X_test, Y_test, NAME):
	model = Sequential()
	model.add(LSTM(128, input_shape=(X_train.shape[1:]), return_sequences=True))
	model.add(Dropout(0.2))
	model.add(BatchNormalization())

	model.add(LSTM(128, return_sequences=True))
	model.add(Dropout(0.1))
	model.add(BatchNormalization())

	model.add(LSTM(128))
	model.add(Dropout(0.1))
	model.add(BatchNormalization())


	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(1, activation='sigmoid'))


	EPOCHS = 200
	BATCH_SIZE = 64 
	opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
	model.compile(
	    loss='binary_crossentropy',
	    optimizer=opt,
	    metrics=['accuracy']
	)
	filepath = "LSTM-{epoch:02d}-{val_acc:.3f}"

	history = model.fit(
	    X_train, Y_train,
	    batch_size=BATCH_SIZE,
	    epochs=EPOCHS,
	    validation_data=(X_test, Y_test),
	)

	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

	model.save("{}.model".format(NAME))