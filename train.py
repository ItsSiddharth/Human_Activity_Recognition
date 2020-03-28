from functions.training_functions import preprocessing_csv, shuffler, train_LSTM, get_final_data_for_model
from functions.train_datagen_functions import generate_training_data


import numpy as np

# generate_training_data('/home/ubuntu/kinetics-downloader/dataset/train/wrestling/', 'training_data_wrestling')

X_action2_train, X_action2_test =  preprocessing_csv('training_data_wrestling1.csv', 50)

X_action1_train, X_action1_test =  preprocessing_csv('training_data_guitar.csv', 50)


X_train, X_test, Y_train, Y_test = get_final_data_for_model(X_action2_train, X_action2_test, X_action1_train, X_action1_test)

X_train, Y_train = shuffler(X_train, Y_train)
X_test, Y_test = shuffler(X_test, Y_test)

train_LSTM(X_train, Y_train, X_test, Y_test, name_of_the_model)
