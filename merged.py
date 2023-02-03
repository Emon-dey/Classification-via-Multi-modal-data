from dependencies import *
from image_preprocess import image_preprocessing1
from one_hot_encode import one_hot_encode

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# from tensorflow.keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,LSTM,Activation,Dropout,BatchNormalization,Input,Embedding,Reshape,Concatenate, GRU
from keras.models import Sequential,Model

DATA_PATH = "data.json"
SAVED_MODEL_PATH = "model.h5"
EPOCHS = 40
BATCH_SIZE = 32
PATIENCE = 5
LEARNING_RATE = 0.0001


df_train_data=pd.read_csv("C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\train_data.csv")
df_test_data=pd.read_csv("C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\test_data.csv")

def vgg_model():

	x_input=Input((224,224,3))
	x=ZeroPadding2D((1,1))(x_input)
	x=Convolution2D(64, (3, 3))(x)
	#     x=BatchNormalization(axis=3)(x)
	x=Activation("relu")(x)

	x=ZeroPadding2D((1,1))(x)
	x=Convolution2D(64,(3,3))(x)
	#     x=BatchNormalization(axis=3)(x)
	x=Activation("relu")(x)
	x=MaxPooling2D((2,2), strides=(2,2))(x)

	x=ZeroPadding2D((1,1))(x)
	x=Convolution2D(128, (3, 3))(x)
	#     x=BatchNormalization(axis=3)(x)
	x=Activation("relu")(x)


	x=MaxPooling2D((2,2), strides=(2,2))(x)


	x=Flatten()(x)

# 	x=Dense(1024,activation="relu")(x)
	x=Dense(256,activation="relu")(x)
	x=Dense(256,activation="softmax")(x)

	model_img=Model(inputs=x_input,outputs=x)
	print(x_input.shape)

	return model_img, x


def model_train():

	model,x=vgg_model()
	train_image_in=image_preprocessing1("C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\helicopter","C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\tank","C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\bomb","C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\gun",df_train_data)/255
	test_image_in=image_preprocessing1("C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\helicopter","C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\tank","C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\bomb","C:\\Users\\povi1\\Downloads\\ARL\\Dataset\\gun",df_test_data)/255

	train_label_in=one_hot_encode(list(df_train_data["Label"]),list(df_train_data["Label"]))
	test_label_in=one_hot_encode(list(df_test_data["Label"]),list(df_test_data["Label"]))


	model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=["accuracy"])

#################################### VGG 16 & LSTM ###########################################################
	callback = [EarlyStopping(monitor='val_loss',mode="min",verbose=1, patience=40),
             ModelCheckpoint('C:\\Users\\povi1\\Downloads\\ARL\\train_{}.h5'.format("vgg"), monitor='val_loss',mode="min" ,verbose=1,save_best_only=True)]
	fit=model.fit(train_image_in,train_label_in, validation_split=0.2, epochs=4,batch_size=32,verbose=1, callbacks=callback)

# 	test_predict=model.predict(test_image_in)
# 	print(test_predict)
# 	test_pred_pos=np.argmax(test_predict,axis=1)
# 	print("pos",test_pred_pos)
# 	test_accu=model.evaluate(test_image_in)
# 	print(test_accu)
# 	test_loss, test_acc = model.evaluate(test_image_in, test_label_in)
# 	print(test_acc)



# model_train()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:



def load_data(data_path):
    """Loads training dataset from json file.
    :param data_path (str): Path to json file containing data
    :return X (ndarray): Inputs
    :return y (ndarray): Targets
    """
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    print("Training sets loaded!")
    return X, y


def prepare_dataset(data_path, test_size=0.2, validation_size=0.2):
    """Creates train, validation and test sets.
    :param data_path (str): Path to json file containing data
    :param test_size (flaot): Percentage of dataset used for testing
    :param validation_size (float): Percentage of train set used for cross-validation
    :return X_train (ndarray): Inputs for the train set
    :return y_train (ndarray): Targets for the train set
    :return X_validation (ndarray): Inputs for the validation set
    :return y_validation (ndarray): Targets for the validation set
    :return X_test (ndarray): Inputs for the test set
    :return X_test (ndarray): Targets for the test set
    """

    # load dataset
    X, y = load_data(data_path)

    # create train, validation, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to nd array
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test



def build_model_test(input_shape, loss="sparse_categorical_crossentropy", learning_rate=0.0001):
    """Build neural network using keras.
    :param input_shape (tuple): Shape of array representing a sample train. E.g.: (44, 13, 1)
    :param loss (str): Loss function to use
    :param learning_rate (float):
    :return model: TensorFlow model
    """

    # build network architecture using convolutional layers
#     model = tf.keras.models.Sequential()
    input_shape=Input(shape=input_shape)
    x=ZeroPadding2D((1,1))(input_shape)
    x=Convolution2D(64, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation("relu")(x)

    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(64,(3,3))(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation("relu")(x)
    x=MaxPooling2D((2,2), strides=(2,2))(x)

    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(128, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=Activation("relu")(x)

    x=MaxPooling2D((2,2), strides=(2,2))(x)


    x=Flatten()(x)

#     x=Dense(1024,activation="relu")(x)
    x=Dense(256,activation="relu")(x)
    x=Dense(128,activation="softmax")(x)


    model=Model(inputs=input_shape,outputs=x)
    print(input_shape.shape)

#     optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer='SGD',
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model,x


def train(model, epochs, batch_size, patience, X_train, y_train, X_validation, y_validation):
    """Trains model
    :param epochs (int): Num training epochs
    :param batch_size (int): Samples per batch
    :param patience (int): Num epochs to wait before early stop, if there isn't an improvement on accuracy
    :param X_train (ndarray): Inputs for the train set
    :param y_train (ndarray): Targets for the train set
    :param X_validation (ndarray): Inputs for the validation set
    :param y_validation (ndarray): Targets for the validation set
    :return history: Training history
    """

#     earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor="accuracy", min_delta=0.001, patience=patience)

    # train model
    history = model.fit(X_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_validation, y_validation))
#                         callbacks=[earlystop_callback])
    return history




def main():
    # generate train, validation and test sets
    X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)

    # create network
    input_shape = (X_train.shape[1], X_train.shape[2],1)

#     model,aa = build_model(input_shape, learning_rate=LEARNING_RATE)
    model,x = build_model_test(input_shape, learning_rate=LEARNING_RATE)
    print(model.summary)
    print(x.shape)
    model1,xx=vgg_model()
    print(model1.summary)
    print(xx.shape)

    # train network
#     history = train(model, EPOCHS, BATCH_SIZE, PATIENCE, X_train, y_train, X_validation, y_validation)

    # plot accuracy/loss for training/validation set as a function of the epochs
#     plot_history(history)

    # evaluate network on test set
#     test_loss, test_acc = model.evaluate(X_test, y_test)
#     print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))

    # save model
    model.save(SAVED_MODEL_PATH)


if __name__ == "__main__":
    main()

