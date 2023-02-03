from dependencies import *
from image_preprocess import image_preprocessing1
from one_hot_encode import one_hot_encode


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

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(128, (3, 3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)
# 	x=MaxPooling2D((2,2), strides=(2,2))(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(256, (3, 3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(256,(3,3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(256,(3,3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)
# 	x=MaxPooling2D((2,2), strides=(2,2))(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(512, (3, 3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(512, (3, 3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(512,(3,3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)
# 	x=MaxPooling2D((2,2), strides=(2,2))(x)


# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(512, (3, 3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(512, (3, 3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)

# 	x=ZeroPadding2D((1,1))(x)
# 	x=Convolution2D(512,(3,3))(x)
# 	#     x=BatchNormalization(axis=3)(x)
# 	x=Activation("relu")(x)
	x=MaxPooling2D((2,2), strides=(2,2))(x)


	x=Flatten()(x)

# 	x=Dense(1024,activation="relu")(x)
	x=Dense(256,activation="relu")(x)
	x=Dense(4,activation="softmax")(x)

	model_img=Model(inputs=x_input,outputs=x)
	print(x.shape)

	return model_img,x


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
	test_loss, test_acc = model.evaluate(test_image_in, test_label_in)
	print(test_acc)



model_train()




