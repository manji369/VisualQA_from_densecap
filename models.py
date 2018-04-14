import numpy as np
import embedding
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Merge, Reshape, Dropout, Convolution2D, MaxPooling2D, ZeroPadding2D, Flatten, Convolution3D, MaxPooling3D
MAX_SEQUENCE_LENGTH = 13

def vis_lstm():
	embedding_matrix = embedding.load()
	# print(embedding_matrix.shape)
	embedding_model = Sequential()
	embedding_model.add(Embedding(
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False))
	embedding_model.add(Dense(
	10
	))
	print(embedding_model.summary())

	#image_model = Sequential()
	#image_model.add(Dense(
	#	embedding_matrix.shape[1],
	#	input_dim=4096,
	#	activation='linear'))
	# image_model.add(Reshape((1,embedding_matrix.shape[1])))

	image_model1 = Sequential()
	image_model1.add(Convolution3D(1, 10, 3, 300,
            border_mode='valid',
            input_shape=(10, MAX_SEQUENCE_LENGTH, 300, 1),
			activation='relu'))
	image_model1.add(MaxPooling3D((1,2,1), strides =(1,1,1)))
	image_model1.add(Reshape((1, 10)))
	print(image_model1.summary())


	main_model = Sequential()
	main_model.add(Merge(
		[image_model1,embedding_model],
		mode = 'concat',
		concat_axis = 1))
	main_model.add(LSTM(1001))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))

	return main_model

def vis_lstm_2():
	embedding_matrix = embedding.load()
	embedding_model = Sequential()
	embedding_model.add(Embedding(
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False))

	image_model_1 = Sequential()
	image_model_1.add(Dense(
		embedding_matrix.shape[1],
		input_dim=4096,
		activation='linear'))
	image_model_1.add(Reshape((1,embedding_matrix.shape[1])))

	image_model_2 = Sequential()
	image_model_2.add(Dense(
		embedding_matrix.shape[1],
		input_dim=4096,
		activation='linear'))
	image_model_2.add(Reshape((1,embedding_matrix.shape[1])))

	main_model = Sequential()
	main_model.add(Merge(
		[image_model_1,embedding_model,image_model_2],
		mode = 'concat',
		concat_axis = 1))
	main_model.add(LSTM(1001))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))

	return main_model

def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides =(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides =(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides =(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides =(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides =(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

if __name__ == '__main__':
	vis_lstm()
