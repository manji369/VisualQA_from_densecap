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
	# 2195885,
	# 300,
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False))
	embedding_model.add(Dense(
	10
	))
	print(embedding_model.summary())

	image_model1 = Sequential()
	image_model1.add(Convolution3D(1, 10, 3, 300,
            border_mode='valid',
            input_shape=(10, MAX_SEQUENCE_LENGTH, 300, 1),
			activation='relu'))
	image_model1.add(MaxPooling3D((1,2,1), strides =(1,1,1)))
	image_model1.add(Reshape((1, 10)))
	print(image_model1.summary())

	spatial_model = Sequential()
	spatial_model.add(Dense(
	1,
	input_shape=(1, 10, 10)
	))
	spatial_model.add(Reshape((1, 10)))
	print(spatial_model.summary())

	main_model = Sequential()
	main_model.add(Merge(
		[image_model1,embedding_model,spatial_model],
		mode = 'concat',
		concat_axis = 1))
	main_model.add(LSTM(1001))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))

	return main_model

def vis_lstm_10():
	embedding_matrix = embedding.load()
	# print(embedding_matrix.shape)
	embedding_model = Sequential()
	embedding_model.add(Embedding(
	# 2195885,
	# 300,
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False))
	embedding_model.add(Dense(
	10
	))
	print(embedding_model.summary())

	image_model1 = Sequential()
	image_model1.add(Convolution3D(1, 10, 3, 300,
            border_mode='valid',
            input_shape=(10, MAX_SEQUENCE_LENGTH, 300, 1),
			activation='relu'))
	image_model1.add(MaxPooling3D((1,2,1), strides =(1,1,1)))
	image_model1.add(Reshape((1, 10)))
	print(image_model1.summary())

	spatial_model = Sequential()
	spatial_model.add(Dense(
	1,
	input_shape=(1, 10, 10)
	))
	spatial_model.add(Reshape((1, 10)))
	print(spatial_model.summary())

	main_model = Sequential()
	main_model.add(Merge(
		[image_model1,embedding_model,spatial_model],
		mode = 'concat',
		concat_axis = 1))
	main_model.add(LSTM(11))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))

	return main_model

def vis_lstm_100():
	embedding_matrix = embedding.load()
	# print(embedding_matrix.shape)
	embedding_model = Sequential()
	embedding_model.add(Embedding(
	# 2195885,
	# 300,
		embedding_matrix.shape[0],
		embedding_matrix.shape[1],
		weights = [embedding_matrix],
		trainable = False))
	embedding_model.add(Dense(
	10
	))
	print(embedding_model.summary())

	image_model1 = Sequential()
	image_model1.add(Convolution3D(1, 10, 3, 300,
            border_mode='valid',
            input_shape=(10, MAX_SEQUENCE_LENGTH, 300, 1),
			activation='relu'))
	image_model1.add(MaxPooling3D((1,2,1), strides =(1,1,1)))
	image_model1.add(Reshape((1, 10)))
	print(image_model1.summary())

	spatial_model = Sequential()
	spatial_model.add(Dense(
	1,
	input_shape=(1, 10, 10)
	))
	spatial_model.add(Reshape((1, 10)))
	print(spatial_model.summary())

	main_model = Sequential()
	main_model.add(Merge(
		[image_model1,embedding_model,spatial_model],
		mode = 'concat',
		concat_axis = 1))
	main_model.add(LSTM(101))
	main_model.add(Dropout(0.5))
	main_model.add(Dense(1001,activation='softmax'))

	return main_model

if __name__ == '__main__':
	vis_lstm()
