import numpy as np
import prepare_data
import models
import argparse
import sys

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-num_epochs', type=int, default=25)
	parser.add_argument('-batch_size', type=int, default=200)
	parser.add_argument('-model', type=int, default=1)
	args = parser.parse_args()

	print('Loading questions ...')
	questions_train = prepare_data.get_questions_matrix('train')
	questions_val = prepare_data.get_questions_matrix('val')
	print('Loading answers ...')
	answers_train = prepare_data.get_answers_matrix('train')
	answers_val = prepare_data.get_answers_matrix('val')
	print('Loading caption matrices ...')
	caption_matrices_train = prepare_data.get_3D_matrices('train')
	caption_matrices_val = prepare_data.get_3D_matrices('val')
	print('Loading spatial matrices ...')
	spatial_matrices_train = prepare_data.get_spatial_matrices('train')
	spatial_matrices_val = prepare_data.get_spatial_matrices('val')
	print('Creating model ...')

	model = models.vis_lstm_100()
	X_train = [caption_matrices_train, questions_train, spatial_matrices_train]
	X_val = [caption_matrices_val, questions_val, spatial_matrices_val]
	model_path = 'weights/model_spat_{0}_{1}_100.h5'.format(args.num_epochs, args.batch_size)

	model.compile(optimizer='adam',
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	model.fit(X_train,answers_train,
		nb_epoch=args.num_epochs,
		batch_size=args.batch_size,
		validation_data=(X_val,answers_val),
		verbose=1)

	model.save(model_path)

if __name__ == '__main__':main()
