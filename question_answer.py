import numpy as np
import embedding as ebd
import prepare_data
import models
import argparse
import sys
import keras.backend as K
from nltk import word_tokenize
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.models import load_model

MAX_LEN = 13

def preprocess_question(question):
	word_idx = ebd.load_idx()
	tokens = word_tokenize(question)
	seq = []
	for token in tokens:
		seq.append(word_idx.get(token,0))
	seq = np.reshape(seq,(1,len(seq)))
	return seq

def get_3D_matrix(captions, embeddings):
    res = [[]]
    for caption in captions:
        sentence = [np.asarray([[0]]*300,dtype='float32')]*MAX_LEN
        words = word_tokenize(caption)
        for i, word in enumerate(words):
            sentence[i] = [[x] for x in embeddings.get(word, np.asarray([0]*300,dtype='float32'))]
        res[0].append(sentence)
    res = np.asarray(res)
    return res

def generate_answer(captions, question, model, embeddings):
	model_path = 'weights/model_'+str(model)+'.h5'
	model = models.vis_lstm()
	model.load_weights(model_path)
	captions_matrix = get_3D_matrix(captions.split(','), embeddings)
	seq = preprocess_question(question)
	x = [captions_matrix, seq]
	probabilities = model.predict(x)[0]
	answers = np.argsort(probabilities[:1000])
	top_answers = [prepare_data.top_answers[answers[-1]],
		prepare_data.top_answers[answers[-2]],
		prepare_data.top_answers[answers[-3]]]

	return top_answers

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-captions', type=str, required=True)
	parser.add_argument('-question', type=str, required=True)
	parser.add_argument('-model', type=int, default=1)
	args = parser.parse_args()
	if args.model != 1 and args.model != 2:
		print('Invalid model selection.')
		sys.exit()
	embeddings = prepare_data.load_embeddings()
	top_answers = generate_answer(args.captions, args.question, args.model, embeddings)
	print('Top answers: %s, %s, %s.' % (top_answers[0],top_answers[1],top_answers[2]))
	get_3D_matrix(args.captions.split(','), embeddings)

if __name__ == '__main__':main()
