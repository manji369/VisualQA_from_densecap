import numpy as np
import pandas as pd
import embedding as ebd
import operator
import sys
import scipy as sc
from collections import defaultdict
from nltk import word_tokenize
from keras.preprocessing.sequence import pad_sequences

TRAIN_DATA_PATH = 'data/train_spat.pkl'
VAL_DATA_PATH = 'data/val_spat.pkl'
MAX_LEN = 13

def int_to_answers():
	data_path = TRAIN_DATA_PATH
	df = pd.read_pickle(data_path)
	answers = df[['answers']].values.tolist()
	freq = defaultdict(int)
	for answer in answers:
		freq[answer[0].lower()] += 1
	int_to_answer = sorted(freq.items(),key=operator.itemgetter(1),reverse=True)[0:1000]
	int_to_answer = [answer[0] for answer in int_to_answer]
	return int_to_answer

top_answers = int_to_answers()

def answers_to_onehot():
	top_answers = int_to_answers()
	answer_to_onehot = {}
	for i, word in enumerate(top_answers):
		onehot = np.zeros(1001)
		onehot[i] = 1.0
		answer_to_onehot[word] = onehot
	return answer_to_onehot

answer_to_onehot_dict = answers_to_onehot()

def get_answers_matrix(split):
	if split == 'train':
		data_path = TRAIN_DATA_PATH
	elif split == 'val':
		data_path = VAL_DATA_PATH
	else:
		print('Invalid split!')
		sys.exit()

	df = pd.read_pickle(data_path)
	answers = df[['answers']].values.tolist()
	answer_matrix = np.zeros((len(answers),1001))
	default_onehot = np.zeros(1001)
	default_onehot[1000] = 1.0

	for i, answer in enumerate(answers):
		answer_matrix[i] = answer_to_onehot_dict.get(answer[0].lower(),default_onehot)

	return answer_matrix

def get_questions_matrix(split):
	if split == 'train':
		data_path = TRAIN_DATA_PATH
	elif split == 'val':
		data_path = VAL_DATA_PATH
	else:
		print('Invalid split!')
		sys.exit()

	df = pd.read_pickle(data_path)
	questions = df[['questions']].values.tolist()
	word_idx = ebd.load_idx()
	seq_list = []

	for question in questions:
		words = word_tokenize(question[0])
		seq = []
		for word in words:
			seq.append(word_idx.get(word,0))
		seq_list.append(seq)
	question_matrix = pad_sequences(seq_list)

	return question_matrix

def get_3D_matrices(split):
	if split == 'train':
		data_path = TRAIN_DATA_PATH
	elif split == 'val':
		data_path = VAL_DATA_PATH
	else:
		print('Invalid split!')
		sys.exit()
	df = pd.read_pickle(data_path)
	matrices = np.asarray(df[['caption_matrix']].values.tolist())
	print(matrices.shape)
	matrices = matrices.swapaxes(1,2)
	matrices = matrices.swapaxes(2,3)
	matrices = matrices.swapaxes(3,4)
	print(matrices.shape)
	return matrices


def load_embeddings():
	embeddings = {}
	cnt = 0
	with open('./embeddings/glove.840B.300d.txt','r') as f:
	    for i, line in enumerate(f):
	        values = line.split()
	        word = values[0]
	        try:
	            coefs = np.asarray(values[1:],dtype='float32')
	        except Exception as ex:
	            cnt += 1
	            print("ex:{}/i:{}".format(cnt, i))
	            continue
	        embeddings[word] = coefs
	return embeddings


def get_3D_matrix(captions, embeddings):
    res = []
    for caption in captions:
        sentence = [np.asarray([0]*300,dtype='float32')]*MAX_LEN
        words = word_tokenize(caption)
        for i, word in enumerate(words):
            sentence[i] = embeddings.get(word, np.asarray([0]*300,dtype='float32'))
        res.append(sentence)
    return res


def get_spatial_matrices(split):
	if split == 'train':
		data_path = TRAIN_DATA_PATH
	elif split == 'val':
		data_path = VAL_DATA_PATH
	else:
		print('Invalid split!')
		sys.exit()
	df = pd.read_pickle(data_path)
	matrices = np.asarray(df[['spatial_matrix']].values.tolist())
	print(matrices.shape)
	return matrices


if __name__ == '__main__':
	get_3D_matrices('train')
