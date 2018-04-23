from nltk import word_tokenize
import json
import pandas as pd
import numpy as np
MAX_LEN = 0
NUM_PARTS = 6

with open('data/questions.json') as f:
    cont = f.read()
questions = json.loads(cont)
with open('data/annotations.json') as f:
    cont = f.read()
answers = json.loads(cont)

mp = {}
for i in range(NUM_PARTS):
    print(i)
    with open('data/results_train_{0}.json'.format(i)) as f:
        cont = f.read()
    results = json.loads(cont)
    for result in results['results']:
        img_id = int(result['img_name'].split('.')[0].split('_')[-1])
        mp[img_id] = []
        mp[img_id].append(result['captions'][:10])
        mp[img_id].append([])
        mp[img_id].append([])
print("Completed reading captions")
for question in questions['questions']:
    img_id, question, question_id = question['image_id'], question['question'], question['question_id']
    if img_id in mp:
        mp[img_id][1].append([question_id, question])
for answer in answers['annotations']:
    img_id, multiple_choice_answer, question_id = answer['image_id'], answer['multiple_choice_answer'], answer['question_id']
    if img_id in mp:
        mp[img_id][2].append([question_id, multiple_choice_answer])

for img_id in mp:
    captions = mp[img_id][0]
    for caption in captions:
        MAX_LEN = max(len(caption.split(' ')), MAX_LEN)
print(MAX_LEN)

embeddings = {}
cnt = 0
with open('embeddings/glove.840B.300d.txt','r') as f:
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

def get3DMatrix(captions, embeddings):
    res = []
    for caption in captions:
        sentence = [np.asarray([0]*300,dtype='float32')]*MAX_LEN
        words = word_tokenize(caption)
        for i, word in enumerate(words):
            try:
                sentence[i] = embeddings.get(word, np.asarray([0]*300,dtype='float32'))
            except:
                print((i, MAX_LEN))
                return
        res.append(sentence)
    return np.asarray(res)

matrix_map = {}
for image_id in mp:
    captions = mp[image_id][0]
    matrix_map[image_id] = get3DMatrix(captions, embeddings)
print("Completed creating 3D matrices")

data = {'image_id': [], 'captions': [], 'questions': [], 'answers': [], 'question_ids': [], 'caption_matrix': []}
for image_id in mp:
    captions, questions, answers = mp[image_id]
    question_answers = []
    questionsMod = {}
    answersMod = {}
    for q_id, q in questions:
        questionsMod[q_id] = q
    for q_id, a in answers:
        answersMod[q_id] = a
    for question_id in questionsMod:
        question_answers.append((question_id, questionsMod[question_id], answersMod[question_id]))
    for question_id, question, answer in question_answers:
        data['image_id'].append(image_id)
        data['captions'].append(captions)
        data['questions'].append(question)
        data['answers'].append(answer)
        data['question_ids'].append(question_id)
        data['caption_matrix'].append(matrix_map[image_id])


df = pd.DataFrame(data=data)
print(df.head())
df.to_pickle('train.pkl')
