from nltk import word_tokenize
import json
import pandas as pd
import numpy as np
MAX_LEN = 13
NUM_PARTS = 6

# split = 'train'
split = 'val'

if split == 'val':
    q_path = 'data/questions_val.json'
    a_path = 'data/annotations_val.json'
else:
    q_path = 'data/questions.json'
    a_path = 'data/annotations.json'
with open(q_path) as f:
    cont = f.read()
questions = json.loads(cont)
with open(a_path) as f:
    cont = f.read()
answers = json.loads(cont)

mp = {}
if split == 'train':
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
            mp[img_id].append(result['boxes'][:10])
else:
    with open('data/results_val.json') as f:
        cont = f.read()
    results = json.loads(cont)
    for result in results['results']:
        img_id = int(result['img_name'].split('.')[0].split('_')[-1])
        mp[img_id] = []
        mp[img_id].append(result['captions'][:10])
        mp[img_id].append([])
        mp[img_id].append([])
        mp[img_id].append(result['boxes'][:10])
print("Completed reading captions")
for question in questions['questions']:
    img_id, question, question_id = question['image_id'], question['question'], question['question_id']
    if img_id in mp:
        mp[img_id][1].append([question_id, question])
for answer in answers['annotations']:
    img_id, multiple_choice_answer, question_id = answer['image_id'], answer['multiple_choice_answer'], answer['question_id']
    if img_id in mp:
        mp[img_id][2].append([question_id, multiple_choice_answer])

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
        for i, word in enumerate(words[:MAX_LEN]):
            sentence[i] = embeddings.get(word, np.asarray([0]*300,dtype='float32'))
        res.append(sentence)
    return np.asarray(res)

matrix_map = {}
for image_id in mp:
    captions = mp[image_id][0]
    matrix_map[image_id] = get3DMatrix(captions, embeddings)
print("Completed creating 3D matrices")

def euclideanDist(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

spatial_map = {}
for image_id in mp:
    boxes = mp[image_id][3]
    points = [((box[0]+box[2])/2, (box[1]+box[2])/2) for box in boxes]
    res = np.asarray([np.asarray([0]*10) for _ in range(10)])
    for i in range(10):
        for j in range(10):
            res[i][j] = euclideanDist(points[i], points[j])
    spatial_map[image_id] = res
print("Completed creating spatial map")


data = {'image_id': [], 'captions': [], 'questions': [], 'answers': [], 'question_ids': [], 'caption_matrix': [], 'spatial_matrix': []}
for image_id in mp:
    captions, questions, answers, spat = mp[image_id]
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
        data['spatial_matrix'].append(spatial_map[image_id])


df = pd.DataFrame(data=data)
print(df.head())
if split == 'train':
    df.to_pickle('data/train_spat.pkl')
else:
    df.to_pickle('data/val_spat.pkl')
