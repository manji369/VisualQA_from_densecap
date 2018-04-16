import pickle
objects = []
with (open("test.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

correctMatch = 0
groundtruthlist = objects[0]["answers"]
predictedAnswerList = objects[0]["top_answers"]
questions = objects[0]["questions"]

for i in range(len(groundtruthlist)):
    # if i<50:
    #     print(''.join(groundtruthlist[i]).rstrip())
    #     print(predictedAnswerList[i])
    #     print(questions[i])
    if ''.join(groundtruthlist[i]).rstrip() in predictedAnswerList[i]:
        correctMatch += 1

print('The validation accuracy of the model is', (correctMatch / len(objects[0]["answers"])) * 100)
    