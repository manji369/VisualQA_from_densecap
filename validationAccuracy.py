import pickle
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_file', type=str, default='test_no_spat.h5')
    args = parser.parse_args()
    objects = []
    with (open("test_files/"+args.test_file, "rb")) as openfile:
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
        if ''.join(groundtruthlist[i]).rstrip() in predictedAnswerList[i]:
            correctMatch += 1

    print('The validation accuracy of the model is', (correctMatch / len(objects[0]["answers"])) * 100)

if __name__ == '__main__':main()
