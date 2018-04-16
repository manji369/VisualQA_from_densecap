**CSE 576 Natural Language Processing – Phase 2**

**Visual Question Answering**

Manjith Chakravarthy, Ravi Teja Pinnaka, Sai Veena Katta

**Motivation:**

Object referring is important in visual question answering, as well as human robot interaction. We use semantically embedded references all the time, and inorder to be better able to interact with humans, the systems we use must understand how we communicate complex ideas like references.

**Architecture :**

![image_0](https://user-images.githubusercontent.com/16779567/38787733-b1db5bda-40e4-11e8-921a-f085a822c764.png)

				Fig.1. Block diagram

**Deep Captioning and Vector Representation :**

As shown in the diagram above, the first step is to generate the image captions using Deep Captioning techniques for all the training images. We have used one of the Deep Captioning open source library to generate the captions. In order to represent the captions in the vector notation, we have used GloVe: Global Vectors for Word Representation, an unsupervised learning algorithm for obtaining vector representations for words. This library uses vast global occurences of words in various corpus from multiple texts, news articles and various internet sources and compiles the vocabulary required.

**Model :**

![image_1](https://user-images.githubusercontent.com/16779567/38787734-b1f2d116-40e4-11e8-9889-1f4e5f23b95c.png)

**			**Fig.2 Convolution Neural Network Architecture

The top 10 captions from the image are represented as a set of 10 2D matrices, stacked to form a 3D matrix with dimensions as 10 x 300 x maximum_sequence_length , where the maximum_sequence_length is the maximum possible length of any caption in the dataset.

![image_2](https://user-images.githubusercontent.com/16779567/38787735-b207839a-40e4-11e8-8bdd-c51669313183.png)


			Fig.3. Neural Network Model Architecture

An LSTM is used to predict the output, based on the inputs, one from the convolutional neural network and the other from the word2vec representation of the input question. The output of the LSTM is then fed to a drop out layer, which is later fed to a dense layer to generate the top 3 outputs for the input question.

**Prediction :**

<table>
  <tr>
    <td>S.No</td>
    <td>Image</td>
    <td>Question</td>
    <td>Predicted Answers</td>
    <td>Actual answer</td>
  </tr>
  <tr>
    <td>1</td>
    <td></td>
    <td>Is this the best sunset picture you've ever seen? </td>
    <td>no, yes, yes</td>
    <td>no</td>
  </tr>
  <tr>
    <td>2</td>
    <td></td>
    <td>What are the people caring in their arms? 	</td>
    <td>cake, down, 9</td>
    <td>Surfboards</td>
  </tr>
  <tr>
    <td>3</td>
    <td></td>
    <td>Was the photographer looking down at the red object?</td>
    <td>yes, no, yes</td>
    <td>yes</td>
  </tr>
  <tr>
    <td>4</td>
    <td></td>
    <td>What color is the carpet?</td>
    <td>brown, gray, red</td>
    <td>gray</td>
  </tr>
  <tr>
    <td>5</td>
    <td></td>
    <td>What is next to the toilet paper?</td>
    <td>soup, water, phone</td>
    <td>brush</td>
  </tr>
</table>


**Training and testing parameters :**

The training dataset has been taken from the MSCOCO folder available openly at [cocodataset.org](http://cocodataset.org/)

<table>
  <tr>
    <td>Training Data Size</td>
    <td>76000</td>
  </tr>
  <tr>
    <td>Test Data Size</td>
    <td>20000</td>
  </tr>
  <tr>
    <td>Number of Epochs</td>
    <td>25</td>
  </tr>
  <tr>
    <td>Batch Size</td>
    <td>200</td>
  </tr>
  <tr>
    <td>Validation Accuracy</td>
    <td>52.593 %</td>
  </tr>
</table>


**Accuracy Calculation :**

The model predicts 3 answers for each question input. The predicted answers are tested with the ground truth and if any of the answers are matching exactly with the ground truth, then that record is considered to be True - Positive.

The test data questions and answers with corresponding image ids has been saved in the test.pkl file in the folder. To compute the accuracy, run the validationAccuracy.py with Python3

python3 validationAccuracy.py

The validation accuracy is now close to 52.593 %.

**Requirements:**

* Python 3

* Numpy

* NLTK (for tokenizer)

* Keras

**Conclusion :**

In the next phase, we plan to improve the validation accuracy by tweaking the filter size of the Convolutional Neural Networks, increasing the no of epochs, batch size and by comparing the similarity of the vector notation of the predicted output and the ground truth.

