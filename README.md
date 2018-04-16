# VisualQA using CNN on densecaptions
This is a python and keras implementation of the idea of using [dense captions](https://arxiv.org/abs/1511.07571) to answer questions.
It uses CNN to process the captions and answer questions.

Details about the dataset are explained at the [VisualQA website](http://www.visualqa.org/).

## Requirements

* Python 2.7
* Numpy
* NLTK (for tokenizer)
* Keras
* Theano

## Training

* The basic usage is `python train.py`.

* The batch size and the number of epochs can also be specified using the options `-num_epochs` and `-batch_size`. The default batch size and number of epochs are 200 and 25 respectively.
