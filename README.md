# Fine-Tuning-BERT-Model-on-A-Custom-Dataset

# Introduction
Bidirectional Encoder Representations from Transformers or commonly known as BERT is a transformer based model that is used in Natural Language Processing for performing Semi-Supervised and Supervised Learning tasks. The BERT model is different from Transformers in such that it is made up of a stacked layer of Encoders. In this project, we will be looking at Supervised Learning where the a pre-trained BERT model is fine tuned on a movie reviews dataset that was downloaded from Kaggle.

# More About BERT
BERT comes in two types and  they are BERT Base and BERT Large. The BERT Base has 12 Encoders, 768 Feed Forward Networks and 12 Attention Heads whereas the BERT Large has 24 Encoders, 1024 feed forward networks and 16 Attention Heads. Since this is a classification model, the model's input is a CLS token. It takes in a sequence of words, applies self attention and the result is sent to the feed forward network which is later passed on to the next encoder.
In the output stage, at each position a vector of size either 768 or 1024 is outputted. For this task, we take just the CLS token which is then used as an input to a classifier model which later outputs the sentiment of the sentence.
In BERT, word embeddings are done using ELMO which provides embedding based on the context of the word. ELMO is basically a bi-directional LSTM which is trained on a task to create embeddings.

# Project Workflow
- Data was downloaded from Kaggle.
- The Huggingface Transformer library is installed.
- Separating the data into train and test sets.
- Next up, tokenize all the sentences.
- Convert all of them to Tensors as we will be using TensorFlow library.
- Train the model
- The model showed quite a high accuracy despite being trained for a few epochs. It  can be  trained for even more epochs and earlystopping can also be added to stop overfitting.

# Tools/Libraries Used:
- Google Colab
- Jupyter Notebook
- Huggingface
- TensorFlow

# References and Acknowledgements
- Jay Alammar's Blog about BERT which helped me understand the concepts properly.
- Krish Naik's tutorial on YouTube on how to implement a pre-trained BERT model.
