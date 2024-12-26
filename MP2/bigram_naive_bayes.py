# bigram_naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

def preprocess_text(doc, stop_words):
    return [word for word in doc if word.lower() not in stop_words]

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    train_set = [preprocess_text(doc, stop_words) for doc in train_set]
    dev_set = [preprocess_text(doc, stop_words) for doc in dev_set]
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.85, bigram_laplace=0.2, bigram_lambda=0.6, pos_prior=0.8, silently=False):

    print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior)

    pos_unigram_probs, neg_unigram_probs, pos_bigram_probs, neg_bigram_probs, unigram_vocabulary, bigram_vocabulary = train_naive_bayes(
        train_set, train_labels, unigram_laplace=unigram_laplace, bigram_laplace=bigram_laplace
    )

    yhats = predict_naive_bayes(
        dev_set, pos_unigram_probs, neg_unigram_probs, pos_bigram_probs, neg_bigram_probs, 
        unigram_vocabulary, bigram_vocabulary, pos_prior=pos_prior, bigram_lambda=bigram_lambda
    )

    return yhats


def train_naive_bayes(train_set, train_labels, unigram_laplace, bigram_laplace):
    pos_unigrams = Counter()
    neg_unigrams = Counter()
    pos_bigrams = Counter()
    neg_bigrams = Counter()
    pos_total_unigrams = 0
    neg_total_unigrams = 0
    pos_total_bigrams = 0
    neg_total_bigrams = 0

    def get_bigrams(doc):
        return [(doc[i], doc[i+1]) for i in range(len(doc) - 1)]

    for idx, doc in enumerate(train_set):
        if train_labels[idx] == 1:  
            pos_unigrams.update(doc)
            pos_bigrams.update(get_bigrams(doc))
            pos_total_unigrams += len(doc)
            pos_total_bigrams += len(doc) - 1
        else:  
            neg_unigrams.update(doc)
            neg_bigrams.update(get_bigrams(doc))
            neg_total_unigrams += len(doc)
            neg_total_bigrams += len(doc) - 1

    unigram_vocabulary = set(pos_unigrams.keys()).union(set(neg_unigrams.keys()))
    bigram_vocabulary = set(pos_bigrams.keys()).union(set(neg_bigrams.keys()))

    pos_unigram_probs = {word: (pos_unigrams[word] + unigram_laplace) / (pos_total_unigrams + unigram_laplace * len(unigram_vocabulary)) for word in unigram_vocabulary}
    neg_unigram_probs = {word: (neg_unigrams[word] + unigram_laplace) / (neg_total_unigrams + unigram_laplace * len(unigram_vocabulary)) for word in unigram_vocabulary}

    pos_bigram_probs = {bigram: (pos_bigrams[bigram] + bigram_laplace) / (pos_total_bigrams + bigram_laplace * len(bigram_vocabulary)) for bigram in bigram_vocabulary}
    neg_bigram_probs = {bigram: (neg_bigrams[bigram] + bigram_laplace) / (neg_total_bigrams + bigram_laplace * len(bigram_vocabulary)) for bigram in bigram_vocabulary}

    return pos_unigram_probs, neg_unigram_probs, pos_bigram_probs, neg_bigram_probs, unigram_vocabulary, bigram_vocabulary


def predict_naive_bayes(dev_set, pos_unigram_probs, neg_unigram_probs, pos_bigram_probs, neg_bigram_probs, unigram_vocabulary, bigram_vocabulary, pos_prior, bigram_lambda):
    yhats = []
    
    def get_bigrams(doc):
        return [(doc[i], doc[i+1]) for i in range(len(doc) - 1)]

    for doc in tqdm(dev_set):
        log_prob_pos = math.log(pos_prior)
        log_prob_neg = math.log(1 - pos_prior)

        for word in doc:
            if word in unigram_vocabulary:
                log_prob_pos += math.log(pos_unigram_probs.get(word, 1.0 / (len(unigram_vocabulary) + 1)))
                log_prob_neg += math.log(neg_unigram_probs.get(word, 1.0 / (len(unigram_vocabulary) + 1)))

        bigrams = get_bigrams(doc)
        for bigram in bigrams:
            if bigram in bigram_vocabulary:
                log_prob_pos += bigram_lambda * math.log(pos_bigram_probs.get(bigram, 1.0 / (len(bigram_vocabulary) + 1)))
                log_prob_neg += bigram_lambda * math.log(neg_bigram_probs.get(bigram, 1.0 / (len(bigram_vocabulary) + 1)))

        if log_prob_pos > log_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)
    
    return yhats




