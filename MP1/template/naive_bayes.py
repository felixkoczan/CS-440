# naive_bayes.py
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


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace, pos_prior)
    pos_probs, neg_probs, vocabulary, pos_total, neg_total = train_naive_bayes(train_set, train_labels, laplace)
    yhats = predict_naive_bayes(dev_set, pos_probs, neg_probs, vocabulary, pos_prior)
    return yhats


def train_naive_bayes(train_set, train_labels, laplace=1.0):
    pos_words = Counter()
    neg_words = Counter()
    pos_total = 0
    neg_total = 0
    
    for idx, doc in enumerate(train_set):
        if train_labels[idx] == 1:
            pos_words.update(doc)
            pos_total += len(doc)
        else:
            neg_words.update(doc)
            neg_total += len(doc)

    vocabulary = set(pos_words.keys()).union(set(neg_words.keys()))

    pos_probs = {word: (pos_words[word] + laplace) / (pos_total + laplace * len(vocabulary)) for word in vocabulary}
    neg_probs = {word: (neg_words[word] + laplace) / (neg_total + laplace * len(vocabulary)) for word in vocabulary}

    return pos_probs, neg_probs, vocabulary, pos_total, neg_total


def predict_naive_bayes(dev_set, pos_probs, neg_probs, vocabulary, pos_prior=0.5):
    yhats = []
    
    for doc in tqdm(dev_set):
        log_prob_pos = math.log(pos_prior)
        log_prob_neg = math.log(1 - pos_prior)

        for word in doc:
            if word in vocabulary:  
                log_prob_pos += math.log(pos_probs.get(word, 1.0 / (len(vocabulary) + 1)))
                log_prob_neg += math.log(neg_probs.get(word, 1.0 / (len(vocabulary) + 1)))

        if log_prob_pos > log_prob_neg:
            yhats.append(1)
        else:
            yhats.append(0)  
    
    return yhats
