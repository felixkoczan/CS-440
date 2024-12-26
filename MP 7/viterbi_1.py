import math
from collections import defaultdict
from math import log, inf


k = 0.5

def training(sentences):
    """
    Computes initial tags, emission words, and transition tag-to-tag probabilities using Laplace smoothing.
    """
    init_prob = defaultdict(int)  # Initial tag counts
    emit_prob = defaultdict(lambda: defaultdict(int))  # Emission probabilities
    trans_prob = defaultdict(lambda: defaultdict(int))  # Transition probabilities
    tag_count = defaultdict(int)  # Tag counts

    # Collect counts for initial, transition, and emission probabilities
    for sentence in sentences:
        prev_tag = "START"  # Starting state
        for i, (word, tag) in enumerate(sentence):
            if i == 0:
                init_prob[tag] += 1  # Count initial tags (first word of each sentence)

            # Count word emissions for each tag
            emit_prob[tag][word] += 1
            # Count occurrences of each tag
            tag_count[tag] += 1

            # If this is not the first word, count the transition from previous tag to current tag
            if prev_tag != "START":
                trans_prob[prev_tag][tag] += 1

            prev_tag = tag  # Set the current tag as the previous tag for the next iteration

        # Transition from the last tag in the sentence to the "END" tag
        trans_prob[prev_tag]["END"] += 1

    total_sentences = len(sentences)
    unique_tags = len(tag_count)

    # Normalize counts to probabilities using Laplace smoothing
    for tag in init_prob:
        init_prob[tag] = (init_prob[tag] + k) / (total_sentences + k * unique_tags)
        print(f"Init prob for {tag}: {init_prob[tag]}")

    for tag in emit_prob:
        total_tag_count = tag_count[tag]
        for word in emit_prob[tag]:
            # Apply Laplace smoothing to emission probabilities
            emit_prob[tag][word] = (emit_prob[tag][word] + k) / (total_tag_count + k * (len(emit_prob[tag]) + 1))

    for prev_tag in trans_prob:
        total_transitions = sum(trans_prob[prev_tag].values())
        for next_tag in trans_prob[prev_tag]:
            # Apply Laplace smoothing to transition probabilities
            trans_prob[prev_tag][next_tag] = (trans_prob[prev_tag][next_tag] + k) / (total_transitions + k * unique_tags)

    # Return the three expected values
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob, k):
    """
    Performs one step of the Viterbi algorithm by computing probabilities for the next word.
    """
    log_prob = {}  # This stores the log probability for all tags at column (i)
    predict_tag_seq = {}  # This stores the tag sequence to reach each tag at column (i)

    for curr_tag in emit_prob:
        max_prob = -inf
        best_tag = None

        for prev_tag in prev_prob:
            # Laplace smoothing for current word
            transition_prob = trans_prob[prev_tag].get(curr_tag, k / (len(trans_prob[prev_tag]) + 1))
           #  print(f'Transition prob for {prev_tag}', {transition_prob})
            emission_prob = emit_prob[curr_tag].get(word, k / (len(emit_prob[curr_tag]) + 1))

            prob = prev_prob[prev_tag] + log(transition_prob) + log(emission_prob)

            if prob > max_prob:
                max_prob = prob
                best_tag = prev_tag

        log_prob[curr_tag] = max_prob
        predict_tag_seq[curr_tag] = prev_predict_tag_seq[best_tag] + [best_tag]

    return log_prob, predict_tag_seq


def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). 
            test data (list of sentences, no tags on the words).
    output: list of sentences, each sentence is a list of (word,tag) pairs.
    '''
    # Updated to unpack only three values
    init_prob, emit_prob, trans_prob = get_probs(train)

    for t in init_prob:
        prob = init_prob[t]
        log_prob = math.log(prob)

    predicts = []

    for sentence in test:
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}

        # Initialize log probabilities for the first word
        for t in emit_prob:
            laplace_smoothing = k / (len(emit_prob[t]) + 1)  # Updated smoothing for emission
            log_init_prob = log(init_prob.get(t, laplace_smoothing))
            log_emission_prob = log(emit_prob[t].get(sentence[0], laplace_smoothing))
            log_prob[t] = log_init_prob + log_emission_prob
            predict_tag_seq[t] = [t]

        # Forward steps for each word in the sentence (start from the second word)
        for i in range(1, length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob, k)

        # Backtrace to find the best sequence
        best_tag = max(log_prob, key=log_prob.get)
        best_sequence = [best_tag]

        for i in range(length - 1, 0, -1):
            best_tag = predict_tag_seq[best_tag][-1]  # Get the previous tag in the sequence
            best_sequence.insert(0, best_tag)

        predicts.append(list(zip(sentence, best_sequence)))
        # Print init probabilities for debugging

    return predicts