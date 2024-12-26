"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    
    word_tag_count = {}
    tag_count = {}

    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag_count:
                word_tag_count[word] = {}
            if tag not in word_tag_count[word]:
                word_tag_count[word][tag] = 0
            word_tag_count[word][tag] += 1

            if tag not in tag_count:
                tag_count[tag] = 0
            tag_count[tag] += 1

    word_to_tag = {}
    for word, tag_freq in word_tag_count.items():
        word_to_tag[word] = max(tag_freq, key=tag_freq.get)

    most_common_tag = max(tag_count, key=tag_count.get)

    result = []
    for sentence in test:
        tagged_sentence = [(word, word_to_tag.get(word, most_common_tag)) for word in sentence]
        result.append(tagged_sentence)

    return result