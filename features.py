import math

#from train import q
import threading
from multiprocessing.pool import Pool
import numpy as np
import time

from scipy.misc import logsumexp

from vocabulary import feature_vec_by_family, all_tags, word_freq
from opt_results1 import simple_vec

num_of_letters = 4
num_of_features = 22
rare_word_freq = 2


def capitalized_seq(tag_2, tag_1, words, index, tag):
    if words[index].islower() or not words[index].isalpha() or index == len(words) - 1 or words[index + 1].islower():
        return []

    i = index + 1
    while i < len(words) - 1 and words[i + 1].isupper():
        i += 1
    return [(words[i], tag)]


def return_adjective_signs(word):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if word[-3].lower() not in vowels and word[-2] in vowels and word[-1] not in vowels:
        comparative = word + word[-1] + "er"
        superlative = word + word[-1] + "est"
    else:
        comparative = word + "er"
        superlative = word + "est"

    return comparative, superlative

def adjective_signs(tag_2, tag_1, words, index, tag):
    if not words[index].isalpha() or len(words[index]) < 3:
        return []

    comparative, superlative = return_adjective_signs(words[index])

    if comparative.lower() in word_freq and superlative.lower() in word_freq:
        return [tag]

    return []


def count_simple_verb_signs(word):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if word[-1] == 'e':
        verb_ing = word[0:-1].lower() + "ing"
    elif len(word) > 2 and word[-3].lower() not in vowels and word[-2] in vowels and word[-1] not in vowels:
        verb_ing = word.lower() + word[-1] + "ing"
    else:
        verb_ing = word.lower() + "ing"

    if word[-1] != 'e' and word[-1] in vowels and word[-2] not in vowels:
        verb_s = word.lower() + "es"
    elif word[-1] == 'y' and word[-2] not in vowels:
        verb_s = word[0:-1].lower() + "ies"
    else:
        verb_s = word.lower() + "s"

    if word[-1] == 'e':
        verb_past = word.lower() + "d"
    elif word[-1] == 'y' and word[-2] not in vowels:
        verb_past = word[0:-1].lower() + "ied"
    elif len(word) > 2 and word[-3].lower() not in vowels and word[-2] in vowels and word[-1] not in vowels:
        verb_past = word.lower() + word[-1] + "ed"
    else:
        verb_past = word.lower() + "ed"

    return int(verb_ing in word_freq) + int(verb_s in word_freq) + int(verb_past in word_freq)


def simple_verb_signs(tag_2, tag_1, words, index, tag):
    if not words[index].isalpha() or len(words[index]) < 2:
        return []

    if count_simple_verb_signs(words[index]) >= 2:
        return [tag]

    return []


def transform_past_to_present(word):
    vowels = ['a', 'e', 'i', 'o', 'u']

    if word[-3] == word[-4]:
        verb_present = word[0:-3].lower()
    elif word[-3] == 'i':
        verb_present = word[0:-3].lower() + "y"
    elif word[-4].lower() in vowels and word[-3] not in vowels:
        verb_present = word[0:-1].lower()
    else:
        verb_present = word[0:-2].lower()

    return verb_present


def past_verb_signs(tag_2, tag_1, words, index, tag):
    if not words[index].isalpha() or not words[index].endswith("ed") or len(words[index]) < 4:
        return []

    if count_simple_verb_signs(transform_past_to_present(words[index])) >= 2:
        return [tag]

    return []


def get_vector_size(families):
    index = 0
    for f in families:
        index += len(feature_vec_by_family[f])

    return index


feature_functor = [

    # 0 - The lower-cased word
    lambda tag_2, tag_1, words, index, tag: [(words[index].lower(), tag)],

    # 1 - Suffix of the lower-case word
    lambda tag_2, tag_1, words, index, tag: [(words[index][-i:].lower(), tag) for i in range(1, 1 + min(4, len(words[index])))],

    # 2 - Prefix of the lower-case word
    lambda tag_2, tag_1, words, index, tag: [(words[index][0:i].lower(), tag) for i in range(1, 1 + min(4, len(words[index])))],

    # 3 - Trigram of tags
    lambda tag_2, tag_1, words, index, tag: [(tag_2, tag_1, tag)],

    # 4 - Bigram of tags
    lambda tag_2, tag_1, words, index, tag: [(tag_1, tag)],

    # 5 - Unigram of tags
    lambda tag_2, tag_1, words, index, tag: [tag],

    # 6 - Lower-cased last word
    lambda tag_2, tag_1, words, index, tag: [(words[index - 1].lower(), tag)] if index > 0 else [],

    # 7 - Lower-cased next word
    lambda tag_2, tag_1, words, index, tag: [(words[index + 1].lower(), tag)] if index < len(words) - 1 else [],

    # 8 - Is the current word upper-cased (Can also mean non-alphanumeric etc.)
    lambda tag_2, tag_1, words, index, tag: [tag] if words[index][0].isupper() else [],

    # 9 - Does the current word contain a digit
    lambda tag_2, tag_1, words, index, tag: [tag] if any(char.isdigit() for char in words[index]) else [],

    # 10 - Does the current word contain a "'"
    lambda tag_2, tag_1, words, index, tag: [tag] if "'" in words[index] else [],

    # 11 - Index of the current word (May over-fit)
    lambda tag_2, tag_1, words, index, tag: [(index, tag)],

    # 12 - Current word's length
    lambda tag_2, tag_1, words, index, tag: [(len(words[index]), tag)],

    # 13 - Does the current word contain a "-"
    lambda tag_2, tag_1, words, index, tag: [tag] if "-" in words[index] else [],

    # 14 - The word before the last word
    lambda tag_2, tag_1, words, index, tag: [(words[index - 2].lower(), tag)] if index > 1 else [],

    # 15 - Is the current word a number
    lambda tag_2, tag_1, words, index, tag: [tag] if all(char.isdigit() or char == "," or char == "-" for char in words[index]) else [],

    # 16 - Is the current word capitalized even though it is not first in the sentence
    lambda tag_2, tag_1, words, index, tag: [tag] if (index > 2 and words[index - 1] != '.') and
                                                  words[index].isupper() and words[index].isalpha() else [],

    # 17 - Is the current word rare (Did it appear less than x times in the training)
    lambda tag_2, tag_1, words, index, tag: [tag] if words[index].isalpha() and word_freq[words[index].lower()] <= rare_word_freq else [],

    # 18 - Is the current word a part of a capitalized sequence ending with word w
    capitalized_seq,

    # 19 - Are there signs that the current word is an adjective
    adjective_signs,

    # 20 - Are there signs that the current word is a simple verb
    simple_verb_signs,

    # 21 - Are there signs that the current word is a past verb
    past_verb_signs,
]


def get_vector_product(vec, families, tag_2, tag_1, words, index, tag):

    product = 0
    base_index = 0

    for family in families:
        keys = feature_functor[family](tag_2, tag_1, words, index, tag)
        local_get = feature_vec_by_family[family].get
        for key in keys:
            key_index = local_get(key, len(vec) - 1 - base_index)
            product += vec[key_index + base_index]
        base_index += len(feature_vec_by_family[family])

    return product


def q(vec, tag_2, tag_1, words, index, families = [0, 3, 4]):

    return_vec = np.asarray(list(map(lambda tag_index: get_vector_product(vec, families, tag_2, tag_1, words, index, tag_index),
                                range(0, len(all_tags)))), dtype=np.float64)

    tags_sum = logsumexp(return_vec)

    return return_vec - tags_sum


if __name__ == '__main__':

    print()

