import math

#from train import q
import threading
from multiprocessing.pool import Pool
import numpy as np
import time

from vocabulary import feature_vec_by_family, all_tags
from opt_results1 import simple_vec

num_of_letters = 4
num_of_features = 17



feature_functor = [
    lambda tag_2, tag_1, words, index, tag: [(words[index].lower(), tag)],
    lambda tag_2, tag_1, words, index, tag: [(words[index][-i:].lower(), tag) for i in range(1, 1 + min(4, len(words[index])))],
    lambda tag_2, tag_1, words, index, tag: [(words[index][0:i].lower(), tag) for i in range(1, 1 + min(4, len(words[index])))],
    lambda tag_2, tag_1, words, index, tag: [(tag_2, tag_1, tag)],
    lambda tag_2, tag_1, words, index, tag: [(tag_1, tag)],
    lambda tag_2, tag_1, words, index, tag: [tag],
    lambda tag_2, tag_1, words, index, tag: [(words[index - 1].lower(), tag)] if index > 0 else [],
    lambda tag_2, tag_1, words, index, tag: [(words[index + 1].lower(), tag)] if index < len(words) - 1 else [],
    lambda tag_2, tag_1, words, index, tag: [tag] if words[index][0].isupper() else [],
    lambda tag_2, tag_1, words, index, tag: [tag] if any(char.isdigit() for char in words[index]) else [],
    lambda tag_2, tag_1, words, index, tag: [tag] if "'" in words[index] else [],
    lambda tag_2, tag_1, words, index, tag: [(index, tag)],
    lambda tag_2, tag_1, words, index, tag: [(len(words[index]), tag)],
    lambda tag_2, tag_1, words, index, tag: [tag] if "-" in words[index] else [],
    lambda tag_2, tag_1, words, index, tag: [(words[index - 2].lower(), tag)] if index > 1 else [],
    lambda tag_2, tag_1, words, index, tag: [tag] if all(char.isdigit() or char == "," or char == "-" for char in words[index]) else [],
    lambda tag_2, tag_1, words, index, tag: [tag] if words[index].isupper() and words[index].isalpha() else [],

]


def get_vector_size(families):
    index = 0
    for f in families:
        index += len(feature_vec_by_family[f])

    return index


def get_vector_product(vec, families, tag_2, tag_1, words, index, tag):

    product = 0
    base_index = 0

    for family in families:
        keys = feature_functor[family](tag_2, tag_1, words, index, tag)
        local_get = feature_vec_by_family[family].get
        for key in keys:
            key_index = local_get(key)
            if key_index is not None:
                product += vec[key_index + base_index]
        base_index += len(feature_vec_by_family[family])

    return product



def feature_jac_dispatch(families, vec, words, tags, lamb = 0):

    jac_vec = np.zeros((len(vec),))

    offset = 0

    for family in families:
        local_get = feature_vec_by_family[family].get
        for i in range(2, len(words)):
            prob = np.exp(q(vec, tags[i - 2], tags[i - 1], words, i, families)[tags[i]])
            for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tags[i]):
                key_index = local_get(key)
                if key_index is not None:
                    jac_vec[key_index + offset] += 1 - prob
        offset += len(feature_vec_by_family[family])

    return jac_vec


def q(vec, tag_2, tag_1, words, index, families = [0, 3, 4]):

    return_vec = np.asarray(list(map(lambda tag_index: get_vector_product(vec, families, tag_2, tag_1, words, index, tag_index),
                                range(0, len(all_tags)))))
    #for tag, tag_index in all_tags.items():
    #    return_vec[tag_index] = get_vector_product(vec, families, tag_2, tag_1, words, index, tag_index)

    tags_sum = np.log(np.sum(np.exp(return_vec)))

    return return_vec - tags_sum



if __name__ == '__main__':

    print()

