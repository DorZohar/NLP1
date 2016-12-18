import functools
import math

import time
from scipy import optimize
import numpy as np

from features import get_vector_size, q, num_of_features, feature_functor, get_vector_product
from vocabulary import all_tags
from vocabulary import feature_vec_by_family
from multiprocessing import Pool

num_of_workers = 3

start_time = time.time()

def parse(file_path):
    file = open(file_path, "r")
    content = file.read()
    file.close()

    words_and_tags = content.replace("\n", " ").split(" ")

    words_list = [word_and_tag.split("_")[0] for word_and_tag in words_and_tags]
    tags_list = [word_and_tag.split("_")[1] for word_and_tag in words_and_tags]

    return words_list, tags_list


class q_wrapper_inner(object):
    def __init__(self, vec, families):
        self.vec = vec
        self.families = families
    def __call__(self, line):
        total_sum = 0
        words = line[0]
        tags = line[1]
        for i in range(2, len(words)):
            prob_vec = q(self.vec, tags[i - 2], tags[i - 1], words, i, self.families)
            total_sum += prob_vec[tags[i]]
        return total_sum


def q_wrapper(vec, lines, lamb = 0, families = [0, 3, 4]):
    print("func enter", time.time() - start_time, vec[-10:])
    total_sum = -np.sum(vec*vec)*lamb/2

    p = Pool(num_of_workers)
    sums = p.map(q_wrapper_inner(vec, families), lines)

    total_sum += sum(sums)

    print("func exit", time.time() - start_time, total_sum)

    return -total_sum


class jacobian_inner(object):
    def __init__(self, vec, families):
        self.vec = vec
        self.families = families
    def __call__(self, line):
        words = line[0]
        tags = line[1]
        jac_vec = {}
        for i in range(2, len(words)):
            prob = np.exp(q(self.vec, tags[i - 2], tags[i - 1], words, i, self.families))
            offset = 0
            for family in self.families:
                local_get = feature_vec_by_family[family].get
                for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tags[i]):
                    key_index = local_get(key, len(self.vec) - 1 - offset)
                    jac_vec[key_index + offset] = jac_vec.setdefault(key_index + offset, 0) + 1 - prob[tags[i]]
                offset += len(feature_vec_by_family[family])

            for tag in range(0, len(all_tags)):
                if tag == tags[i]:
                    continue
                offset = 0
                for family in self.families:
                    local_get = feature_vec_by_family[family].get
                    for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tag):
                        key_index = local_get(key, len(self.vec) - 1 - offset)
                        jac_vec[key_index + offset] = jac_vec.setdefault(key_index + offset, 0) - prob[tag]
                    offset += len(feature_vec_by_family[family])
        return jac_vec


def jac_reduce_func(acc_val, sparse_vec):
    for index, val in sparse_vec.items():
        acc_val[index] += val
    return acc_val

def jacobian(vec, lines, lamb = 0, families = [0, 3, 4]):
    print("jac enter", time.time() - start_time, vec[-10:])

    jac_vec = np.zeros((len(vec),))

    p = Pool(num_of_workers)
    dicts = p.map(jacobian_inner(vec, families), lines)
    jac_vec = functools.reduce(jac_reduce_func, list(dicts), jac_vec)

    jac_vec -= lamb * jac_vec
    jac_vec[len(vec) - 1] = 0

    print("jac exit", time.time() - start_time, jac_vec[0:10])

    return -jac_vec


def calc_weight_vector(file_path, families = [0, 3, 4], lamb = 0):

    file = open(file_path, "r")
    content = file.read()
    file.close()

    feat_num = get_vector_size(families) + 1
    initial_guess = np.ones((feat_num,))
    initial_guess[feat_num - 1] = 0

    lines = [line.split(" ") for line in content.split("\n")]
    lines_as_tuples = []
    for line in lines:
        words = ['*', '*'] + [word.split("_")[0] for word in line]
        tags = [all_tags['*']] * 2 + [all_tags[tag.strip().split("_")[1]] for tag in line]
        lines_as_tuples.append((words, tags))

    res = optimize.minimize(q_wrapper,
                            initial_guess,
                            (lines_as_tuples, lamb, families),
                            'L-BFGS-B',
                            jacobian,
                            options={'disp': True})
    print(res)

    return res


if __name__ == '__main__':
    w1 = calc_weight_vector("train.wtag")
    file = open("opt_results1.py", "w")
    file.write("simple_vec = %s\n" % w1.x.tolist())
    file.close()

    w2 = calc_weight_vector("train.wtag", list(range(0, num_of_features)))

    file_ = open("opt_results2.py", "w")
    file.write("simple_vec = %s\n" % w1.x.tolist())
    file.write("advancedvec = %s\n" % w2.x.tolist())
    file.close()

