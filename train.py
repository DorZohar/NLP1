import functools
import math

import time
from scipy import optimize
import numpy as np

from features import get_vector_size, q, num_of_features, feature_functor, get_vector_product
from vocabulary import all_tags
from vocabulary import feature_vec_by_family
from multiprocessing import Pool

num_of_workers = 4

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
    return 0

    p = Pool(num_of_workers)
    results = p.map(q_wrapper_inner(vec, families), lines)

    total_sum += np.sum(results)

    print("func exit", time.time() - start_time, total_sum)

    return -total_sum


class jacobian_inner(object):
    def __init__(self, vec, families):
        self.vec = vec
        self.families = families
    def __call__(self, line):
        words = line[0]
        tags = line[1]
        jac_vec = []
        for i in range(2, len(words)):
            prob = np.exp(q(self.vec, tags[i - 2], tags[i - 1], words, i, self.families))
            offset = 0
            for family in self.families:
                local_get = feature_vec_by_family[family].get
                for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tags[i]):
                    key_index = local_get(key, len(self.vec) - 1 - offset)
                    #jac_vec.append((key_index + offset, 1))
                offset += len(feature_vec_by_family[family])

            for tag in range(0, len(all_tags)):
                #if tag == tags[i]:
                #    continue
                offset = 0
                for family in self.families:
                    local_get = feature_vec_by_family[family].get
                    for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tag):
                        key_index = local_get(key, len(self.vec) - 1 - offset)
                        jac_vec.append((key_index + offset, -prob[tag]))
                    offset += len(feature_vec_by_family[family])
        return jac_vec


def jacobian(vec, lines, lamb = 0, families = [0, 3, 4]):
    print("jac enter", time.time() - start_time, vec[-10:])

    jac_vec = np.zeros((len(vec),))
    p = Pool(num_of_workers)
    vectors_it = p.imap_unordered(jacobian_inner(vec, families), lines)

    for res in vectors_it:
        for idx, val in res:
            jac_vec[idx] += val
    p.close()
    p.join()

    jac_vec -= lamb * jac_vec
    jac_vec[len(vec) - 1] = 0

    print("jac exit", time.time() - start_time, jac_vec[0:10])

    return -jac_vec


class calculate_qs(object):
    def __init__(self, vec, families):
        self.vec = vec
        self.families = families
    def __call__(self, line):
        words = line[0]
        tags = line[1]
        qs_list = []
        total_sum = 0
        for i in range(2, len(words)):
            probs = np.exp(q(self.vec, tags[i - 2], tags[i - 1], words, i, self.families))
            total_sum += np.log(probs[tags[i]])
            qs_list.append(probs)
        return total_sum, qs_list



class func_and_grad(object):
    def __init__(self, lines, lamb, families):
        self.lines = lines
        self.lamb = lamb
        self.families = families
        feat_num = get_vector_size(families) + 1
        self.empirical = np.zeros((feat_num,))
        self.expected = []

        line_idx = 0
        for words, tags in lines:
            line_list = []
            for i in range(2, len(words)):
                offset = 0
                for family in self.families:
                    local_get = feature_vec_by_family[family].get
                    for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tags[i]):
                        key_index = local_get(key)
                        if key_index is not None:
                            self.empirical[key_index + offset] += 1
                    offset += len(feature_vec_by_family[family])

                word_list = []
                for tag in range(0, len(all_tags)):
                    tag_list = []
                    offset = 0
                    for family in self.families:
                        local_get = feature_vec_by_family[family].get
                        for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tag):
                            key_index = local_get(key)
                            if key_index is not None:
                                tag_list.append(key_index + offset)
                        offset += len(feature_vec_by_family[family])
                    word_list.append(tag_list)
                line_list.append(word_list)
            self.expected.append(line_list)
            line_idx += 1
        print("init done", time.time() - start_time)


    def __call__(self, vec):
        print("func enter", time.time() - start_time, vec[-10:])
        jac_vec = (self.empirical - self.lamb) * vec
        #jac_vec = np.zeros((len(vec),))
        p = Pool(num_of_workers)
        all_qs = p.map(calculate_qs(vec, self.families), self.lines)
        p.close()
        p.join()

        func_val = np.sum([sums for sums, qs in all_qs]) - np.sum(vec*vec)*self.lamb/2
        qs_lines = [qs for sums, qs in all_qs]

        for qs_line, expected_line in zip(qs_lines, self.expected):
            for qs_word, expected_word in zip(qs_line, expected_line):
                qs_tag = 0
                for expected_tag in expected_word:
                    tag_val = qs_word[qs_tag]
                    for idx in expected_tag:
                        jac_vec[idx] -= tag_val
                    qs_tag += 1

        print("func exit", time.time() - start_time, func_val, jac_vec[0:10])

        return -func_val, -jac_vec


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

    res = optimize.minimize(func_and_grad(lines_as_tuples, lamb, families),
                            x0=initial_guess,
                            method='L-BFGS-B',
                            jac=True,
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

