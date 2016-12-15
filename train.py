import math

import time
from scipy import optimize
import numpy as np

from features import get_vector_product, feature_jac_dispatch, get_vector_size, q, num_of_features
from vocabulary import all_tags

start_time = time.time()

def parse(file_path):
    file = open(file_path, "r")
    content = file.read()
    file.close()

    words_and_tags = content.replace("\n", " ").split(" ")

    words_list = [word_and_tag.split("_")[0] for word_and_tag in words_and_tags]
    tags_list = [word_and_tag.split("_")[1] for word_and_tag in words_and_tags]

    return words_list, tags_list


def line_to_all_qs(words, tags, vec, families = [0, 3, 4], lamb = 0):
    total_sum = 0

    for i in range(2, len(words)):
        prob_vec = q(vec, tags[i - 2], tags[i - 1], words, i, families)
        total_sum += prob_vec[tags[i]]

    lambda_sum = sum([v*v for v in vec]) * lamb / 2

    return total_sum - lambda_sum


def q_wrapper(vec, lines, lamb = 0, families = [0, 3, 4]):
    total_sum = 0
    print("func enter", time.time() - start_time, vec[0:10])
    dict_vec = {}
    for i in range(0, len(vec)):
        dict_vec[i] = vec[i]

    for line in lines:
        #print("start:", time.time() - start_time)
        total_sum += line_to_all_qs(line[0], line[1], dict_vec, families, lamb)
        #print("end:", time.time() - start_time)

    print("func exit", time.time() - start_time, total_sum)

    return -total_sum


def jacobian(vec, lines, lamb = 0, families = [0, 3, 4]):
    print("jac enter", time.time() - start_time, vec[0:10])

    jac_vec = [0] * len(vec)
    dict_vec = {}
    for i in range(0, len(vec)):
        dict_vec[i] = vec[i]

    for line in lines:
        #print("line", time.time() - start_time)
        temp_jac = feature_jac_dispatch(families, dict_vec, line[0], line[1], lamb)
        jac_vec = [a + b for a, b in zip(jac_vec, temp_jac)]

    jac_vec = [-j for j in jac_vec]

    print("jac exit", time.time() - start_time, jac_vec[0:10])

    return np.array(jac_vec)


def calc_weight_vector(file_path, families = [0, 3, 4], lamb = 0):

    file = open(file_path, "r")
    content = file.read()
    file.close()

    feat_num = get_vector_size(families)
    initial_guess = [0] * feat_num

    lines = [line.split(" ") for line in content.split("\n")]
    lines_as_tuples = []
    for line in lines:
        words = ['*', '*'] + [word.split("_")[0] for word in line]
        tags = [all_tags['*']] * 2 + [all_tags[tag.strip().split("_")[1]] for tag in line]
        lines_as_tuples.append((words, tags))

    res = optimize.minimize(q_wrapper, initial_guess, (lines_as_tuples, lamb, families), 'L-BFGS-B', jacobian, options={'disp': True})
    print(res)

    return res


def q_for_inference(is_basic, tag_2, tag_1, words, index):
    if is_basic:
        families = [0, 3, 4]
        vec = simple_vec
    else:
        families = list(range(0, num_of_features))
        vec = advanced_vec

    return q(vec, tag_2, tag_1, words, index, families)


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

