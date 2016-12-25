import math

import time
from multiprocessing.pool import Pool
from os import makedirs

from scipy import optimize
import numpy as np
from vector import simple_vec

from features import get_vector_size, q, num_of_features, feature_functor, get_vector_product
from vocabulary import all_tags
from vocabulary import feature_vec_by_family

start_time = time.time()

num_of_workers = 3

def parse(file_path):
    file = open(file_path, "r")
    content = file.read()
    file.close()

    words_and_tags = content.replace("\n", " ").split(" ")

    words_list = [word_and_tag.split("_")[0] for word_and_tag in words_and_tags]
    tags_list = [word_and_tag.split("_")[1] for word_and_tag in words_and_tags]

    return words_list, tags_list


def get_dir_path(families, lamb):
    families_str = "Families_" + "_".join([str(family) for family in families])
    lamb_str = "Lamb_" + str(lamb).replace(".", "-")

    dir_path = "vectors/%s/%s/" % (families_str, lamb_str)
    return dir_path


def create_and_get_path(families, lamb):
    dir_path = get_dir_path(families, lamb)
    try:
        makedirs(dir_path)
    except FileExistsError:
        pass

    return dir_path + "vector.py"


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
            probs = q(self.vec, tags[i - 2], tags[i - 1], words, i, self.families)
            qs_list.append(probs)
            total_sum += probs[tags[i]]
        return total_sum, qs_list


class func_and_jacobian(object):
    def __init__(self, lines, lamb=0, families=[0, 3, 4]):
        self.lines = lines
        self.lamb = lamb
        self.families = families
        feat_num = get_vector_size(families) + 1
        self.observed = np.zeros((feat_num,))
        self.expected = []
        self.calls = 0
        self.vec_path = create_and_get_path(families, lamb)

        for line in self.lines:
            words = line[0]
            tags = line[1]
            for i in range(2, len(words)):
                offset = 0
                for family in self.families:
                    local_get = feature_vec_by_family[family].get
                    for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tags[i]):
                        key_index = local_get(key)
                        if key_index is not None:
                            self.observed[key_index + offset] += 1
                    offset += len(feature_vec_by_family[family])

            line_list = []
            for i in range(2, len(words)):
                word_list = []
                for tag in range(0, len(all_tags)):
                    offset = 0
                    for family in self.families:
                        local_get = feature_vec_by_family[family].get
                        for key in feature_functor[family](tags[i - 2], tags[i - 1], words, i, tag):
                            key_index = local_get(key)
                            if key_index is not None:
                                word_list.append((key_index + offset, tag))
                                #jac_vec[key_index + offset] -= prob[tag]
                        offset += len(feature_vec_by_family[family])
                line_list.append(word_list)
            self.expected.append(line_list)


    def keep_current_vec(self, vec):
        file = open(self.vec_path, "w")
        file.write("simple_vec = %s\n" % vec.tolist())
        file.close()


    def __call__(self, vec):
        print("jac enter", time.time() - start_time, vec[-10:])

        self.calls += 1

        if self.calls % 200 == 0:
            self.keep_current_vec(vec)

        jac_vec = np.zeros((len(vec),)) + self.observed

        p = Pool(num_of_workers)
        calc_q = calculate_qs(vec, self.families)
        all_qs_lines = p.map(calc_q, self.lines)
        total_sum = 0
        p.close()
        p.join()

        print("sync enter", time.time() - start_time)

        for line, qs_sum_and_line in zip(self.expected, all_qs_lines):
            qs_line = qs_sum_and_line[1]
            total_sum += qs_sum_and_line[0]
            for word_list, qs in zip(line, qs_line):
                prob = np.exp(qs)
                for feat, tag in word_list:
                    jac_vec[feat] -= prob[tag]

        jac_vec -= self.lamb * vec
        total_sum -= 0.5 * self.lamb * np.sum(vec*vec)
        jac_vec[len(vec) - 1] = 0

        print("jac exit", time.time() - start_time, total_sum, jac_vec[0:10])

        return -total_sum, -jac_vec


def calc_weight_vector(file_path, families = [0, 3, 4], lamb = 0):

    file = open(file_path, "r")
    content = file.read()
    file.close()

    feat_num = get_vector_size(families) + 1
    initial_guess = np.asarray(simple_vec)  #np.ones((feat_num,)) #
    initial_guess[feat_num - 1] = 0

    lines = [line.split(" ") for line in content.split("\n")]
    lines_as_tuples = []
    for line in lines:
        words = ['*', '*'] + [word.split("_")[0] for word in line]
        tags = [all_tags['*']] * 2 + [all_tags[tag.strip().split("_")[1]] for tag in line]
        lines_as_tuples.append((words, tags))

    jac = func_and_jacobian(lines_as_tuples, lamb, families)

    res = optimize.minimize(jac,
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

