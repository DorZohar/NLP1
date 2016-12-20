import math

from features import num_of_features, rare_word_freq, capitalized_seq

all_tags = {'PRP': 25, 'VBN': 37, 'NNP': 20, 'SYM': 31, '$': 1, '-RRB-': 6, 'UH': 33, 'FW': 13, 'PRP$': 26, 'NNPS': 21, 'JJS': 17, 'IN': 14, '*': 3, 'VBZ': 39, "''": 2, 'VBD': 35, 'RP': 30, 'DT': 11, 'CC': 9, 'TO': 32, 'WP$': 42, '``': 44, ':': 8, 'VBP': 38, 'PDT': 23, 'WDT': 40, 'WRB': 43, 'RBS': 29, 'JJ': 15, 'EX': 12, 'CD': 10, 'WP': 41, 'MD': 18, 'VB': 34, ',': 4, 'NNS': 22, 'RBR': 28, 'POS': 24, '.': 7, 'NN': 19, 'JJR': 16, 'RB': 27, 'VBG': 36, '#': 0, '-LRB-': 5}

all_tags_by_index = {0: '#', 1: '$', 2: "''", 3: '*', 4: ',', 5: '-LRB-', 6: '-RRB-', 7: '.', 8: ':', 9: 'CC', 10: 'CD', 11: 'DT', 12: 'EX', 13: 'FW', 14: 'IN', 15: 'JJ', 16: 'JJR', 17: 'JJS', 18: 'MD', 19: 'NN', 20: 'NNP', 21: 'NNPS', 22: 'NNS', 23: 'PDT', 24: 'POS', 25: 'PRP', 26: 'PRP$', 27: 'RB', 28: 'RBR', 29: 'RBS', 30: 'RP', 31: 'SYM', 32: 'TO', 33: 'UH', 34: 'VB', 35: 'VBD', 36: 'VBG', 37: 'VBN', 38: 'VBP', 39: 'VBZ', 40: 'WDT', 41: 'WP', 42: 'WP$', 43: 'WRB', 44: '``'}


def write_to_file(file_path):
    file = open(file_path, "r")
    content = file.read()
    lines = [line.split(" ") for line in content.split("\n")]
    lines_as_tuples = []
    word_freq = {}
    for line in lines:
        words = ['*'] * 2 + [word.split("_")[0] for word in line]
        tags = [all_tags['*']] * 2 + [all_tags[tag.split("_")[1]] for tag in line]
        lines_as_tuples.append((words, tags))

    func_sets = []
    for i in range(0, num_of_features):
        func_sets.append(set())

    # Count words in vocabulary
    for line in lines_as_tuples:
        for i in range(0, len(line[0])):
            if line[0][i].lower() in word_freq:
                word_freq[line[0][i].lower()] += 1
            else:
                word_freq[line[0][i].lower()] = 1

    for line in lines_as_tuples:
        for i in range(0, len(line[0])):
            func_sets[0].add((line[0][i].lower(), line[1][i]))

            if i >= 2:
                for j in range(1, 1 + min(len(line[0][i]), 4)):
                    func_sets[1].add((line[0][i][-j:].lower(), line[1][i]))
                    func_sets[2].add((line[0][i][0:j].lower(), line[1][i]))

                    func_sets[3].add((line[1][i-2], line[1][i-1], line[1][i]))

            if i >= 1:
                func_sets[4].add((line[1][i - 1], line[1][i]))

            func_sets[5].add(line[1][i])

            if i >= 1:
                func_sets[6].add((line[0][i - 1].lower(), line[1][i]))

            if i < len(line[0]) - 1:
                func_sets[7].add((line[0][i + 1].lower(), line[1][i]))

            if line[0][i][0].isupper() and line[0][i][0].isalpha():
                func_sets[8].add(line[1][i])

            if any(char.isdigit() for char in line[0][i]):
                func_sets[9].add(line[1][i])

            if "'" in line[0][i]:
                func_sets[10].add(line[1][i])

            func_sets[11].add((i, line[1][i]))
            func_sets[12].add((len(line[0][i]), line[1][i]))

            if "-" in line[0][i]:
                func_sets[13].add(line[1][i])

            if i >= 2:
                func_sets[14].add((line[0][i - 2].lower(), line[1][i]))

            if all(char.isdigit() or char == "," or char == "-" for char in line[0][i]):
                func_sets[15].add(line[1][i])

            if i > 2 and line[0][i-1] != '.' and line[0][i].isupper() and line[0][i].isalpha():
                func_sets[16].add(line[1][i])

            if line[0][i].isalpha() and word_freq.get(line[0][i].lower(), 0) <= rare_word_freq:
                func_sets[17].add(line[1][i])

            if i >= 2:
                cap_word_family18 = capitalized_seq(line[1][i-2], line[1][i-1], line[0], i, line[1][i])
                if len(cap_word_family18) > 0:
                    func_sets[18].add(cap_word_family18[0])


    file = open("vocabulary2.py", "w")

    sets_len = []
    for func_set in func_sets:
        sets_len.append(len(func_set))

    file.write("num_of_features = %s\n\n" % sets_len)

    file.write("all_tags = %s\n" % all_tags)
    file.write("all_tags_by_index = %s\n\n" % all_tags_by_index)

    file.write("feature_vec_by_family = {}\n\n")

    func_vector = []
    for i in range(0, num_of_features):
        func_vector.append({})
        ordered_list = list(func_sets[i])
        ordered_list.sort()
        for j in range(0, len(ordered_list)):
            func_vector[i][ordered_list[j]] = j
        file.write("feature_vec_by_family[%d] = %s\n" % (i, func_vector[i]))

    file.write("\nword_freq = %s\n" % word_freq)

    file.close()


if __name__ == '__main__':
    write_to_file("train.wtag")
