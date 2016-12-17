import copy
from features import q
from opt_results0_0001 import simple_vec
from vocabulary import all_tags, all_tags_by_index
import numpy as np
import time

class Viterbi:
    def __init__(self, vec, families):
        self.possible_tags = []
        self.q = q
        self.vec = np.array(vec)
        self.families = families

    def parse_training_file(self, path):
        with open(path) as f:
            sentences = f.readlines()

        sentences = sentences[1:20]

        words = [[tuple.split('_')[0] for tuple in sentence.strip().split(' ')] for sentence in sentences]
        tags = [[tuple.split('_')[1] for tuple in sentence.strip().split(' ')] for sentence in sentences]

        self.possible_tags = list(set([tag for x in tags for tag in x])) # flatten 2D list and remove duplicates

        num_words = sum(len(sentence) for sentence in words)

        return words, tags, num_words

    def tag_sentence(self, sentence):
        n = len(sentence)

        num_tags = len(all_tags)

        # create a table for pi
        pi = np.ones((n+1, num_tags, num_tags))
        bp = np.zeros((n+1, num_tags, num_tags))

        # initialization
        start_tag_index = all_tags['*']
        pi *= float('inf')
        pi[0][start_tag_index][start_tag_index] = 0

        for k in range(1,n+1):
            # set the tag sets for each position
            for u in range(num_tags):
                if k > 1 and u == start_tag_index:
                    continue
                for t in range(num_tags):
                    if k > 2 and t == start_tag_index:
                        continue
                    q_for_all_tags = -self.q(self.vec, t, u, sentence, k-1, self.families)
                    for v in range(num_tags):
                        if v == start_tag_index:
                            continue
                        curr_prob = pi[k-1][t][u]+q_for_all_tags[v]
                        if curr_prob < pi[k][u][v]:
                            pi[k][u][v] = curr_prob
                            bp[k][u][v] = t

        # initialize last two tags
        tags = np.zeros((n+1,), dtype=np.int)
        max_prob = float('inf')
        for u in range(num_tags):
            for v in range(num_tags):
                curr_prob = pi[n][u][v]
                if curr_prob < max_prob:
                    max_prob = curr_prob
                    tags[n] = v
                    tags[n-1] = u

        # find all other tags
        for k in range(n-2,0,-1):
            #print(k, tags[k+1], tags[k+2])
            tags[k] = bp[k+2][tags[k+1]][tags[k+2]]

        tag_names = [all_tags_by_index[tag] for tag in tags]

        return tag_names[1:]

    def evaluate(self, path):
        words, tags, num_words = self.parse_training_file(path)
        count_correct = 0
        for sentence, sentence_tags in zip(words, tags):
            predicted_tags = self.tag_sentence(sentence)
            count_correct += sum([1 for tag, predicted_tag in zip(sentence_tags, predicted_tags) if tag == predicted_tag])
            print(sentence_tags)
            print(predicted_tags)
            print("-----------------------")

        accuracy = count_correct/float(num_words)
        error = 1 - accuracy

        print(accuracy)


if __name__ == '__main__':
    vit = Viterbi(simple_vec, [0, 3, 4])

    vit.evaluate("test.wtag")
