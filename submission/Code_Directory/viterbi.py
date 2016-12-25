import copy
from features import q
from vector import simple_vec
from train import get_dir_path
from vocabulary import all_tags, all_tags_by_index
import numpy as np
import time
from multiprocessing import Pool
import argparse

class Tagger():
    def __init__(self, viterbi):
        self.viterbi = viterbi
    def __call__(self, sentence):
        return self.viterbi.tag_sentence(sentence)

class Viterbi:
    def __init__(self, vec, families):
        self.possible_tags = []
        self.q = q
        self.vec = np.array(vec)
        self.families = families

    def parse_training_file(self, path, num_sentences):
        with open(path) as f:
            sentences = f.readlines()

        if num_sentences > 0:
            sentences = sentences[:num_sentences]

        # get a list of sentences and a list of sentences tags
        words = [[tuple.split('_')[0] for tuple in sentence.strip().split(' ')] for sentence in sentences]
        tags = [[tuple.split('_')[1] for tuple in sentence.strip().split(' ')] for sentence in sentences]

        # store all the possible tags in a list
        self.possible_tags = list(set([tag for x in tags for tag in x])) # flatten 2D list and remove duplicates

        num_words = sum(len(sentence) for sentence in words)

        return words, tags, num_words

    def parse_test_file(self, path, num_sentences):
        with open(path) as f:
            sentences = f.readlines()

        if num_sentences > 0:
            sentences = sentences[:num_sentences]

        words = [[tuple.split('_')[0] for tuple in sentence.strip().split(' ')] for sentence in sentences]

        num_words = sum(len(sentence) for sentence in words)

        return words, num_words

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

    def evaluate(self, path, num_of_workers = 3, num_sentences = 10, lamb = 0.0, no_tags=False):
        start = time.time()
        num_of_workers = 50
        if no_tags:
            words, num_words = self.parse_test_file(path, num_sentences)
        else:
            words, tags, num_words = self.parse_training_file(path, num_sentences)

        num_words = 0
        count_correct = 0
        confusion_matrix = np.zeros((len(all_tags),len(all_tags)))

        p = Pool(num_of_workers)
        all_predicted_tags = p.imap(Tagger(self), words)

        if no_tags:
            # there are no tags in the test set (for the competition)
            with open(path + '.wtag', 'wb') as f:
                for predicted_tags, sentence in zip(all_predicted_tags, words):
                    for predicted_tag, word in zip(predicted_tags, sentence):
                        f.write(word + '_' + predicted_tag + ' ')
                    f.write('\n')
        else:
            # count correct and incorrect predictions
            for predicted_tags, sentence_tags in zip(all_predicted_tags, tags):
                for tag, predicted_tag in zip(sentence_tags, predicted_tags):
                    count_correct += (tag == predicted_tag)
                    try:
                        confusion_matrix[all_tags[tag], all_tags[predicted_tag]] += 1
                    except:
                        # didn't see this tag before?
                        pass
                num_words += len(sentence_tags)

            # calculate accuracy
            accuracy = count_correct / float(num_words)

            print("accuracy = " + str(accuracy))
            print("confusion matrix:")
            print(confusion_matrix)
            path = get_dir_path(self.families, lamb)
            with open("%s/confusion_matrix.csv" % path, 'wb') as f:
                np.savetxt(f, confusion_matrix, delimiter=",", fmt='%i', header=','.join(all_tags_by_index.values()))

            with open("%s/run_log.txt" % path, 'w') as f:
                f.write("accuracy = " + str(accuracy))
                f.write("time = " + str(time.time() - start))

        print("time = " + str(time.time()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', default="test.wtag")
    parser.add_argument('-n', '--num_workers', type=int, default=50)
    parser.add_argument('-s', '--num_sentences', type=int, default=0)
    parser.add_argument('--lamb', type=float, default=0)
    parser.add_argument('-f', '--families', nargs='+', type=int, default=[0, 3, 4])
    parser.add_argument('--no_tags', action='store_true')

    args = parser.parse_args()

    vit = Viterbi(simple_vec, args.families)

    vit.evaluate(args.test, args.num_workers, args.num_sentences, args.lamb, args.no_tags)
