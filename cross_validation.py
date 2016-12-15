import argparse
import train
from opt_results1 import simple_vec
from viterbi import Viterbi

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lamb', type=float, default=0)
    args = parser.parse_args()

    #vec = train.calc_weight_vector("train.wtag", lamb = args.lamb )

    vec = simple_vec

    vit = Viterbi(vec, [0, 3, 4])
    vit.evaluate("validation.wtag")

    print(args.lamb)


