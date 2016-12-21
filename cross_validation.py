import argparse
import train
from opt_results1 import simple_vec
from viterbi import Viterbi

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lamb', type=float, default=0)
    parser.add_argument('-f', '--families', nargs='+', type=int, default=[0, 3, 4])
    args = parser.parse_args()

    print(args.lamb, " ", args.families)

    vec = train.calc_weight_vector("train.wtag", families=args.families, lamb = args.lamb )
    path = train.create_and_get_path(args.families, args.lamb)
    file = open("path", "w")
    file.write("simple_vec = %s\n" % vec.x.tolist())
    file.close()

    #vec = simple_vec

    vit = Viterbi(vec, args.families)
    vit.evaluate("validation.wtag", 3, 0, args.lamb)

    print(args.lamb)


