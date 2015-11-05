
import numpy as np
import argparse

import cPickle as pickle


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set for\
            X-val", type=float, default=90., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--load", help="Load precomputed features",
            type=bool, default=False, required=False)
    argparser.add_argument("--loadex", help="Load precomputed examples only",
            type=bool, default=False, required=False)
    argparser.add_argument("--test", help="Make predictions on the true test set",
            type=bool, default=False, required=False)

    args = argparser.parse_args()

    #make_predictions(args)
    #run_development(args)

