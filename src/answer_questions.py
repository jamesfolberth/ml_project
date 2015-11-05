
import numpy as np
import argparse

import cPickle as pickle

import read_csv_data

def run_development(args):
    """
    Do development things, such as computing and saving features, X-val, etc.
    """
    
    train_reader = read_csv_data.read_csv_data('../data/sci_train.csv')
    pages_dict = pickle.load(open('../data/wiki_pages_dict.pkl', 'rb'))
    
    # iterate over all answers and print the corresponding wiki page title
    # as a check that pages_dict has all the right keys
    for row in train_reader:
        for ans in (row['answerA'], row['answerB'], row['answerC'], row['answerD']):
            print (u"answer: {}\n  title: {}".format(ans, pages_dict[ans]['title']))


def make_predictions(args):
    """
    Make predictions on the full test set.
    """
    raise NotImplementedError("TODO")


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
    
    run_development(args)
    #make_predictions(args)

