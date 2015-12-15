
import sys
import numpy as np
import argparse
import random
import re
import string
import itertools
import csv
import cPickle as pickle

import pdb
from nltk import word_tokenize, ne_chunk, pos_tag
import nltk

# our modules
import read_csv_data
import similarity
import utils

def gen_pos(question_data):
	"""
	Generate POS struct
	"""
	chunked = ne_chunk(pos_tag(word_tokenize(question_data['question'])))
	#chunked = question_data['question'].split()

	pos_dict = {}
	pos_dict['pos'] = chunked
	pos_dict['id'] = question_data['id']

	return pos_dict


def load_questions(args):
	"""
	Load the question text to be analyzed
	"""

	train_reader, test_reader = read_csv_data.read_csv_data()
	train = list(train_reader); test = list(test_reader)
	random.shuffle(train)

	# split train for X-val
	if args.limit > 0:
	    train = train[0:args.limit]

	#trainx, testx = utils.split_list(train, (args.split, 100.-args.split))
	#print ("len(xval_train) = {}, len(xval_test) = {}"\
	#        .format(len(trainx), len(testx)))

	

	pdb.set_trace()
	questions_pos = []

	for ind in range(0,len(test)):
		kk = test[ind]
		questions_pos.append(gen_pos(kk))
		sys.stdout.write("Parse Progress: %f%%   \r" % (ind*100/float(len(test))) )
        sys.stdout.flush()


	#filepath = 'pos/train_pos.pkl'
	filepath = 'pos/test_pos.pkl'
	pickle.dump(questions_pos,open(filepath, 'wb'))

	pdb.set_trace()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--split", help="Percent going to training set for\
            X-val", type=float, default=100., required=False)
    argparser.add_argument("--limit", help="Limit training size",
            type=int, default=-1, required=False)
    argparser.add_argument("--test", help="Make predictions on the real-deal test set",
            action="store_true")
    args = argparser.parse_args()

    load_questions(args)