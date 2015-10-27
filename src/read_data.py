
import csv
import cPickle as pickle

FIELDNAMES = ("id", "question", "correctAnswer", "answerA",\
    "answerB", "answerC", "answerD")

def read_data(filename=None):
    if filename:
        return csv.DictReader(open(filename, 'r'))

    else:
        train_reader = csv.DictReader(open("../data/sci_train.csv", "r"))
        test_reader = csv.DictReader(open("../data/sci_test.csv", "r"))
        return train_reader, test_reader


if __name__ == '__main__':
    
    # this is just an example of how to use read_data
    train_reader, test_reader = read_data()
    
    count = 10
    for row in train_reader:
        for key, val in row.iteritems():
            print (val)

        count -= 1
        if count < 0:
            break

