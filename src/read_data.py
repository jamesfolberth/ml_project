"""
A simple front-end to read sci_train.csv and sci_test.csv
"""

import csv

FIELDNAMES = ("id", "question", "correctAnswer", "answerA",\
    "answerB", "answerC", "answerD")

def read_data(filename=None):
    """
    If a filename is given, open that filename with a csv.DictReader

    If no filename is given, attempt to open the train and test files
    as csv.DictReaders and return both.
    """

    if filename:
        return csv.DictReader(open(filename, 'r'))

    else:
        train_reader = csv.DictReader(open("../data/sci_train.csv", "r"))
        test_reader = csv.DictReader(open("../data/sci_test.csv", "r"))
        return train_reader, test_reader


if __name__ == '__main__':
    
    # this is just an example of how to use read_data
    #train_reader, test_reader = read_data()
    #count = 10
    #for row in train_reader:
    #    for key, val in row.iteritems():
    #        print (val)

    #    count -= 1
    #    if count < 0:
    #        break
    
    # compute the number of distinct answers
    train_reader, test_reader = read_data()
    
    train_answers = set()
    for row in train_reader:
        for ans in (row["answerA"], row["answerB"], row["answerC"],\
                row["answerD"]):
            train_answers.add(ans)

    test_answers = set()
    for row in test_reader:
        for ans in (row["answerA"], row["answerB"], row["answerC"],\
                row["answerD"]):
            test_answers.add(ans)
    
    print (len(train_answers), len(test_answers), len(test_answers-train_answers))

