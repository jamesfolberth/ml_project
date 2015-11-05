import cPickle as pickle
from collections import defaultdict
import json

pages = pickle.load(open('../wiki_pages.pkl', 'rb'))
correct = pickle.load(open('../correct.pkl', 'rb'))

# Initialize list of categories
categories = defaultdict(dict)

for i in xrange(len(pages)):
    for j in xrange(len(correct)):
        if pages[i]['title'].lower() in str(correct[j]).lower():
            categories[j] = pages[i]['categories']

pickle.dump(categories, open("../categories.pkl", 'wb'))
