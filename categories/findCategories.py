import cPickle as pickle
from collections import defaultdict
import json

pages = pickle.load(open('C:/Users/Benjamin/Desktop/Project/wiki_pages.pkl', 'rb'))
correct = pickle.load(open('C:/Users/Benjamin/Desktop/Project/correct.pkl', 'rb'))

# Initialize list of categories
categories = defaultdict(dict)

for i in xrange(len(pages)):
    for j in xrange(len(correct)):
        if pages[i]['title'].lower() in str(correct[j]).lower():
            categories[j] = pages[i]['categories']

pickle.dump(categories, open("C:/Users/Benjamin/Desktop/Project/categories/categories.pkl", 'wb'))
