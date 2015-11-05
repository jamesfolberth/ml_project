import cPickle as pickle
from collections import defaultdict

'''
Load wiki pages, correct answers, and unique answers
'''
pages = pickle.load(open('../wiki_pages.pkl', 'rb'))
correct = pickle.load(open('../correct.pkl', 'rb'))
ans = pickle.load(open('../ans.pkl', 'rb'))

# Initialize list of categories
correct_cat = defaultdict(dict)
ans_cat = defaultdict(dict)

# Generate categories for answers to questions (i.e. genres for questions)
for i in xrange(len(correct)):
    for j in xrange(len(pages)):
        if pages[j]['title'].lower() in str(correct[i]).lower():
            correct_cat[i] = pages[j]['categories']

# Generate database for answer type (i.e. genres for answers)
for i in xrange(len(ans)):
    for j in xrange(len(pages)):
        if ans[i].lower() in pages[j]['title'].lower():
            ans_cat[ans[i]] = pages[j]['categories']

'''
Save data
'''
# pickle.dump(categories, open("../correct_cat.pkl", 'wb'))
# pickle.dump(categories, open("../ans_cat.pkl", 'wb'))
