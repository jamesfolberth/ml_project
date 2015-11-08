import cPickle as pickle
from collections import defaultdict

"""
Import data: Load wiki pages, correct answers, and unique answers (our data)
    Ex: pages = pickle.load(open('C:/Users/Benjamin/Desktop/Project/wiki_pages.pkl', 'rb'))
    Will need to change directory to your computer preferences
"""
pages = pickle.load(open('C:/Users/Benjamin/Desktop/Project/wiki_pages.pkl', 'rb'))
correct = pickle.load(open('C:/Users/Benjamin/Desktop/Project/correct.pkl', 'rb'))
ans = pickle.load(open('C:/Users/Benjamin/Desktop/Project/ans.pkl', 'rb'))

# Bad words in categories (too general)
WORDS = {u'article', u'references', u'sources', u'pages', u'script', u'dmy',
         u'wikidata', u'maint', u'use', u'links', u'mdy', u'Engvarb', u'cs1'}
    
"""
Generate unique categories
    cat_pages: pages with categories (see below functions)
"""
def unique_cat(cat_pages):
    new_pages = defaultdict(dict)
    for k, v in cat_pages.items():
        if v:
            for j in v:
                if all(l not in j.lower() for l in WORDS):
                    if not k in new_pages:
                        new_pages[k] = {j}
                    else:
                        new_pages[k].update({j})
    return new_pages

"""
Generate categories for answers to questions (i.e. genres for questions)
    pages: Wiki pages
    correct: correct answers (see data)
"""
def get_correct_cat(pages, correct):
    # Initialize list of categories
    correct_cat = defaultdict(dict)
    for i in xrange(len(correct)):
        for j in xrange(len(pages)):
            if pages[j]['title'].lower() in str(correct[i]).lower():
                correct_cat[pages[j]['title']] = pages[j]['categories']
    return correct_cat

"""
Generate database for answer type (i.e. genres for answers)
    pages: Wiki pages
    correct: correct answers (see data)
(Under construction)
"""
def get_ans_cat(pages, correct):
    # Initialize list of categories
    ans_cat = defaultdict(dict)
    for i in xrange(len(correct)):
        for j in xrange(len(pages)):
            if pages[j]['title'].lower() in str(correct[i]).lower():
                correct_cat[i] = pages[j]['categories']
    return correct_cat

"""
Generate categories for Wiki pages
    pages: Wiki pages (see data)
"""
def get_pages_cat(pages):
    # Initialize list of categories
    pages_cat = defaultdict(dict)
    for i in xrange(len(pages)):
        pages_cat[pages[i]['title']] = pages[i]['categories']
    return pages_cat

"""
Save data to machine location
    name: file name
    files: file directory
    dictionary: file to be saved
"""
def save_data(name, files, dictionary):
    ## Example:
    # name = cat
    # files = 'C:/Users/Benjamin/Desktop/Project'
    name = name+'.pkl'
    filename = files+'/'+name
    pickle.dump(dictionary, open(filename, 'wb'))
