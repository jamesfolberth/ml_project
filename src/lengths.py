import cPickle as pickle
from collections import defaultdict
pages_dict = pickle.load(open('../data/wiki_pages_dict.pkl', 'rb'))

"""
Generate lengths categories
    pages_dict: pages with wiki data
    Example -
        length_dict = lengths(pages_dict)
        length_dict['Sri Lanka']
        {'links': 1232, 'title': 9, 'summary': 1983, 'content': 70077, 'sections': 0, 'categories': 32}
"""
def lengths(pages_dict):
    length_dict = defaultdict(dict)
    for k, v in pages_dict.iteritems():
        # Combine various lengths into one dict with multiple values
        if v['summary']:
            length_dict[k]['summary'] = len(v['summary'])
        else:
            length_dict[k]['summary'] = 0
        if v['content']:
            length_dict[k]['content'] = len(v['content'])
        else:
            length_dict[k]['content'] = 0
        if v['sections']:
            length_dict[k]['sections'] = len(v['sections'])
        else:
            length_dict[k]['sections'] = 0
        if v['categories']:
            length_dict[k]['categories'] = len(v['categories'])
        else:
            length_dict[k]['categories'] = 0
        if v['title']:
            length_dict[k]['title'] = len(v['title'])
        else:
            length_dict[k]['title'] = 0
        if v['links']:
            length_dict[k]['links'] = len(v['links'])
        else:
            length_dict[k]['links'] = 0
        
    return length_dict

