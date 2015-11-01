"""
Module to download and cache relevant pages from Wikipedia.
The hard work is done with the `wikipedia' package.
"""

import os
import copy
import datetime
import wikipedia as wiki

try:
    import cPickle as pickle
except:
    raise ImportWarning('Using pickle instead of cPickle; pickle might be slower.') 
    import pickle

import read_data

PAGE_KEYS = ('title', 'summary', 'content', 'links', 'categories', 'sections')

# These answers don't have an _exactly_ corresponding Wiki page.
# Could be different spellings and/or encoding issues
# Most of these can be fixed by taking the first result 
# {{{
BAD_ANSWERS = [(9, 'Accretion disc'), (62, 'Eigenvalue'), (80, 'Bose-Einstein condensate'), (88, 'Erythrocyte'), (149, 'Cu Chulainn'), (151, 'Tyr'), (154, 'Lisp'), (157, 'Secant function'), (163, 'Even number'), (170, 'Jahn-Teller effect'), (206, 'Quark-gluon plasma'), (223, 'Cosine'), (233, 'Aluminum'), (264, 'E2 reaction'), (306, 'Thermoelectric effect#Seebeck effect'), (316, 'Stern-Gerlach experiment'), (330, 'Colon (anatomy)'), (331, 'Reduction (chemistry)'), (351, 'Hans Christian Orsted'), (369, 'Mohorovicic discontinuity'), (397, 'Tang Dynasty'), (523, 'Electrical resistance'), (545, 'Mossbauer spectroscopy'), (549, 'Myelin sheath gap'), (597, 'Joule-Thomson effect'), (605, 'Creutzfeldt-Jakob disease'), (678, 'Light-independent reactions#Calvin Cycle'), (693, 'Clausius-Clapeyron relation'), (750, 'Gibbs-Duhem equation'), (777, 'Epstein-Barr virus'), (778, 'Meselson-Stahl experiment'), (781, 'Navier-Stokes equations'), (782, 'Born-Oppenheimer approximation'), (801, 'Church-Turing thesis'), (812, 'Ziegler-Natta catalyst'), (813, 'G protein-coupled receptor'), (816, 'Henderson-Hasselbalch equation'), (824, 'Time'), (864, 'Claude Levi-Strauss'), (891, 'Rain, Steam and Speed - The Great Western Railway'), (892, 'Vermiform appendix'), (917, 'Debye-Huckel equation'), (921, 'Rene Descartes'), (947, 'Lewis acid'), (953, 'Davisson-Germer experiment'), (955, 'Redlich-Kwong equation of state'), (995, 'Diels-Alder reaction'), (1019, 'Galapagos Islands'), (1026, 'Eugene Ionesco'), (1047, 'Friedel-Crafts reaction'), (1049, 'Acid-base reaction'), (1055, 'Stress-energy tensor'), (1056, 'Jons Jacob Berzelius'), (1069, 'Magnetic potential#Magnetic vector potential'), (1118, 'Tay-Sachs disease'), (1120, 'Convergence (mathematics)'), (1122, '3'), (1141, 'Ming Dynasty'), (1165, 'Gd T cells'), (1217, 'Michelson-Morley experiment'), (1233, 'Winds'), (1247, 'Epidermis (skin)'), (1277, 'Hall-Heroult process'), (1335, 'Wolff-Kishner reduction'), (1351, 'Franck-Hertz experiment'), (1394, 'Mossbauer effect'), (1401, 'D orbitals'), (1410, 'Nyquist-Shannon sampling theorem')]

FIXED_ANSWERS = {'Accretion disc': u'Accretion disk',
        'Eigenvalue': u'Eigenvalues and eigenvectors',
        'Bose-Einstein condensate': u'Bose\u2013Einstein condensate',
        'Erythrocyte': u'Red blood cell',
        'Cu Chulainn': u'C\xfa Chulainn',
        'Tyr': u'T\xfdr',
        'Lisp': u'Lisp (programming language)',
        'Secant function': u'Trigonometric functions',
        'Even number': u'Parity (mathematics)',
        'Jahn-Teller effect': u'Jahn\u2013Teller effect',
        'Quark-gluon plasma': u'Quark\u2013gluon plasma',
        'Cosine': u'Trigonometric functions',
        'Aluminum': u'Aluminium',
        'E2 reaction': u'Elimination reaction',
        'Thermoelectric effect#Seebeck effect': u'Thermoelectric effect',
        'Stern-Gerlach experiment': u'Stern\u2013Gerlach experiment',
        'Colon (anatomy)': u'Large intestine',
        'Reduction (chemistry)': u'Redox',
        'Hans Christian Orsted': u'Hans Christian \xd8rsted',
        'Mohorovicic discontinuity': u'Mohorovi\u010di\u0107 discontinuity',
        'Tang Dynasty': u'Tang dynasty',
        'Electrical resistance': u'Electrical resistance and conductance',
        'Mossbauer spectroscopy': u'M\xf6ssbauer spectroscopy',
        'Myelin sheath gap': u'Node of Ranvier',
        'Joule-Thomson effect': u'Joule\u2013Thomson effect',
        'Creutzfeldt-Jakob disease': u'Creutzfeldt\u2013Jakob disease',
        'Light-independent reactions#Calvin Cycle': u'Light-independent reactions',
        'Clausius-Clapeyron relation': u'Clausius\u2013Clapeyron relation',
        'Gibbs-Duhem equation': u'Gibbs\u2013Duhem equation',
        'Epstein-Barr virus': u'Epstein\u2013Barr virus',
        'Meselson-Stahl experiment': u'Meselson\u2013Stahl experiment',
        'Navier-Stokes equations': u'Navier\u2013Stokes equations',
        'Born-Oppenheimer approximation': u'Born\u2013Oppenheimer approximation',
        'Church-Turing thesis': u'Church\u2013Turing thesis',
        'Ziegler-Natta catalyst': u'Ziegler\u2013Natta catalyst',
        'G protein-coupled receptor': u'G protein\u2013coupled receptor',
        'Henderson-Hasselbalch equation': u'Henderson\u2013Hasselbalch equation',
        'Time': u'Time',
        'Claude Levi-Strauss': u'Claude L\xe9vi-Strauss',
        'Rain, Steam and Speed - The Great Western Railway': u'Rain, Steam and Speed \u2013 The Great Western Railway',
        'Vermiform appendix': u'Appendix (anatomy)',
        'Debye-Huckel equation': u'Debye\u2013H\xfcckel equation',
        'Rene Descartes': u'Ren\xe9 Descartes',
        'Lewis acid': u'Lewis acids and bases',
        'Davisson-Germer experiment': u'Davisson\u2013Germer experiment',
        'Redlich-Kwong equation of state': u'Redlich\u2013Kwong equation of state',
        'Diels-Alder reaction': u'Diels\u2013Alder reaction',
        'Galapagos Islands': u'Gal\xe1pagos Islands',
        'Eugene Ionesco': u'Eug\xe8ne Ionesco',
        'Friedel-Crafts reaction': u'Friedel\u2013Crafts reaction',
        'Acid-base reaction': u'Acid\u2013base reaction',
        'Stress-energy tensor': u'Stress\u2013energy tensor',
        'Jons Jacob Berzelius': u'J\xf6ns Jacob Berzelius',
        'Magnetic potential#Magnetic vector potential': u'Magnetic potential',
        'Tay-Sachs disease': u'Tay\u2013Sachs disease',
        'Convergence (mathematics)': u'Limit (mathematics)',
        '3': u'3 (number)',
        'Ming Dynasty': u'Ming dynasty',
        'Gd T cells': u'Gamma delta T cell',
        'Michelson-Morley experiment': u'Michelson\u2013Morley experiment',
        'Winds': u'Wind',
        'Epidermis (skin)': u'Epidermis',
        'Hall-Heroult process': u'Hall\u2013H\xe9roult process',
        'Wolff-Kishner reduction': u'Wolff\u2013Kishner reduction',
        'Franck-Hertz experiment': u'Franck\u2013Hertz experiment',
        'Mossbauer effect': u'M\xf6ssbauer effect',
        'D orbitals': u'Atomic orbital',
        'Nyquist-Shannon sampling theorem': u'Nyquist\u2013Shannon sampling theorem'}

# }}}

def read_answers():
    """
    Read in the test and train data and find all distinct answers.  It is
    assumed that each answer is the title of a wikipedia page.

    Args:
        None

    Returns:
        list of distinct answer strings
    """
    train_reader, test_reader = read_data.read_data()
    
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
    
    return list(train_answers | test_answers)


def get_bad_answers(answers, delay=50000, cutoff=-1):
    """
    Query Wikipedia for each answer, and return a list of answers that 
    _don't_ correspond directly to page names.

    Args:
        answers: list of answer strings that should corespond to Wiki pages

        delay=50000: number of microseconds to wait between pages
        cutoff=-1: only process cutoff pages (useful for development)

    Returns:
        None
    """
    # wikipedia module does its own rate limiting, so let's use that
    wiki.set_rate_limiting(True, min_wait=datetime.timedelta(0,0,delay))
    
    bad_results = []
    for i, answer in enumerate(answers):
        print (i, len(answers))
        res = wiki.search(answer, results=3)
        if res[0] != answer:
            print ("bad result!", answer, res)
            bad_results.append((i,answer))

        if cutoff > 0 and i >= cutoff-1:
            break
    
    print (bad_results)


def get_wiki_data(answers, pkl_file, delay=50000):
    """
    Download pages from Wikipedia and store them in a binary pickle file.
    The pickle file is a list of dicts, where each dict has the following keys:
        'title'
        'summary'
        'content'
        'links'
        'categories'
        'sections'

    This should be able to recover from crashes.  Just run it again with the
    same arguments, and it'll pick up where it left off.

    Args:
        answers: list of answer strings that should corespond to Wiki pages
        pkl_file: file used to store pickled output

        delay=50000: number of microseconds to wait between pages

    Returns:
        None
    """
    # funtion to try/except a page property (e.g. summary, links)
    def try_page_property(dict_page, page, attr):
        try:
            tmp = getattr(page, attr)
            dict_page[attr] = tmp
        except KeyError: # sometimes links doesn't work
            dict_page[attr] = None


    # wikipedia module does its own rate limiting, so let's use that
    wiki.set_rate_limiting(True, min_wait=datetime.timedelta(0,0,delay))

    # try to read which pages we've already done (i.e. if we're restarting)
    pages = []
    if os.path.isfile(pkl_file):
        with open(pkl_file, 'rb') as f:
            try:
                pages = pickle.load(f)
            except Exception: # if nothing gets loaded/bad file descriptor
                pass 
    
    # find pages we need to do
    pages_todo = set(answers)
    for page in pages:
        update_keys = (key in page.keys() for key in PAGE_KEYS)
        if all(update_keys): # if we have entries for all data keys
            pages_todo.discard(page['title'])
        
    pages_todo = list(pages_todo)

    # download wiki pages
    for i, title in enumerate(pages_todo):
        print (u"page {} of {} (title={})".format(i+1, len(pages_todo), title))

        page = wiki.page(title=title)
        
        # get the page data and store it in a dict
        dict_page = dict()
        for key in PAGE_KEYS:
            try_page_property(dict_page, page, key) 

        pages.append(dict_page)
        
        # dumping each time is safe, but slow
        # TODO: how safe is this?
        pickle.dump(pages, open(pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    answers = read_answers()
    
    #get_bad_answers(answers)
    
    # these answers correspond directly to pages, so we 
    updated_answers = [FIXED_ANSWERS[ans] if ans in FIXED_ANSWERS.keys() else ans for ans in read_answers()]
    
    get_wiki_data(updated_answers, '../data/wiki_pages.pkl')
