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

import read_csv_data

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
        'Nyquist-Shannon sampling theorem': u'Nyquist\u2013Shannon sampling theorem',
        'Hyperbolic': 'Hyperbola',
        'Lagrangian': 'Lagrangian mechanics',
        'Bremsstrahlung': 'Bremsstrahlung radiation'}


# This has issues if there are multiple pages that get ``fixed''
# to the same page (e.g. secand function and cosine)
#FIXED_ANSWERS_INV = {val: key for key,val in FIXED_ANSWERS.iteritems()}

FIXED_ANSWERS_INV = dict()
for key,val in FIXED_ANSWERS.iteritems():
    if val in FIXED_ANSWERS_INV:
        FIXED_ANSWERS_INV[val].append(key)
    else:
        FIXED_ANSWERS_INV[val] = [key]

#ANSWER_TO_PAGEID = {'Pascal (programming language)': 23773,
#        'C (programming language)': 7004623,
#        'Java (programming language)': 6877888,
#        'Time': 30012,
#        'Bremsstrahlung': 48427746}

ANSWER_TO_PAGEID = {'Pascal (programming language)': 23773,
        'Time': 30012}

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
    train_reader, test_reader = read_csv_data.read_csv_data()
    
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
        
        try:
            #XXX even though we checked for bad pages, this sometimes raises
            # an exception.  For example, we could do 
            # title=u"Pascal (programming language)", but still get a PageError
            # saying Page id "pascal programming langauge" does not match any
            # pages. We can manually look up the Mediawiki pageid (using their API)
            # and look up the page using that 
            if title in ANSWER_TO_PAGEID.keys():
                page = wiki.page(pageid=ANSWER_TO_PAGEID[title])
            else:
                page = wiki.page(title=title, auto_suggest=False)
            
            # get the page data and store it in a dict
            dict_page = dict()
            for key in PAGE_KEYS:
                try_page_property(dict_page, page, key) 
            
            print (page.title)
            pages.append(dict_page)
            
            # dumping each time is safe, but slow
            # TODO: how safe is this?
            pickle.dump(pages, open(pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)

        except wiki.exceptions.WikipediaException as e:
            print (u"wiki issues with page (title={})".format(title))
            print (e)

        except Exception as e:
            print ("something else bad has happened!")
            print (e)


def clean_wiki_pages(pkl_file, answers=None):
    """
    Remove duplicates from wiki pages pickle file.  Keep the highest index
    (closest to end) for non-unique elements.  Uniquity is measured only
    by the title of the page.

    Args:
        pkl_file: file used to store pickled output

        answers=None: see which titles we don't have

    Returns:
        missing_pages: set of answers with no matching page
    """

    pages = []
    if os.path.isfile(pkl_file):
        with open(pkl_file, 'rb') as f:
            try:
                pages = pickle.load(f)
            except Exception: # if nothing gets loaded/bad file descriptor
                pass 
  
    unique_pages = [] # uniqueness is measured by title
    unique_titles = set()
    if answers:
        missing_pages = set(answers)
    else:
        missing_pages = set()

    for page in pages:
        title = page['title']
        if title not in unique_titles:
            unique_pages.append(page)
            unique_titles.add(title)
            missing_pages.discard(title)
    
    pickle.dump(unique_pages, open(pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    
    return missing_pages


def make_pages_dict(answers, pages_pkl_file, dict_pkl_file):
    """
    Make a dictionary mapping answer string to wiki page data.
    Ensure that each answer corresponds to the correct page.

    Args:
        answers: list of answers coming from CSV data
        pages_pkl_file: file with list of wiki pages
        dict_pkl_file: where to save the dict

    Returns:
        None
    """
     
    pages = []
    if os.path.isfile(pages_pkl_file):
        with open(pages_pkl_file, 'rb') as f:
            try:
                pages = pickle.load(f)
            except Exception: # if nothing gets loaded/bad file descriptor
                pass 
    
    answers_set = frozenset(answers)
    found_answers = set()
    pages_dict = dict()
    if pages:
        for page in pages:
            
            # sometimes there exists a page in answers_set _and_ in FIXED_...
            bad_title = True
            if page['title'] in answers_set:
                pages_dict[page['title']] = page
                found_answers.add(page['title'])
                bad_title = False

            if page['title'] in FIXED_ANSWERS_INV:
                found_ans = False
                for ans in FIXED_ANSWERS_INV[page['title']]:
                    if ans in answers_set:
                        pages_dict[ans] = page
                        found_answers.add(ans)
                        found_ans = True
                        bad_title = False

                if not found_ans:
                    print ("FAI: " + page['title'])


            if bad_title:
                print ("title without answer: " + page['title'])

    pickle.dump(pages_dict, open(dict_pkl_file, 'wb'), pickle.HIGHEST_PROTOCOL)


def check_pages_dict(answers, dict_pkl_file):
    """
    Ensure that CSV answers are keys by iterating through all answers

    Args:
        answers: list of answers from CSV files
        dict_pkl_file: file with dictionary mapping answers to wiki page data

    Returns:
        no_matches: set of answers with no matching entry
    """
        
    pages_dict = pickle.load(open(dict_pkl_file, 'rb'))
    
    no_matches = set()
    for ans in answers:
        if not ans in pages_dict:
            no_matches.add(ans)

    return no_matches


if __name__ == '__main__':
    answers = read_answers()
    
    #get_bad_answers(answers)
    
    # these answers correspond directly to pages, so we try to fix them manually
    updated_answers = [FIXED_ANSWERS[ans] if ans in FIXED_ANSWERS.keys() else ans for ans in read_answers()]
    
    # do the hard work
    get_wiki_data(updated_answers, '../data/wiki_pages.pkl')


    missing_pages = clean_wiki_pages('../data/wiki_pages.pkl', answers=updated_answers)
    if missing_pages:
        print ("missing pages: ")
        print (missing_pages)
    else:
        print ("No missing pages")

    
    make_pages_dict(answers, '../data/wiki_pages.pkl', '../data/wiki_pages_dict.pkl')


    no_matches = check_pages_dict(answers, '../data/wiki_pages_dict.pkl')
    if not no_matches:
        print ("All answers have a corresponding entry in pages_dict!")
    else:
        print ("No entry for the following answers:")
        print (no_matches)
 

