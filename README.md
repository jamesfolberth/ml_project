# Final Project for Machine Learning Course
CSCI 5622, Fall 2015

## Requirements
```
Python 2.7
cPikle (or pickle)
wikipedia
```
There are also some standard packages (e.g. `os`, `datetime`).

## Usage
I assume we're all using Python 2.7, as that's what JBG said we're going to use for the course.  Make sure you install the [wikipedia](https://pypi.python.org/pypi/wikipedia/) package and its dependencies.

To download the wikipedia data and save it to `data/wiki_data.pkl`, run
```
james@folberjm-2: src$ python2 get_wiki_data.py
```
This will take a while.

The data are stored as a list of dicts, where each dict has the following keys.
```
'title', 'summary', 'content', 'links', 'categories', 'sections'
```
