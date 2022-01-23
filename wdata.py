#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 21:27:22 2022

@author: hk_nien
"""
import re

WORDS = []
WSIZE = 5

def _get5words(files, blacklist=()):
    """Return words list for dictionaries"""
    words = set()
    for fn in files:
        words.update(open(fn).read().split('\n'))
    exp = re.compile('[a-zA-Z]{5}$')
    words = sorted(set(
        w.lower() for w in words
        if re.match(exp, w)
        ) - set(blacklist))
    return words

def _get5words_en():
    return _get5words(
        ['/usr/share/dict/american-english', '/usr/share/dict/british-english'],
        blacklist={'aires', 'aries', 'bligh'}
        )

def _get5words_en_2(n=2700):
    """Return sorted word list, most frequent on top.

    Corpus size 2700 seems to be about what Wordle uses.
    """
    words = open('wordlist-en-freq.txt').read().split('\n')
    exp = re.compile('[A-Za-z]{5}$')
    words = [
        w.lower()
        for w in words
        if re.match(exp, w) and w not in ('aries', 'bligh', 'begum')
        ]
    return words[:n]

def _get5words_nl():
    return _get5words(
        ['woordle-nl.txt'],
        blacklist={
            'aties','olink', 'molin', 'limon', 'leoni', 'pilon',
            'budak', 'bedum', 'dumps'
            }
        )
def _get5words_nl():
    return _get5words(
        ['woordle-nl.txt'],
        blacklist={
            'aties','olink', 'molin', 'limon', 'leoni', 'pilon',
            'budak', 'bedum', 'dumps'
            }
        )

def init(wset):
    """Init global WORDS list from dataset 'en', 'enbig', 'nl', 'nlbig'."""
    global WORDS
    if wset == 'nl':
        WORDS = _get5words(['woordle-nl.txt'])
    elif wset == 'nlbig':
        WORDS = _get5words(['woordle-nl-full.txt'])
    elif wset == 'en':
        WORDS = _get5words_en_2(2700)
    elif wset == 'enbig':
        WORDS = _get5words_en_2(4000)
    else:
        raise ValueError(f'wset={wset!r}')