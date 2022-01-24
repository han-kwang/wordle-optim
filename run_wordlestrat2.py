#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 22:22:13 2022

@author: hk_nien
"""
from wordlestrat2 import Wordle

if __name__ == '__main__':

    wrd = Wordle('nl')
    if 0:
        # This is slow; run this with F9 only.
        wrd.find_optimal_first_word(select=1)
    if 0:
        # This is very slow; run this with F9 only.
        wrd.find_optimal_first_word(select=2)

    if 0:
        print('Demo: how it would find words:')
        for w in wrd.words_arr[::len(wrd.words_arr)//5]:
            wrd.count_tries(w, 'tenor')
    if 1:
        print('Demo: playing')
        twords = 'toren/gebak' # words as tried
        pattern = '...e.' # green letters (position correct)
        badpos = '..r../ge.a.' # yellow letters (wrong position)
        print(f'Tried: {twords}; hits: {pattern}; bad position: {badpos}')
        wrd.get_next_tword(pattern, badpos, twords)
        wrd.match_hints(pattern, badpos, twords)
