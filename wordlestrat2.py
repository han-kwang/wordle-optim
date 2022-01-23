#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wordle strategy 2 - brute force

Created on Sat Jan 22 21:25:10 2022

@author: hk_nien
"""
from time import time, sleep
from multiprocessing import Pool
import numpy as np
import wdata

def str2iarr(words):
    """Convert word (str) list to int16 array. '.' becomes -1.
    Single str becomes 1D array.
    """
    if isinstance(words, str):
        return_1d = True
        words = np.array([words])
    else:
        return_1d = False

    # all words as (nw, 5) array of int8
    assert isinstance(words[0], str)
    wsize = len(words[0])
    a = np.array(words).view(np.uint32).astype(np.int16)
    a = a.reshape(-1, wsize)
    a[a == ord('.')] = -1
    if return_1d:
        a = a[0, :]

    return a

def iarr2str(warr):
    """Convert int array to string or string array.

    If shape is (n, m): return array (n,) of str.
    IF shape is (m,): return str.

    Value -1 becomes '.'."""
    wsize = warr.shape[-1]
    sarr = warr.astype(np.int32)
    sarr[sarr == -1] = ord('.')
    sarr = sarr.view(f'<U{wsize}')
    if warr.ndim == 1:
        sarr = sarr[0]
    else:
        sarr = sarr.reshape(-1)
    return sarr

wdata.init('nl')
WORDS_ARR = str2iarr(wdata.WORDS)
ALPH_START = ord('a')
ALPH_LEN = ord('z') - ALPH_START + 1



def imatch_hints(pattern, badpos, badlets, warr=None):
    """Match vocabulary (int representation) against hints.

    Parameters:

    - pattern: int array with -1 for unknowns.
    - badpos: sequence of (pos, letter) tuples.
    - badlet: sequence of letters (as int) that are not in the word.
    - warr: optional candidate words array (n, 5); default full vocabulary.

    Return:

    - warr_match: matching words array (n, 5)
    """
    if warr is None:
        warr_match = WORDS_ARR
    else:
        assert warr.ndim == 2 and warr.dtype == np.int16
        warr_match = warr

    for let in badlets:
        mask = ~np.any(warr_match == let, 1)  # axis=1
        warr_match = warr_match[mask, :]
        if warr_match.shape[0] == 0:
            return warr_match
        for i, let in badpos:
            mask = np.any(warr_match == let, 1)  # axis=1
            mask &= warr_match[:, i] != let
            warr_match = warr_match[mask, :]
        if warr_match.shape[0] == 0:
            return warr_match
    for i, let in enumerate(pattern):
        if let < 0:
            continue
        mask = warr_match[:, i] == let
        warr_match = warr_match[mask, :]
        if warr_match.shape[0] == 0:
            return warr_match

    if warr_match is WORDS_ARR or warr_match is warr:
        warr_match = warr_match.copy()

    return warr_match


def gethints_all(iword, warr=None):
    """Try a word (int array) against entire vocabulary.

    Parameters:

    - iword: int array (5,) representing a trial word.
    - warr: vocabulary array (n, 5) to test against; default all.

    Return:

    - patterns: (n, 5) array with matching positions and letters
      (-1 for no match at that position)
    - badposs: (n, 5) array with matching letters at wrong positions.
      (-1 for no match at all.)
    - badletss: (n, 26) bool array, True for bad letters.
    """
    if warr is None:
        warr = WORDS_ARR
    else:
        assert warr.dtype==np.int16
    nw, wsize = warr.shape

    patterns = np.full((nw, wsize), np.int16(-1))
    badposs = patterns.copy()
    badletss = np.full((nw, ALPH_LEN), False)
    irange = np.arange(nw)
    for i, let in enumerate(iword):
        mask = (warr[:, i] == let)
        patterns[mask, i] = let
        mask2 = np.any(warr[~mask, :] == let, axis=1)
        ii = irange[~mask][mask2]
        badposs[ii, i] = let
        j = let - ALPH_START
        badletss[~np.any(warr == let, axis=1), j] = True

    return patterns, badposs, badletss


def gethints_1iword(tword, sol_word):
    """Get hints for try-word, given solution word.

    Return:

    - pattern, badpos, badlets as can be fed into imatch_hints().
    """
    pat, badpos, badlet = [
        # convert to 1D arrays
        x[0]
        for x in gethints_all(tword, warr=sol_word.reshape(1, -1))
        ]
    badpos = [
        (i, let)
        for i, let in enumerate(badpos)
        if let > 0
        ]
    badlet = (np.arange(ALPH_LEN) + ALPH_START)[badlet]
    return pat, badpos, badlet


WORKER_PERSISTENT = {}
def _test_1word(tword):
    """Return mean number of matches against all words.

    Parameter:

    - tword: word array (n,) int to use as a try word.

    For use in Pool.

    Uses WORKER_PERSISTENT keys:

    - 'warr': words dictionary to match against; (n, 5) int.
    - 'alphabet': array of alphabet (typically length 26).
    """
    alphabet = WORKER_PERSISTENT['alphabet']
    warr = WORKER_PERSISTENT['warr']
    patterns, badposs, badletss = gethints_all(tword, warr)
    nmatch = np.zeros(len(warr))

    for j, (pat, bpos, blets) in enumerate(zip(patterns, badposs, badletss)):
        blets = alphabet[blets]
        bpos = [(k, let) for k, let in enumerate(bpos) if let >= 0]
        bpos = np.array(bpos, dtype=np.int16).reshape(-1, 2)
        matches = imatch_hints(pat, bpos, blets, warr=warr)
        nmatch[j] = len(matches)
    return nmatch.mean()

def test_words(warr, twarr=None, nshow=3):
    """Test all vocabulary words for selectivity.

    Parameters:

    - warr: word array (n, 5) to match against
    - twarr: try words array (m, 5); default: same as words.
    - nshow: number of best matches to show. Set to 0 to be silent.

    Return:

    - words: sorted words array (n, 5) int, best word first.
    - mean_matches: number of remaining matches, array (n,) float.
    """
    if twarr is None:
        twarr = warr
    ntw = len(twarr)
    tm_start = tm_prev = tm = time()
    WORKER_PERSISTENT['warr'] = warr
    WORKER_PERSISTENT['alphabet'] = np.arange(ALPH_LEN) + ALPH_START

    def callback_func(_):
        print('.', end='', flush=True)

    mean_nmatchs = []
    with Pool() as pool:
        asyncs = [
            pool.apply_async(_test_1word, (tw,))
            for tw in twarr
            ]

        for i, a in enumerate(asyncs):
            mean_nmatchs.append(a.get())
            tm = time()
            if tm - tm_prev > 1:
                print(f'\rtest_words {i+1}/{ntw}...', end='')
                tm_prev = tm

    if tm - tm_start > 1:
        print(f'Done ({tm - tm_start:.0f} s).')
    mean_nmatchs = np.array(mean_nmatchs)
    ii = np.argsort(mean_nmatchs)
    twarr, mean_nmatchs = twarr[ii], mean_nmatchs[ii]

    if nshow > 0:
        show_words = [
            f'{iarr2str(w)} ({mnm:.3g})'
            for w, mnm in zip(twarr[:nshow], mean_nmatchs[:nshow])
            ]
        print(f'Best words: {", ".join(show_words)}')

    return twarr, mean_nmatchs


def _get_best_tword(warr):
    """Get best test word (and # remaining) to select from specified list."""
    if len(warr) <= 2:
        tword = warr[0]
        n = len(warr)
    else:
        twords1, scores = test_words(warr, WORDS_ARR, nshow=0)
        # get all candidate words that are optimal
        n = scores[0]
        twords1 = twords1[scores == n]
        # if any of the matching words is in the candidate list, use it.
        twords2 = set(iarr2str(twords1)).intersection(iarr2str(warr))
        if twords2:
            tword = str2iarr(sorted(twords2)[0])
        else:
            tword = twords1[0]
    return tword, n


def count_tries(secret_word, first_tword='tenor', maxtries=6, warr=None, verbose=True):
    """Return number of attempts to find the specified word.

    Parameters:

    - secret_word: secret word (int array or str)
    - first_tword: first try word (int array or str)
    - maxtries: ...
    - warr: all words to match against; (n, 5) int array.


    Return maxtries+1 if no succes.

    This includes the first word as provided and the word as found.
    """
    tword = first_tword
    if isinstance(tword, str):
        tword = str2iarr(tword)
    if isinstance(secret_word, str):
        secret_word = str2iarr(secret_word)
    if warr is None:
        warr = WORDS_ARR

    wlen = len(tword)
    pat = np.full(wlen, np.int16(-1))
    badpos = set()
    badlet = set()
    twords = []

    for itry in range(maxtries+1):
        pat1, badpos1, badlet1 = gethints_1iword(tword, secret_word)
        # breakpoint()
        twords.append(tword)
        if np.all(pat1 == tword):
            break

        # merge hints
        for i, let in enumerate(pat1):
            if let > 0:
                pat[i] = let
        for i, let in badpos1:
            badpos.add((i, let))
        for let in badlet1:
            badlet.add(let)
        # get words that match the hints
        warr = imatch_hints(pat, badpos, badlet, warr)
        if len(warr) == 0:
            itry = maxtries
            break
        else:
            tword, _ = _get_best_tword(warr)

    if verbose:
        print(f'{iarr2str(secret_word)}: {", ".join([iarr2str(tw) for tw in twords])}')
    return itry+1


def get_next_tword(pattern, badpos, twords):
    """Suggest next word to try (human-friendly interface).

    Parameters:

    - pattern: e.g. '..a.n' (green letters)
    - badpos: e.g. 'z..../bl.../....k' (letters at green positions)
    - twords: e.g. 'zuster/bloem/draak'
    """
    # Parse
    pattern = str2iarr(pattern)
    wlen = len(pattern)
    goodletters = set([let for let in pattern if let != -1])
    badpos1 = []
    for w in badpos.split('/'):
        if len(w) != wlen:
            raise ValueError(f'badpos word {w!r} wrong length')
        for i, let in enumerate(w):
            if let != '.':
                badpos1.append((i, ord(let)))
                goodletters.add(ord(let))
    badlet = set()
    for let in twords:
        if let not in './' and ord(let) not in goodletters:
            badlet.add(ord(let))
    badlet = np.array(list(badlet), dtype=np.int16)

    # Get matches
    warr = imatch_hints(pattern, badpos1, badlet)

    # Get optimal next word
    if len(warr) == 0:
        print('No match!')
    else:
        tword, n = _get_best_tword(warr)
        print(f'Best next word: {iarr2str(tword)} ({n})')


if __name__ == '__main__':
    warr = WORDS_ARR[::1]
    if 0:
        str2iarr(['aarde'])[0]


        # small dictionary for testing
        # print(list(iarr2str(warr)))

        # breakpoint()
        pats, bposs, blets = gethints_all(str2iarr(['aarde'])[0], warr)
        print(f'patterns: {list(iarr2str(pats))}')
        print(f'bposs: {list(iarr2str(bposs))}')
        print(f'badlet:\n{blets.astype(int)}')
    # goodwords, scores = test_words(warr)
    # tenor, toren, later
    # count_tries('natix', 'tenor')


