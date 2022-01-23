#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wordle strategy 2 - brute force

For online game:

- https://www.powerlanguage.co.uk/wordle/ (English, ~2700 dictionary)
- https://hellowordl.net/ (English, ~5000 dictionary?)
- https://woordle.nl/ (Dutch, 860 solutions but recognizes ~5500 words).


Finding optimal words to try, given hints.

Strings are handled as numpy int16 arrays; positive values for unicode/ascii
characters and value -1 for 'undefined/unknown'.

Functions with int array interface:

- str2iarr(): convert strings to int arrays
- iarr2str(): convert int array to strings
- gethints_1iword: Get hints for try-word, given solution word.
- gethints_all: Try a word (int array) against entire vocabulary
- imatch_hints: Match vocabulary against hints

Functions with string interface:

- init(): load/initialize dictionaries
- count_tries(): Return number of attempts to find the specified word.
- find_optimal_first_word(): Find optimal first word. This is slow.
- get_next_tword(): Suggest next word to try
- test_words: Test all vocabulary words for selectivity.

Globals:

- ALPH_LEN: alphabet length (typically 26)
- ALPH_START: ascii code of first alphabet letter (all is lowercase)
- WORDS_ARR: puzzle word array, shape (n, 5)
- WORDS_ARR_BIG: recognized word array, shape (nn, 5).

Created on Sat Jan 22 21:25:10 2022 // author: hk_nien
"""
from time import time
from multiprocessing import Pool
import numpy as np
import wdata

WORDS_ARR = None  # Will be (n, 5) int array
WORDS_ARR_BIG = None  # Will be (nn, 5) int array
ALPH_START = ord('a')
ALPH_LEN = ord('z') - ord('a') + 1

#%%

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


def init(dataset='nl'):
    """Initialize for 'nl', 'en-orig', or 'en-hello'."""
    global WORDS_ARR, WORDS_ARR_BIG, ALPH_START, ALPH_LEN
    ALPH_START = ord('a')
    ALPH_LEN = ord('z') - ALPH_START + 1

    if dataset == 'nl':
        wdata.init('nlbig')
        WORDS_ARR_BIG = str2iarr(wdata.WORDS)
        wdata.init('nl')
        WORDS_ARR = str2iarr(wdata.WORDS)
    elif dataset == 'en-orig':
        wdata.init('en')
        WORDS_ARR = WORDS_ARR_BIG = str2iarr(wdata.WORDS)
    elif dataset == 'en-hello':
        wdata.init('enbig')
        WORDS_ARR = WORDS_ARR_BIG = str2iarr(wdata.WORDS)
    else:
        raise ValueError(f'dataset={dataset!r}')

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

def test_words(warr, twarr=None, nshow=3, pri_time=2):
    """Test all vocabulary words for selectivity.

    Parameters:

    - warr: word array (n, 5) to match against
    - twarr: try words array (m, 5); default: same as words.
    - nshow: number of best matches to show. Set to 0 to be silent.
    - pri_time: show progress indicator if run time exceeds this.

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
            if tm - tm_prev > 1 and tm - tm_start > pri_time:
                print(f'\rtest_words {i+1}/{ntw}...', end='')
                tm_prev = tm

    if tm - tm_start > pri_time:
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
        n = max(1, len(warr)-1)
    else:
        twords1, scores = test_words(warr, WORDS_ARR, nshow=0, pri_time=3)
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


def color_str_try(secret_word, try_word):
    """Return string with ANSI escape sequences for hints.

    Parameters:

    - secret_word: string
    - try_word: string
    """
    esc = lambda c, x: f'\033[{c}m{x}'
    out = [esc('38;2;0;0;0', '')]  # black foregroud
    for tl, sl in zip(try_word, secret_word):
        if tl == sl:
            out.append(esc('48;2;40;200;40', tl)) # green
        elif tl in secret_word:
            out.append(esc('48;2;200;150;40', tl)) # orange
        else:
            out.append(esc('48;2;180;180;180', tl))
    out.append(esc('0', ''))  # reset
    return ''.join(out)


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
        color_words = [
            color_str_try(iarr2str(secret_word), iarr2str(tw))
            for tw in twords
            ]
        if itry >= maxtries:
            msg = '\033[31;1m  (Not found)\033[0m'
        else:
            msg = ''
        print(f'{iarr2str(secret_word)}: {", ".join(color_words)}{msg}')
    return itry+1


def match_hints(pattern, badpos, twords, mode='print'):
    """Show or return words that match the hints; human-friendly.

    Parameters:

    - pattern: e.g. '..a.n' (green letters)
    - badpos: e.g. 'z..../bl.../....k' (letters at green positions)
    - twords: e.g. 'zuster/bloem/draak'
    - mode: 'print' or 'return'

    Return (only for mode='return'):

    - warr: list of matching words (as int array)
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
                ilet = ord(let)
                if pattern[i] == ilet:
                    raise ValueError(
                        f'Inconsistent hints: {let} at position {i+1}'
                        )
                badpos1.append((i, ilet))
                goodletters.add(ilet)
    badlet = set()
    for let in twords:
        if let not in './' and ord(let) not in goodletters:
            badlet.add(ord(let))
    badlet = np.array(list(badlet), dtype=np.int16)

    # Get matches
    warr = imatch_hints(pattern, badpos1, badlet)
    if mode == 'print':
        wlist = iarr2str(warr)
        txt = ', '.join(wlist[:20])
        if len(wlist) > 20:
            print(f'Showing 20 of {len(wlist)} matches: {txt}')
        elif len(wlist) == 0:
            print('No matches!')
        else:
            print(f'Found matches: {txt}')
        return None
    elif mode == 'return':
        return warr
    else:
        raise ValueError(f'mode={mode!r}')


def get_next_tword(pattern, badpos, twords):
    """Suggest next word to try (human-friendly interface).

    Parameters:

    - pattern: e.g. '..a.n' (green letters)
    - badpos: e.g. 'z..../bl.../....k' (letters at green positions)
    - twords: e.g. 'zuster/bloem/draak'
    """
    try:
        warr = match_hints(pattern, badpos, twords, mode='return')
    except ValueError as e:
        if 'Inconsistent hints' in e.args[0]:
            print(f'Error: {e.args[0]}')
            return
        raise
    # Get optimal next word
    if len(warr) == 0:
        print('No match!')
    else:
        if len(warr)*len(WORDS_ARR) > 30000:
            print(f'({len(warr)} words match the hints)')
        tword, n = _get_best_tword(warr)
        print(f'Best next word: {iarr2str(tword)} ({n:.2f}/{len(warr)})')


def find_optimal_first_word(select=1, ret_full=False, print_=True):
    """Find optimal first word. This is slow.

    Parameters:

    - select: indicate dictionary size to work with; 0: small for testing;
      1: select words from WORDS_ARR; 2: select from WORDS_ARR_BIG.
    - ret_full: True to return full list; False to return best 10 and
      another 10 picked from the rest.

    Return:

    - List of (word_str, mean_matches)

    For language NL, select=2 (865/5539 words):
    tenor (21.6), raten (21.9), toner (22.3), ..., puppy (491)

    For en-orig, select=1 (2700/2700 words):
    raise (57.9), arise (61.7), rates (62), ..., fuzzy (1202)
    """
    if select == 0:
        warr, twarr = WORDS_ARR[::5], WORDS_ARR[::5]
    elif select == 1:
        warr, twarr = WORDS_ARR, WORDS_ARR
    elif select == 2:
        warr, twarr = WORDS_ARR, WORDS_ARR_BIG
    else:
        raise ValueError(f'select={select}')
    words, scores = test_words(warr, twarr)
    words = iarr2str(words)
    scores = np.around(scores, 2)
    scores[scores >= 100] = np.around(scores[scores >= 100])

    nw = len(words)
    m = 10
    if nw <= 2*m or ret_full:
        ii = np.arange(nw)
    else:
        ii1 = np.arange(m)
        ii2 = np.around(np.linspace(m-1, nw-1, m+1)[1:]).astype(int)
        ii = np.concatenate((ii1, ii2))
    wlist = [(words[i], scores[i]) for i in ii]
    if print_:
        print('First word rankings.\n'
              f'(Selected from {nw} against dictionary {len(warr)})\n'
              'Rank Word   Number remaining):')
        for i in ii:
            print(f'{i+1:4d} {words[i]:<6s} {scores[i]:g}')
    return wlist

#%%
if __name__ == '__main__':

    init('nl')

    if 0:
        # This is slow; run this with F9 only.
        find_optimal_first_word(select=1)
    if 0:
        # This is very slow; run this with F9 only.
        find_optimal_first_word(select=2)

    if 0:
        print('Demo: how it would find words:')
        for w in WORDS_ARR[::len(WORDS_ARR)//5]:
            count_tries(w, 'tenor')
    if 1:
        print('Demo: playing')
        twords = 'toren/gebak' # words as tried
        pattern = '...e.' # green letters (position correct)
        badpos = '..r../ge.a.' # yellow letters (wrong position)
        print(f'Tried: {twords}; hits: {pattern}; bad position: {badpos}')
        get_next_tword(pattern, badpos, twords)
        match_hints(pattern, badpos, twords)