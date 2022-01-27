#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wordle strategy 2 - brute force

For online game:

- https://www.powerlanguage.co.uk/wordle/ (English, ~2700 dictionary)
- https://hellowordl.net/ (English, ~5000 dictionary?)
- https://woordle.nl/ (Dutch, 860 solutions but recognizes ~5500 words).

It's encapsulated in the Wordle class.

Finding optimal words to try, given hints.

Created on Sat Jan 22 21:25:10 2022 // author: hk_nien
"""
from time import time
import re
import sys
from pathlib import Path
from multiprocessing import Pool
from threadpoolctl import threadpool_limits
import numpy as np

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


# Words that occur in generic word lists that are not recognized.
BLACKLISTS = {
    # For original Wordle
    'en-2700': {'alton', 'ethan', 'aires', 'aries', 'bligh'}
    }

_FIRST_WORDS = {
    # Dataset name -> (word, n_expected)
    # in comments: other words and calculation time (2 cores, 4 threads)
    'en': ('raise', 61.0),  # arise (63.7), irate (63.8) / 22min
    'en-2700': ('raise', 57.9),  # didn't update
    'en-full': ('roate', 60.4), # raise (61.0) raile (61.3), soare (62.5) / 1h03m
    'en-hello': ('raise', 95.4),
    'nl': ('tenor', 21.6),
}


def _load_wlist(fname, wlen=5, maxnum=99999, iarr=False, blacklist_key=None):
    """Load word list (list of str or iarray) from file."""
    if blacklist_key is None or blacklist_key not in BLACKLISTS:
        blacklist = set()
    else:
        blacklist = BLACKLISTS[blacklist_key]
    exp = re.compile(f'[a-zA-Z]{{{wlen}}}$', re.I)
    with open(fname) as f:
        wlist = [
            w[:-1].lower()
            for w in f
            if exp.match(w)
            ]
    wlist = [w for w in wlist if w not in blacklist]
    wlist = wlist[:maxnum]
    if iarr:
        wlist = str2iarr(wlist)
    return wlist

# Data for workers is stored here.
_WORKER_PERSISTENT = {}

def _test_1word(tword):
    """Return mean number of matches against all words. For multiprocessing worker.

    Parameter:

    - tword: word array (n,) int to use as a try word.

    For use in Pool.

    Uses _WORKER_PERSISTENT keys:

    - 'warr': words dictionary to match against; (n, 5) int.
    - 'alphabet': array of alphabet (typically length 26).
    """
    warr = _WORKER_PERSISTENT['warr']
    wordle = _WORKER_PERSISTENT['wordle_obj']
    alphabet = wordle.alphabet
    patterns, badposs, badletss = wordle.gethints_all(tword, warr)
    nmatch = np.zeros(len(warr))

    for j, (pat, bpos, blets) in enumerate(zip(patterns, badposs, badletss)):
        blets = alphabet[blets]
        bpos = [(k, let) for k, let in enumerate(bpos) if let >= 0]
        bpos = np.array(bpos, dtype=np.int16).reshape(-1, 2)
        matches = wordle.imatch_hints(pat, bpos, blets, warr=warr)
        nmatch[j] = len(matches)
    return nmatch.mean()


class Wordle:
    """Wordle player/hinter/searcher.

    Strings are handled as numpy int16 arrays; positive values for
    unicode/ascii characters and value -1 for 'undefined/unknown'.

    Word lists are 'a' and 'b' lists. a=list of possible solutions.
    b=list of recognized words.

    Initialize with dataset, which is one of:

      - 'nl': Dutch word list (a:865, b:5541)
      - 'en': Reduced Wordle list (a:2315, b:3911)
      - 'en-hello': List for Hello Wordl (a:3500, b:4100)
      - 'en-2700': English word list (2700 words)
      - 'en-full': Original Wordle list (a:2315, b:10657)

    To get the list of dataset names, use::

        Wordle.get_datasets()

    Functions with string interface:

    - count_tries(): Return number of attempts to find the specified word.
    - find_optimal_first_word(): Find optimal first word. This is slow.
    - get_next_tword(): Suggest next word to try
    - match_hints(): Find words that match the hints.
    - gnt(), mh(): shorthand forms of get_next_tword(), match_hints()
    - test_words(): Test all vocabulary words for selectivity.
    - build_cache(): build cache file for first words.

    Functions with int array interface for internal use:

    - str2iarr(): convert strings to int arrays
    - iarr2str(): convert int array to strings
    - gethints_1iword: Get hints for try-word, given solution word.
    - gethints_all: Try a word (int array) against entire vocabulary
    - imatch_hints: Match vocabulary against hints

    Attributes:

    - self.alphabet: int16 array with alphabet, typically length 26 (lowercase).
      Must be consecutive (no gaps).
    - self.warr_a: puzzle word array, shape (n, 5)
    - self.warr_b: recognized word array, shape (nn, 5).
    - first_word: optimal first word
    - first_word_
    - dataset: dataset used for initialization (do not modify).
    - cache: dictionary with keys '<firstword> <pattern> <badpos>',
      values (num_match, next_word, num_match_next)
    """
    @classmethod
    def get_datasets(cls):
        """Return list of supported dataset names."""
        return ['nl', 'en-2700', 'en-full', 'en', 'en-hello']

    def __init__(self, dataset='en'):
        """Initialize, see class doc."""
        self.alphabet = np.arange(26, dtype=np.int16) + ord('a')
        self.dataset = str(dataset)
        self.first_word, self.first_word_expected = _FIRST_WORDS[dataset]
        if dataset == 'nl':
            self.warr_b = _load_wlist('data/woordle-nl-b.txt', iarr=True)
            self.warr_a = _load_wlist('data/woordle-nl-a.txt', iarr=True)
            self.first_word = 'tenor'
        elif dataset == 'en-2700':
            self.warr_a = _load_wlist(
                'data/wordlist-en-freq.txt', maxnum=2700, iarr=True,
                blacklist_key=dataset
                )
            self.warr_b = self.warr_a
            self.first_word = 'raise'
        elif dataset == 'en-full':
            self.warr_a = _load_wlist('data/wordle-en-a.txt', iarr=True)
            self.warr_b = _load_wlist('data/wordle-en-b.txt', iarr=True)
        elif dataset == 'en':
            warr_a = _load_wlist('data/wordle-en-a.txt', iarr=False)
            warr_b1 = _load_wlist('data/wordle-en-b.txt', iarr=False)
            warr_b2 = _load_wlist('data/wordlist-en-freq.txt', iarr=False)
            warr_b2 = set(warr_a).union(set(warr_b1).intersection(warr_b2))
            self.warr_a = str2iarr(warr_a)
            self.warr_b = str2iarr(sorted(warr_b2))
        elif dataset == 'en-hello':
            self.warr_a = _load_wlist('data/wordlist-en-freq.txt', maxnum=3500, iarr=True)
            self.warr_b = _load_wlist('data/wordlist-en-freq.txt', maxnum=4200, iarr=True)
            self.warr_a = self.warr_b
            self.first_word = 'raise'
        else:
            raise ValueError(f'dataset={dataset!r} / try get_all() method.')
        self.cache = self._load_cache()


    def _apply_blacklist(self, words):
        """Apply to list of str and return new list of str."""
        if self.dataset not in BLACKLISTS:
            return words
        bl = BLACKLISTS[self.dataset]
        words_2 = [w for w in words if w not in bl]
        return words_2

    def __repr__(self):
        cn = self.__class__.__name__
        na, nb = len(self.warr_a), len(self.warr_b)
        return f'<{cn}: {self.dataset!r}, num_a={na}, num_b={nb}>'

    def _load_cache(self):
        """Load and return cache dict."""
        cfpath = self._get_cache_fpath()
        cache = {}
        if not cfpath.is_file():
            print(f'No cache file {cfpath}; use build_cache() to build one.')
            return cache

        with cfpath.open('r') as f:
            for lno, line in enumerate(f):
                if line == '' or line == '\n' or line[0] == '#':
                    continue
                fields = line.split()
                if len(fields) != 6:
                    raise ValueError(f'{cfpath}: parse error on line {lno+1}')
                key = f'{fields[0]} {fields[1]} {fields[2]}'
                cache[key] = (int(fields[3]), fields[4], float(fields[5]))
        return cache

    @staticmethod
    def i2s(iarr):
        """Convert int16 array to string or string list"""
        return iarr2str(iarr)

    @staticmethod
    def s2i(s):
        """Convert string or string list to int16 array."""
        return str2iarr(s)

    def imatch_hints(self, pattern, badpos, badlets, warr=None):
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
            warr_match = self.warr_a
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

        if warr_match is self.warr_a or warr_match is warr:
            warr_match = warr_match.copy()

        return warr_match

    def gethints_all(self, iword, warr=None):
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
            warr = self.warr_a
        else:
            assert warr.dtype==np.int16
        warr = warr.copy()  # we're going to replace matched letters by -1
        nw, wsize = warr.shape

        patterns = np.full((nw, wsize), np.int16(-1))
        badposs = patterns.copy()
        badletss = np.full((nw, len(self.alphabet)), False)
        irange = np.arange(nw)
        masks = []
        for i, let in enumerate(iword):
            mask = (warr[:, i] == let)
            patterns[mask, i] = let
            warr[mask, i] = -1 # this is to prevent double counting
            masks.append(mask)

        for i, (let, mask) in enumerate(zip(iword, masks)):
            # Correct letters in wrong positions.
            # Recurring letters in test words are tricky.
            #            tword solution pattern badpos batlets
            #   CORRECT  foo   zob      .o.     ...    f
            #   WRONG    foo   zob      .o.     ..o    f
            #   WRONG    foo   zob      .o.     ...    f,o
            mask2 = np.any(warr[~mask, :] == let, axis=1)
            ii = irange[~mask][mask2]
            badposs[ii, i] = let
            j = let - self.alphabet[0]
            mask3 = ~np.any(warr == let, axis=1) & ~np.any(patterns == let, axis=1)
            badletss[mask3, j] = True

        return patterns, badposs, badletss

    def gethints_1iword(self, tword, sol_word):
        """Get hints for try-word, given solution word.

        Return:

        - pattern, badpos, badlets as can be fed into imatch_hints().
        """
        pat, badpos, badlet = [
            # convert to 1D arrays
            x[0]
            for x in self.gethints_all(tword, warr=sol_word.reshape(1, -1))
            ]
        badpos = [
            (i, let)
            for i, let in enumerate(badpos)
            if let > 0
            ]
        badlet = self.alphabet[badlet]
        return pat, badpos, badlet

    def test_1word(self, tword, warr=None):
        """Get expected number of matches for a try word.

        Parameters:

        - tword: try word (str or int16 array).
        - warr: vocabulary array (int array); default self.warr_a.

        Return:

        - mean_nmatch: float, number of matches expected.
        """
        if isinstance(tword, str):
            tword = str2iarr(tword)
        if warr is None:
            warr = self.warr_a
        _WORKER_PERSISTENT['warr'] = warr
        _WORKER_PERSISTENT['wordle_obj'] = self
        mnm = _test_1word(tword)
        for k in ['warr', 'wordle_obj']:
            del _WORKER_PERSISTENT[k]
        return np.around(mnm, 3)

    def get_optimal_first_word(self):
        """Find optimal first word from self.warr_b. Warning: slow!

        This will update self.first_word and self.first_n_expect.
        """
        iwords, nex = self.test_words(self.warr_a, self.warr_b)
        self.first_word = iarr2str(iwords[0])
        self.first_n_expect = np.around(nex, 3)

    def test_words(self, warr, twarr=None, nshow=3, pri_time=2):
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
        _WORKER_PERSISTENT['warr'] = warr
        _WORKER_PERSISTENT['wordle_obj'] = self

        mean_nmatchs = []
        if sys.platform == 'linux' and len(twarr) >= 3:
            # Parallellized
            with threadpool_limits(limits=1), Pool() as pool:
                asyncs = [
                    pool.apply_async(_test_1word, (tw,))
                    for tw in twarr
                    ]
                for i, a in enumerate(asyncs):
                    mean_nmatchs.append(a.get())
                    tm = time()
                    if tm - tm_start > pri_time and (tm - tm_prev > 1 or i == ntw-1):
                        print(f'\rtest_words {i+1}/{ntw}...', end='')
                        tm_prev = tm
        else:
            # non-parallel operation
            for i, tw in enumerate(twarr):
                mean_nmatchs.append(_test_1word(tw))
                tm = time()
                if tm - tm_prev > 1 and tm - tm_start > pri_time:
                    print(f'\rtest_words {i+1}/{ntw}...', end='')
                    tm_prev = tm

        for k in ['warr', 'wordle_obj']:
            del _WORKER_PERSISTENT[k]

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

    def _get_best_tword(self, warr):
        """Get best test word (and # remaining) to select from specified list."""
        if len(warr) <= 2:
            tword = warr[0]
            n = max(1, len(warr)-1)
        else:
            twords1, scores = self.test_words(warr, self.warr_a, nshow=0, pri_time=3)
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

    def color_str_try(self, secret_word, try_word):
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

    def _get_cache(self, ipat, ibadpos, itword):
        """Get cache entry (n_match, iword, n_expected) from int word, pattern, badpos;
        return None if no hit.
        """
        tword = iarr2str(itword)
        pattern = iarr2str(ipat)
        bp = ['.'] * len(tword)
        for i, ilet in ibadpos:
            if bp[i] != '.':
                return None
            bp[i] = chr(ilet)
        bp = ''.join(bp)
        key = f'{tword} {pattern} {bp}'
        if key in self.cache:
            nm, w, nex = self.cache[key]
            return (nm, str2iarr(w), nex)
        return None

    def count_tries(self, secret_word, first_tword=None, maxtries=6, warr=None, verbose=True):
        """Return number of attempts to find the specified word.

        Parameters:

        - secret_word: secret word (int array or str)
        - first_tword: first try word (int array or str) (optional)
        - maxtries: ...
        - warr: all words to match against; (n, 5) int array.
        - verbose: True to show the words and statistics.

        Return maxtries+1 if no succes.

        This includes the first word as provided and the word as found.

        Output for verbose=True:

        - test words, color-coded
        - (n_expected / n_match): number of remaining words that match the
          hints as expected and actual value.
        """
        tword = self.first_word if first_tword is None else first_tword
        if isinstance(tword, str):
            tword = str2iarr(tword)
        if isinstance(secret_word, str):
            secret_word = str2iarr(secret_word)
        if warr is None:
            warr = self.warr_a

        wlen = len(tword)
        pat = np.full(wlen, np.int16(-1))
        badpos = set()
        badlet = set()
        twords_stats = []  # lists [tword, n_before, n_expected, n_after]
        is_solved = False
        n_expected = '?'

        for itry in range(maxtries+1):
            # At the beginning of each iteration:
            # - tword: next test word to apply
            # - twords_stats: does not yet contain this test word
            # - hints: previously existing hints (pat, badpos, badlet)
            # - warr: remaining candidate words after applying previous hints
            # - n_expected: number of remaining words expected after applying
            #   the next test word.

            n_before = len(warr)
            # Try the test word, merge hints
            pat1, badpos1, badlet1 = self.gethints_1iword(tword, secret_word)
            for i, let in enumerate(pat1):
                if let > 0:
                    pat[i] = let
            for i, let in badpos1:
                badpos.add((i, let))
            for let in badlet1:
                badlet.add(let)

            # Solved?
            if np.all(pat1 == tword):
                twords_stats.append([tword, n_before, n_expected, 1])
                is_solved = True
                break

            # get words that match the hints
            warr = self.imatch_hints(pat, badpos, badlet, warr)
            twords_stats.append([tword, n_before, n_expected, len(warr)])
            if len(warr) == 0:
                break

            # Find next test word from cache or from brute force
            if itry == 0:
                nb_tw_ne = self._get_cache(pat1, badpos1, tword)
            else:
                nb_tw_ne = None
            if nb_tw_ne is None:
                tword, n_expected = self._get_best_tword(warr)
            else:
                tword, n_expected = nb_tw_ne[1:]
            n_expected = f'{n_expected:.3g}'
            # end of loop

        if verbose:
            summary = []
            for tword, _, n_expected, n_after in twords_stats:
                tword = self.color_str_try(iarr2str(secret_word), iarr2str(tword))
                txt = f'{tword} ({n_expected}/{n_after})'
                summary.append(txt)
            if not is_solved:
                msg = '\033[31;1m  (Not found)\033[0m'
            else:
                msg = ''
            summary = ' → '.join(summary)
            print(f'{iarr2str(secret_word)}: {summary}{msg}')

    def match_hints(self, pattern, badpos, twords, mode='print'):
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
        warr = self.imatch_hints(pattern, badpos1, badlet)
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

    def get_next_tword(self, pattern, badpos, twords):
        """Suggest next word to try (human-friendly interface).

        Parameters:

        - pattern: e.g. '..a.n' (green letters)
        - badpos: e.g. 'z..../bl.../....k' (letters at green positions)
        - twords: e.g. 'zuster/bloem/draak'
        """
        cache_key = f'{twords} {pattern} {badpos}'
        if cache_key in self.cache:
            nm, tword, nmn = self.cache[cache_key]
        else:
            try:
                warr = self.match_hints(pattern, badpos, twords, mode='return')
            except ValueError as e:
                if 'Inconsistent hints' in e.args[0]:
                    print(f'Error: {e.args[0]}')
                    return
                raise
            nm = len(warr)
            # Get optimal next word
            if nm > 0:
                if nm*len(self.warr_a) > 30000:
                    print(f'({len(warr)} words match the hints)')
                tword, nmn = self._get_best_tword(warr)
                tword = iarr2str(tword)
                self.cache[cache_key] = (nm, tword, nmn)

        if nm == 0:
            print('No match!')
        else:
            print(f'Best next word: {tword} ({nmn:.4g}/{nm})')

    def play_ai(self, first_word=None, maxtries=6):
        """Interactive play against human or website.
        """
        tword = self.first_word if first_word is None else first_word
        wlen = len(tword)
        print('Example response: .a.T. -> a has wrong position, T is correct.')
        ipattern = np.full(wlen, np.int16(-1))
        ibadpos = set()
        ibadlet = set()
        warr = self.warr_a
        itry = 0
        while True:
            # get/parse user input
            try:
                r = input(f'Try "{tword}", what is the response? >')
            except EOFError:
                print('  Aborted.')
                break
            if len(r) != wlen:
                print(f'  Wrong length, expected {wlen}. Try again.')
                continue
            if r == tword.upper():
                print('  Gotcha!')
                break
            if r == tword:
                print(f'Did you mean {tword.upper()!r} rather than {tword!r}?')
                continue
            elif itry == maxtries - 1:
                print('  Game over')
                break
            good_response = True
            for (twl, rl) in zip(tword, r):
                if rl not in ('.', twl, twl.upper()):
                    print(f"This response doesn't match {tword!r}")
                    good_response = False
                    break
            if not good_response:
                continue

            for i, let in enumerate(r):
                ilet = ord(let.lower())
                if let.isupper():
                    ipattern[i] = ilet
                else:
                    if let == '.':
                        ibadlet.add(ord(tword[i]))
                    else:
                        ibadpos.add((i, ilet))
            # get next test word
            warr = self.imatch_hints(ipattern, ibadpos, ibadlet, warr)
            print(
                f'  Remaining: {len(warr)}: {", ".join(iarr2str(warr[:7]))}'
                f'{", ..." if len(warr) > 7 else "."}'
                )

            if len(warr) == 0:
                print('  Giving up!')
                break
            if itry == 0:
                nb_tw_ne = self._get_cache(ipattern, ibadpos, str2iarr(tword))
                tword = None if nb_tw_ne is None else nb_tw_ne[1]
            else:
                tword = None
            if tword is None:
                tword, _ = self._get_best_tword(warr)
            tword = iarr2str(tword)
            itry += 1

    def gnt(self, hints):
        """Shorthand notation for get_next_tword() for interactive use.

        Parameters:

        - hints: string '<pattern> <badpos> <twords>': like for
          get_next_tword(), but as single str with spaces.
          Example: '...an z..../bl... zuster/bloem'.
        """
        self.get_next_tword(*hints.split(' ', maxsplit=3))

    def mh(self, hints):
        """Shorthand notation for match_hints().

        Parameters:

        - hints: string '<pattern> <badpos> <twords>': like for
          get_next_tword(), but as single str with spaces.
          Example: '...an z..../bl... zuster/bloem'.
        """
        self.match_hints(*hints.split(' ', maxsplit=3))

    def find_optimal_first_word(self, select=1, ret_full=False, print_=True):
        """Find optimal first word. This is slow.

        Parameters:

        - select: indicate dictionary size to work with; 0: small for testing;
          1: select words from self.warr_a; 2: select from self.warr_b.
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
            warr, twarr = self.warr_a[::5], self.warr_a[::5]
        elif select == 1:
            warr, twarr = self.warr_a, self.warr_a
        elif select == 2:
            warr, twarr = self.warr_a, self.warr_b
        else:
            raise ValueError(f'select={select}')
        words, scores = self.test_words(warr, twarr)
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

    def _get_cache_fpath(self):
        """Return cache file path for current dataset"""
        cpath = Path(__file__).resolve().parent / 'cache'
        dpath = Path(__file__).resolve().parent / 'data'
        if not cpath.exists():
            cpath.mkdir()
            print(f'Created {cpath}')
        cfpath = cpath / f'cache-{self.dataset}.txt'

        if not cfpath.is_file():
            cfpath_dist = (dpath / cfpath.name)
            if cfpath_dist.is_file():
                print(f'No cache; copying default cache to {cfpath}')
                with cfpath_dist.open() as f1, cfpath.open('w') as f2:
                    f2.write(f1.read())

        return cfpath

    def build_cache(self, num=99, start=0):
        """Build cache for given first word (str)

        Set num to small value for testing.
        """
        wlen = self.warr_a.shape[1]
        fw = self.first_word
        assert len(fw) == wlen
        def mkhint(*args):
            h = ['.'] * wlen
            for i, let in args:
                h[i] = let
            return ''.join(h)
        hlistY = []
        hlistG = []
        hlistYY = []
        hlistGY = []
        hlistGG = []
        hlistYYY = []
        # 1 yellow letter
        for i, let in enumerate(fw):
            hlistY.append((mkhint(), mkhint((i, let))))
            hlistG.append((mkhint((i, let)), mkhint()))
        for i, let in enumerate(fw[:-1]):
            for j in range(i+1, wlen):
                hlistYY.append((mkhint(), mkhint((i, let), (j, fw[j]))))
                hlistGG.append((mkhint((i, let), (j, fw[j])), mkhint()))
                for k in range(j+1, wlen):
                    hlistYYY.append((mkhint(), mkhint((i, let), (j, fw[j]), (k, fw[k]))))
        for i, let in enumerate(fw):
            for j in range(wlen):
                if i != j:
                    hlistGY.append(
                        (mkhint((i, let)), mkhint((j, fw[j])))
                        )
        hintlist = ([(mkhint(), mkhint())] + hlistY + hlistG
                    + hlistYY + hlistGY + hlistGG + hlistYYY
                    )
        cfpath = self._get_cache_fpath()
        cfpath_tmp = Path(f'{cfpath}.tmp')
        with cfpath_tmp.open('w') as f:
            f.write(
                '# Cache for dataset={self.dataset}\n'
                '# first_word, hit_pattern, badpos, nmatch, nextword, nmatch_next\n'
                )
            print(f'Scanning optimal words (first_word={fw}, dataset={self.dataset})')
            for pattern, badpos in hintlist[start:num]:
                warr = self.match_hints(pattern, badpos, fw, mode='return')
                nw = len(warr)
                print(f'Trying {pattern} {badpos} {fw} {nw}:')
                # Get optimal next word
                if len(warr) == 0:
                    print('no match!')
                    line = f'# {fw} {pattern} {badpos} {nw} No_match! 0\n'
                else:
                    tword, n = self._get_best_tword(warr)
                    tword = iarr2str(tword)
                    print(f'--> {tword} {n:.3g}')
                    line = f'{fw} {pattern} {badpos} {nw} {tword} {n:.3g}\n'
                f.write(line)
                f.flush()

        if cfpath.exists():
            bakname = f'{cfpath}.bak'
            cfpath.rename(bakname)
            print(f'Renamed old cache to {bakname} .')
        cfpath_tmp.rename(cfpath)
        print(f'Wrote cache to {cfpath} .')
        self._load_cache()
