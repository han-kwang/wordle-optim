#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 12:54:43 2022

@author: hk_nien
"""

import re
import numpy as np
import matplotlib.pyplot as plt


ALPHABET ='abcdefghijklmnopqrstuvwxyz'
WORDS = None  # list, all eligible 5-letter words
LFREQ = None  # dict, key: letter -> value: fraction of words containing that letter
LFPOS = None  # array (i_alphabet, j_pos) -> fraction


#%%
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

def get5words_en():
    return _get5words(
        ['/usr/share/dict/american-english', '/usr/share/dict/british-english'],
        blacklist={'aires', 'aries', 'bligh'}
        )

def get5words_en_2(n=2700):
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




def get5words_nl():
    return _get5words(
        ['woordle-nl.txt'],
        blacklist={
            'aties','olink', 'molin', 'limon', 'leoni', 'pilon',
            'budak', 'bedum', 'dumps'
            }
        )


def getletfreq():
    """Return LFREQ"""

    lcount = []
    for let in ALPHABET:
        lcount.append((len([w for w in WORDS if let in w]), let))
    lfreq = {let:num/len(WORDS) for num, let in sorted(lcount, reverse=True)}
    return lfreq


def get_letfreq_pos():
    """Get letter frequencies by position."""
    lfpos = {}
    wlen = 5
    na = len(ALPHABET)
    count = np.zeros((na, wlen), dtype=int)
    for i in range(wlen):
        ltrs = [w[i] for w in WORDS]
        for ia, let in enumerate(ALPHABET):
            count[(ia, i)] = ltrs.count(let)
    lfpos = count / count.sum(axis=1).reshape(-1, 1)
    return lfpos


def find_optimal_word(prev_words, tweaks=(0.1, 0.2, -0.5), ntop=100, verbose=True):
    """Find highest scoring word given previous word tries.

    - prev_words: list of previous words
    - tweaks: tuple of:

        - bonus for letter position
        - bonus for repeated letter in different position
        - penalty for repeated letter in same position

    - ntop: how many words to apply (slow) tweaks to.
    - verbose: print extra info.
    """
    prev_letters = set(''.join(prev_words))
    letters = ''.join(sorted(set(ALPHABET) - prev_letters))
    wscores = {k: 0 for k in WORDS}
    for let in letters:
        for w in WORDS:
            if let in w:
                wscores[w] += LFREQ[let]
    wscores = {
        k:wscores[k]
        for k in sorted(wscores, key=lambda x:wscores[x], reverse=True)
        }

    topwords = [w for _, w in zip(range(ntop), wscores.keys())]
    # Bonus based on position
    for w in topwords:
        for i, let in enumerate(w):
            pbonus = LFPOS[ord(let)-ord('a'), i]
            lbonus = LFREQ[let]
            wscores[w] += tweaks[0]*pbonus*lbonus

    # Small bonus for previously seen letters on different positions
    # (more bonus for high-scoring letters).
    for iw, w in enumerate(topwords):
        # if w == 'blimp':
        #     breakpoint()
        for i, let in enumerate(w):
            if let not in prev_letters:
                continue
            for pw in prev_words:
                if pw[i] == let:
                    break
            else:
                bonus = tweaks[1]*LFREQ[let]
                if iw < 10 and verbose:
                    print(f'{w}: bonus {bonus:.2f}')
                wscores[w] *= 1 + bonus
    # penalty for double letters
    for w in topwords:
        if len(set(w)) < len(w):
            wscores[w] *= 0.8

    # give penalty to words having previously seen letters in the same
    # place.
    exps = []
    for i, ll in enumerate(zip(*prev_words)):
        exp = ('.'*i) + f'[{"".join(ll)}]' + ('.'*(4-i)) + '$'
        exps.append(re.compile(exp))
    for iw, w in enumerate(topwords):
        for exp in exps:
            if re.match(exp, w):
                if iw < 5 and verbose:
                    print(f'{w}: penalty 0.2')
                wscores[w] *= (1 + tweaks[2])

    # re-sort topwords
    topwords = sorted(topwords, key=lambda x: -wscores[x])

    tws_w_score = [
        f'{w} ({wscores[w]*100:.0f})'
        for w in topwords
        ]

    if verbose:
        print(f'For {letters}: {", ".join(tws_w_score[:5])}')
    return topwords[0]


def match_1word(word, pwords):
    """Match probe words against word.

    Return:

    - hits1: set of letters that match
    - hits2: set of matched (letter, pos) tuples.
    - wlets: unique letters in the word
    - badlets: set of bad letters
    - badlets_p: list of bad letter sets by position
    """
    hits1 = set()  # letters
    hits2 = set()  # (letter, pos) tuples
    #hits2u = set() # letters in correct position
    wlets = set(word)
    plets = set()
    badlets_p = [set() for _ in range(len(word))]
    badlets = set()

    for pw in pwords:
        hits1.update(wlets.intersection(set(pw)))
        plets.update(set(pw))
        badlets.update(set(pw) - wlets)
        for i, let in enumerate(word):
            if pw[i] == let:
                hits2.update([(let, i)])
            else:
                badlets_p[i].add(pw[i])

    return hits1, hits2, wlets, badlets, badlets_p



def evaluate_1word_fast(word, pwords, verbose=False, w1=0.5, ret_full=False):
    """Try probe words on word, return score 0 <= score <= 100

    Heuristic method.

    Parameters:

    - word: 1 word
    - pwords: list of probe words
    - verbose: True to print details.
    - w1: weight for hits (0<=w1<=1); weigth for position hits is 1-w1.
    - ret_full: True to return both metrics.

    Return:

    - score
    - (for ret_full=True) fraction of letters found.
    - (for ret_full=True) fraction of letters found on correct position.
    """
    hits1, hits2, wlets, _, _ = match_1word(word, pwords)

    score = w1*len(hits1)/len(wlets) + (1-w1)*len(hits2)/len(word)
    score = np.around(score*100, 1)
    if verbose:
        print(
            f'  {word}: {len(hits1)}/{len(wlets)} '
            f'{len(hits2)}/{len(word)} '
            f'{score:.1f}'
            )

    if ret_full:
        return score, len(hits1)/len(wlets), len(hits2)/len(word)
    return score


def evaluate_1word_slow(word, pwords, verbose=False):
    """Try probe words on word, return score 0 <= score <= 100

    Score based on % of all words that can be ruled out.

    Parameters:

    - word: 1 word
    - pwords: list of probe words
    - verbose: True to print details.

    Return: score
    """
    hits1, hits2, wlets, badlets, badlets_p = match_1word(word, pwords)

    words = set(WORDS)
    nws = [len(words)]
    if hits2:
        # filter based on known letters
        exp = ['.']*len(word)
        for let, i in hits2:
            exp[i] = let
        exp = re.compile(''.join(exp))
        words = [w for w in words if re.match(exp, w)]
    nws.append(len(words))

    # filter based on occurence of bad letters
    exp = ''.join(badlets)
    exp = re.compile(f'[{exp}]')
    words = [w for w in words if not re.search(exp, w)]
    nws.append(len(words))

    # filter based on occurence of good letters
    words1 = []
    for w in words:
        for let in hits1:
            if not let in w:
                break
        else:
            words1.append(w)

    words = words1
    nws.append(len(words))
    score = max(0, 100 - (nws[-1]-1)*10)
    nws_str = ",".join([str(x) for x in nws])
    if verbose:
        print(f'{word}: {nws_str} - score {score}')
    return score


def evaluate_1word(word, pwords, verbose=False, w1=0.5, smethod='fast',
                   ret_full=False):

    if smethod == 'fast':
        return evaluate_1word_fast(word, pwords, verbose=verbose, w1=w1,
                                   ret_full=ret_full)
    elif smethod == 'slow':
        return evaluate_1word_slow(word, pwords, verbose=verbose)
    else:
        raise ValueError(f'smethod={smethod!r}')


def evaluate_words(pwords, w1=0.5, smethod='fast', speedup=2):
    """Evaluate list of probe words on corpus; return mean score.

    smethod can be 'slow' or 'fast'
    """
    scores = [
        evaluate_1word(w, pwords, w1=w1, smethod=smethod)
        for w in WORDS[::speedup]
        ]

    return np.mean(scores)


def evaluate_words_verbose(pwords, w1=0.5, smethod='fast', nsamp=10):

    score = evaluate_words(pwords, w1=w1, smethod=smethod)
    print(f'  Probe words: {", ".join(pwords)}')

    print(f'Performance: {score:.1f}')
    print('Sample:')
    results = []
    for i, w in enumerate(WORDS[::len(WORDS)//nsamp]):
        r = evaluate_1word(w, pwords, w1=w1, verbose=(i < 10), smethod=smethod,
                           ret_full=True)
        results.append(r)
    if i > 10:
        print('(Truncated after 10)')
    if smethod == 'slow':
        smean = np.mean(results)
        print(f'Mean score: {smean:.1f}')
    else:
        smean = np.array(results).mean(axis=0)  # columns: score, fhit, fpos
        wlen = len(pwords[0])
        print(f'Mean: score {smean[0]:.1f}, '
              f'letters found {smean[1]*100:.0f}%, '
              f'positions found {smean[2]*wlen:.2f}/{wlen}.')


    return score

def evaluate_hyper(tweaks=(0.25, -0.2, -0.5), num=5, w1=0.5, verbose=True,
                   smethod='fast'):
    """smethod: 'slow' or 'fast'

    Return:

    - score
    - probe_words
    """

    wseen = []
    for _ in range(num):
        next_word = find_optimal_word(
            wseen, verbose=False,
            tweaks=tweaks
            )
        wseen.append(next_word)

    if verbose:
        print(f'\n** tweaks={tweaks}, num={num}, w1={w1}, smethod={smethod!r}')
        score = evaluate_words_verbose(wseen, w1=w1, smethod=smethod)
    else:
        score = evaluate_words(wseen, w1=w1, smethod=smethod)
    return score, wseen

def scan_hyper(num=5, w1=0.5, tweak2=-0.2, plot=True, smethod='fast',
               t0range=(0, 0.4, 9), t1range=(0, 0.4, 9)):
    """Plot and return optimal kwargs."""
    scores = []
    t0s = np.linspace(*t0range)
    t1s = np.linspace(*t1range)
    scores = np.zeros((len(t0s), len(t1s)))
    print('scan ...', end='', flush=True)
    ndone = 0
    for i0, t0 in enumerate(t0s):
        for i1, t1 in enumerate(t1s):
            score, _ = evaluate_hyper(
                tweaks=(t0, t1, tweak2), num=num, w1=w1, verbose=False,
                smethod=smethod
                )
            scores[i0, i1] = score
            ndone += 1
            print(f'\rscan {ndone}/{scores.size}', end='', flush=True)
    print(' done.')
    i0, j0 = np.unravel_index(np.argmax(scores), scores.shape)
    opt_tweak = (t0s[i0], t1s[j0], tweak2)

    if plot:

        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(len(t1s)))
        ax.set_xticklabels([f'{t1:.2f}' for t1 in t1s])
        ax.set_xlabel('tweak1: different position bonus')
        ax.set_yticks(np.arange(len(t0s)))
        ax.set_yticklabels([f'{t0:.2f}' for t0 in t0s])
        ax.set_ylabel('tweak0: position bonus')

        cm = ax.matshow(scores)
        fig.colorbar(cm)
        fig.show()

    return dict(tweaks=opt_tweak, w1=w1, num=num, smethod=smethod)


def analyze_wordle_stats():
    """Original wordle statistics"""
    picks = (
        'panic,tangy,abbey,favor,drink,query,gorge,crank,slump,banal,tiger,'
        'siege,truss,boost,rebus'
        ).split(',')
    global WORDS
    WORDS = get5words_en_2(5000) # Large corpus, sorted.
    poss = []
    notfound = set()
    for w in picks:
        try:
            poss.append(WORDS.index(w))
        except ValueError:
            notfound.add(w)
    poss = np.array(poss)
    print(f'Word positions in corpus: {poss}')
    m, s = poss.mean(), poss.std()
    sm = 1/np.sqrt(3)
    print(f'Mean: {m:.0f}, std={s:.0f}, ratio {s/m:.2f}'
          f' (expected for flat-top: {sm:.2f}')
    print(f'not found: {notfound}')


def run_nl(hyperscan=False, numw=4, w1=0.7):
    """NL:

    Best for either hits or pos hits:
        ['toren', 'balie', 'drugs', 'gemak', 'schop']

    """

    global WORDS, LFREQ, LFPOS
    WORDS = get5words_nl()
    LFREQ = getletfreq()
    LFPOS = get_letfreq_pos()
    plt.close('all')


    kwargs1 = dict(num=numw, w1=w1, smethod='fast')

    if hyperscan:
        kwargs2 = scan_hyper(
            **kwargs1,
            t0range=(0, 0.4, 9), t1range=(-0.3, 0.4, 15), tweak2=-0.5
            )
    else:
        # from a previous run
        kwargs2 = {**kwargs1, 'tweaks': (0.25, -0.15, -0.5)}

    _, pwords = evaluate_hyper(**kwargs2, verbose=False)
    for _ in range(5 - numw):
       pwords.append(
           find_optimal_word(pwords, tweaks=kwargs2['tweaks'], verbose=False)
           )
    print(f'Optimized for {numw} words: {repr(pwords)}')
    print(f'tweaks={kwargs2["tweaks"]}')
    for inum in (4, 5):
        evaluate_words_verbose(pwords[:inum], w1=kwargs2['w1'], smethod='fast')
    print('\n\n')
    for w in pwords:
        print(f'     {w}')
    print()
    return pwords



def run_en(hyperscan=False, n_corpus=2700, numw=5, w1=0.7):
    """Run for English.

    Best for hit rate: ['raise', 'clout', 'nymph', 'bowed', 'kings']
    Best for pos hits: ['raise', 'count', 'piled', 'shaky', 'began']
    Manually tweaked: ['cares', 'point', 'bulky', 'width', 'gnome']

    (Manual tweak: allow 'c' in first word for better position hits and
     very little penalty on letter hit rate for the first 4 words together.)
    """
    global WORDS, LFREQ, LFPOS
    WORDS = get5words_en_2(n_corpus)
    LFREQ = getletfreq()
    LFPOS = get_letfreq_pos()
    plt.close('all')

    kwargs1 = dict(num=numw, w1=w1, smethod='fast')

    if hyperscan:
        kwargs2 = scan_hyper(
            **kwargs1,
            t0range=(0, 0.4, 9), t1range=(-0.3, 0.4, 15), tweak2=-0.5
            )
    else:
        # from a previous run
        kwargs2 = {**kwargs1, 'tweaks': (0.25, -0.15, -0.5)}

    _, pwords = evaluate_hyper(**kwargs2, verbose=False)
    for _ in range(5 - numw):
       pwords.append(
           find_optimal_word(pwords, tweaks=kwargs2['tweaks'], verbose=False)
           )
    print(f'Optimized for {numw} words: {repr(pwords)}')
    print(f'tweaks={kwargs2["tweaks"]}')
    for inum in (4, 5):
        evaluate_words_verbose(pwords[:inum], w1=kwargs2['w1'], smethod='fast')
    print('\n\n')
    for w in pwords:
        print(f'     {w}')
    print()
    return pwords

def search(regexp, good_letters, tried_letters):
    """Search for words.

    - regexp: like '..a.t'
    - goodletters: str with letters that must be present.
    - triedletters: str with all letters as tried.
    """
    good_letters = set(good_letters)
    for let in re.sub(r'\[.*?\]', '', regexp):
        if let != '.':
            good_letters.add(let)
    regexp = re.compile(regexp)

    bad_letters = ''.join(set(tried_letters) - set(good_letters))
    badexp = re.compile(f'[{bad_letters}]')
    words = [
        w
        for w in WORDS
        if re.match(regexp, w) and not re.search(badexp, w)
        ]
    words2 = []

    for w in words:
        for let in good_letters:
            if let not in w:
                break
        else:
            words2.append(w)
    print(', '.join(words2[:20]))

if __name__ == '__main__':
    # run_en(hyperscan=True, n_corpus=1200)
    run_nl(hyperscan=True,w1=0.5)

