Wordle-optim: optimization for Wordle game
==========================================

Author: Han-Kwang Nienhuys (@hk_nien on Twitter).

This code is for optimizing the choice of word guesses in the
Wordle game, in particular:

- https://www.powerlanguage.co.uk/wordle/ (original English)
- https://hellowordl.net/ (English)
- https://woordle.nl/ (Dutch)
- https://qntm.org/absurdle (English; adversial version)

There must be tens of other projects doing the similar things. Well, this
is mine; I wrote it for fun without checking other approaches.

I usually run this interactively from Spyder. It uses numpy rather than
string manipulations for efficiency. In order to find the optimal first
word in the original Wordle, it needs to try 56 billion (56x10<sup>9</sup>)
combinations of secret word, first guess, and potential other secret words
matching the hints. This takes over an hour on a laptop with
2 cores, 4 threads. Multiprocessing is only supported on Linux. Contact me
if you want multiprocessing support in Windows. The optimal first words are
included, so you don't have to wait for this optimization.

The word lists for the original Wordle are in the data folder. There are
2300 possible solution words ('a' list) and over 10000 recognized words
('b' list). The optimal first word is _roate_, by the way.

Files:

- wordlestrat2.py: importable module
- run_wordlestrat2.py: example use of wordlestrat2
- wordle-strategy.py: older code, for finding optimal starting words
  to find most letters (for humans).
- build_cache.py: build cache for the second guess.
- player.py: stand-alone program that plays against the computer.

How to add a game dataset
-------------------------
- Figure out the a and b wordlists: a for possible words to figure out
  and b for allowed words to enter. Store in text files ('\n' line endings)
  in the data directory.
- Update `Wordle.__init__` and `Wordle.get_datasets`.
- Run: `wrd = Wordle('yourdataset'); wrd.get_optimal_first_word()`
- Use output to update `_FIRST_WORDS` in wordlestrat2.py
- Run: `Wordle('yourdataset'); wrd.build_cache()`
- Before committing, copy cache/cache-yourdataset.txt to the data/
  directory.
