- Wordle-optim -- optimization for Wordle game
============================================

Author: Han-Kwang Nienhuys (@hk_nien on Twitter).

This code is for optimizing the choice of word guesses in the
Wordle game, in particular:

- https://www.powerlanguage.co.uk/wordle/ (original English)
- https://hellowordl.net/ (English)
- https://woordle.nl/ (Dutch)

There must be tens of other projects doing the similar things. Well, this
is mine; I wrote it for fun without checking other approaches.

I usually run this interactively from Spyder.

Files:

- wordlestrat2.py: importable module 
- run_wordlestrat2.py: example use of wordlestrat2
- wordle-strategy.py: older code, for finding optimal starting words
  to find most letters (for humans).
- build_cache.py: build cache for the second guess.

How to add a game dataset
-------------------------
- Figure out the a and b wordlists: a for possible words to figure out
  and b for allowed words to enter. Store in text files ('\n' line endings)
  in the data directory.
- Update Wordle.__init__ and Wordle.get_datasets.
- Run: `wrd = Wordle('yourdataset'); wrd.test_words(wrd.warr_a)`
- Use output to update _FIRST_WORDS in wordlestrat2.py
- Run: `Wordle('yourdataset'); wrd.build_cache()`
- Before committing, copy cache/cache-yourdataset.txt to the data/
  directory.
  