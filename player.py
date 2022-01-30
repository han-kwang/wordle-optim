#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wordle player.

Usage:

    python3 player.py [dataset]

Where dataset is a predefined dataset, e.g. 'en'. Leave out to be prompted.

Created on Fri Jan 28 22:34:28 2022 // @hk_nien
"""
import sys
from wordlestrat2 import Wordle

def get_wordle():
    """Parse command line, prompt user, return initialized Wordle instance."""

    dsets = Wordle.get_datasets()
    if len(sys.argv) == 2 and sys.argv[1].startswith('-') or len(sys.argv) > 2:
        print(__doc__)
        sys.exit(1)

    if len(sys.argv) == 1:

        for i, ds in enumerate(dsets):
            print(f'{i} {ds}')
        r = input('Pick a dataset: ')
        dset = dsets[int(r)]
        print(f'Next time, you can run:\n  {sys.argv[0]} {dset}')
    else:
        dset = sys.argv[1]
        if dset not in dsets:
            print(f'Unsupported dataset: {dset}')
            sys.exit(1)

    return Wordle(dset)


if __name__ == '__main__':

    wrd = get_wordle()
    print(f'Running Wordle player for dataset {wrd.dataset}.\n'
          'Ctrl-D/Ctrl-Z to abort input; Ctrl-C to end.')
    while True:
        wrd.play_ai()
