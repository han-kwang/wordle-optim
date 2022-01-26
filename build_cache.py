#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build cache dist files.

Stand-alone script. Run this before pushing if necessary.

Created on Wed Jan 26 22:21:48 2022 // @hk_nien
"""
from pathlib import Path
import os
from wordlestrat2 import Wordle
os.chdir(str(Path(__file__).parent))
print('----Rebuilding cache----')
for dataset in Wordle.get_datasets():
    print(f'--- {dataset} ----')
    w = Wordle(dataset)
    w.build_cache()
    cpath = Path(w._get_cache_fpath())
    cpath_target = Path('data') / cpath.name
    cpath.rename(cpath_target)
    print(f'Created {cpath_target}')


