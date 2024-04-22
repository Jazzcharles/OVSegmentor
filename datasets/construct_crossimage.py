import json
import numpy as np
import nltk
import argparse
import os
import pandas as pd
from ipdb import set_trace

def construct(args):
    metafile = pd.read_csv(args.metafile, sep='\t')
    

if __name__ == '__main__':
    '''
    Given the metafile, we construct the 
    Simply run: python construct_crossimage.py
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--metafile', type=str, default='/mnt/petrelfs/xujilan/data/cc12m_100/cc12m_filtered_subset.tsv')
    args = parser.parse_args()
    construct(args)
