import json
import numpy as np
import nltk
import argparse
import os
import pandas as pd
from ipdb import set_trace
import subprocess
import random
from multiprocessing import Pool

### You can modify this as you want ###
TOP_CLASSES_1=[
    'people', 'man', 'men', 'woman', 'women', 'girl', 'boy', 'lady', 'kid', 'child', 'children', 'baby', 'student', 'bride', 'groom', 'couple', 'prince', 'princess', \
    'car', 'bus', 'truck', 'motorcycle', 'train', 'bicycle', 'boat', 'aeroplane', 'airplane', 'motorbike', 'bike',\
    'cup', 'bottle', 'bowl', 'knife', 'spoon',  'glass', 'fork',\
    'chair', 'table', 'bench', 'clock', 'laptop', 'light', 'vase', 'plant', 'remote', 'microwave', 'toaster', 'oven','mouse', 'keyboard','sofa', 'monitor','desk', 'tv','TV', 'couch', 'flower','refrigerator', \
    'house', 'building', 'hotel',\
    'handbag', 'umbrella','book', 'backpack', 'phone', 'shirt', 'tie', 'suitcase','T-shirt', 'bag',  'box', \
    'sink','bed','toilet',\
    'cat','dog',  'horse', 'bird','cow', 'sheep' ,'elephant', 'bear', 'zebra', 'giraffe', \
    'ball', 'racket', 'skateboard', 'skis', 'snowboard', 'surfboard', 'kite', \
    'pizza', 'cake', 'apple', 'banana', 'sandwich', 'orange', 'carrot', 'donut' ,\
]


def judge_noun(word):
    if word in TOP_CLASSES_1:
        return 1
    return 0

def make_filter(infos):
    args, cur_index, subset_list = infos[0], infos[1], infos[2]
    new_dataframe = pd.DataFrame()

    print(f'Begin processing {cur_index}')
    for i, item in enumerate(subset_list.iterrows()):
        each_cap = item[1]['caption']
        all_words = nltk.word_tokenize(each_cap)
        valid_list = [judge_noun(word) for word in all_words]
        valid = sum(valid_list)
        if valid:
            valid_words = np.array(all_words)[np.argwhere(valid_list)][:,0].tolist()
            valid_words = list(set(valid_words)) ## keep unique entities
            item[1]['entity'] = ','.join(valid_words)
            new_dataframe = new_dataframe._append(item[1])
            
    print('Filtered {} out of {}'.format(len(new_dataframe), len(subset_list)))
    new_dataframe.to_csv(f'{args.dstdir}/cc12m_{cur_index}.csv', index=False)
    return 

def filter_subset_with_entities(args):
    # all_captions = pd.read_csv(f'{args.srcdir}/cc12m.tsv', sep='\t')
    if args.srcdir.endswith('.tsv'):
        all_captions = pd.read_csv(args.srcdir, sep='\t')
    elif args.srcdir.endswith('.csv'):
        all_captions = pd.read_csv(args.srcdir)
        
    all_captions.columns = ['image_id', 'caption']
    total_len = len(all_captions)

    all_ranges = np.linspace(0.0, 1.0, args.processors + 1)
    
    chunk_list = []
    for i in range(args.processors):
        begin = int(all_ranges[i] * total_len)
        ender = int(all_ranges[i + 1] * total_len)
        subset_list = all_captions[begin:ender]
        chunk_list.append([args, i, subset_list])
    
    print(f'Begin filtering with {args.processors}')
    pool = Pool(args.processors)
    pool.map(make_filter, tuple(chunk_list))


def merge_all_subset(args):
    all_files = os.listdir(args.dstdir)
    all_files = [f for f in all_files if f.endswith('.csv')]
    all_files = sorted(all_files)
    
    all_data = pd.DataFrame()
    for f in all_files:
        each_data = pd.read_csv(os.path.join(args.dstdir, f))
        all_data = all_data._append(each_data)

    if args.remove_subfiles:
        print('Removing sub-files')
        for f in all_files:
            cmd = f'rm -f {os.path.join(args.dstdir, f)}'
            print(cmd)
            os.system(cmd)

    all_data.to_csv(os.path.join(args.dstdir, 'cc12m_filtered_subset.csv'), index=False)
    

def construct_crossimage(args):
    '''
    For each image, we randomly sample K (e.g. 10) images that contain shared entity.
    '''
    metafile = pd.read_csv(args.metafile)
    all_entites = metafile['entity'].tolist()

    entity_dict = {}
    for i, each_entity in enumerate(all_entites):
        each_entity = each_entity.split(',')
        # print(i, each_entity)
        for sub_entity in each_entity:
            if sub_entity not in entity_dict:
                entity_dict[sub_entity] = []
            entity_dict[sub_entity].append(i)
    
    print('Done calculating entity dict')
    for k, v in entity_dict.items():
        print(k, len(v))
    
    ### assign entity ###
    topK = 10
    all_pairs = []
    all_paired_entity = []
    print(f'Begin sampling {topK} pairs for each element')

    for i, each_entity in enumerate(all_entites):
        each_entity = each_entity.split(',')
        sampled_entity = np.random.choice(each_entity, size=topK, replace=True)
        sampled_pair = [random.choice(entity_dict[x]) for x in sampled_entity]
        all_pairs.append(sampled_pair)
        all_paired_entity.append(sampled_entity.tolist())
    
    assert len(all_pairs) == len(all_entites) == len(all_paired_entity)
    metafile['pairindex'] = all_pairs
    metafile['pairentity'] = all_paired_entity
    metafile.to_csv(args.metafile.replace('.csv', '_pair.csv'), index=False)
    print('Done constructing pairs')

def convert_json_to_csv(args):
    jsonfile = args.metafile
    df = pd.DataFrame()
    print('Start converting')
    with open(jsonfile) as f:
        lines = f.readlines()
        for line in lines:
            info = json.loads(line)
            df = df._append(pd.Series(info), ignore_index=True)
    outdir = args.metafile.replace('.json','.csv')
    df.to_csv(outdir)
    print(f'Done converting to {outdir}')
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processors', type    =int, default=8, help='processors for data filtering')
    parser.add_argument('--srcdir', type=str, default=None, help='source dir that contains the original cc12m metafile')
    parser.add_argument('--dstdir', type=str, default=None, help='target dir to save the filtered subset')
    parser.add_argument('--mode', type=str, default='filter', help='choices: [filter, merge, makepair]')
    parser.add_argument('--metafile', type=str, default=None, help='the metafile used for constructing cross-image pairs')
    parser.add_argument('--remove_subfiles', type=bool, default=False, help='whether to remove the generated sub-files')
    
    args = parser.parse_args()

    if args.mode == 'filter':
        assert args.srcdir is not None, 'Please specify the source dir containing the cc12m metafile'
        if args.dstdir is None:
            args.dstdir = f'{"/".join(args.srcdir.split("/")[:-1])}/subsets'
            print(f'Target dir not specified, use {args.dstdir}')
        
        os.makedirs(args.dstdir, exist_ok=True)
        filter_subset_with_entities(args)    

    elif args.mode == 'merge':
        assert args.dstdir is not None, 'Please specify the target dir containing the filtered metafiles'
        merge_all_subset(args)

    elif args.mode == 'makepair':
        assert args.metafile is not None, 'Please specify the metafile for constructing the cross-image relation'
        construct_crossimage(args)

    elif args.mode == 'json2csv':
        assert args.metafile is not None, 'Please specify the metafile for converting'
        convert_json_to_csv(args)

    else:
        raise NotImplementedError
