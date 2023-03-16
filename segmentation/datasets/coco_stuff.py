# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual
# property and proprietary rights in and to this software, related
# documentation and any modifications thereto.  Any use, reproduction,
# disclosure or distribution of this software and related documentation
# without an express license agreement from NVIDIA CORPORATION is strictly
# prohibited.
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------

from mmseg.datasets import DATASETS, CustomDataset


@DATASETS.register_module()
class COCOStufferDataset(CustomDataset):
    """COCO-Stuff dataset.

    1 bg class + 80 things + 91 stuff classes from the COCO-Stuff dataset.
    """

    CLASSES = ('background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
               'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 
               'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 
               'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile', 
               'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain', 'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble', 
               'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 
               'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 
               'net', 'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform', 'playingfield', 'railing', 'railroad', 'river', 'road', 
               'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs', 'stone', 'straw',
               'structural-other', 'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel', 
               'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops', 'window-blind', 'window-other', 'wood')

    PALETTE = [[0, 0, 0], [0, 192, 64], [0, 192, 64], [0, 64, 96], [128, 192, 192], [0, 64, 64], [0, 192, 224],
               [0, 192, 192], [128, 192, 64], [0, 192, 96], [128, 192, 64], [128, 32, 192], [0, 0, 224], [0, 0, 64],
               [0, 160, 192], [128, 0, 96], [128, 0, 192], [0, 32, 192], [128, 128, 224], [0, 0, 192], [128, 160, 192],
               [128, 128, 0], [128, 0, 32], [128, 32, 0], [128, 0, 128], [64, 128, 32], [0, 160, 0], [0, 0, 0],
               [192, 128, 160], [0, 32, 0], [0, 128, 128], [64, 128, 160], [128, 160, 0], [0, 128, 0], [192, 128, 32],
               [128, 96, 128], [0, 0, 128], [64, 0, 32], [0, 224, 128], [128, 0, 0], [192, 0, 160], [0, 96, 128],
               [128, 128, 128], [64, 0, 160], [128, 224, 128], [128, 128, 64], [192, 0, 32],
               [128, 96, 0], [128, 0, 192], [0, 128, 32], [64, 224, 0], [0, 0, 64], [128, 128, 160], [64, 96, 0],
               [0, 128, 192], [0, 128, 160], [192, 224, 0], [0, 128, 64], [128, 128, 32], [192, 32, 128], [0, 64, 192],
               [0, 0, 32], [64, 160, 128], [128, 64, 64], [128, 0, 160], [64, 32, 128], [128, 192, 192], [0, 0, 160],
               [192, 160, 128], [128, 192, 0], [128, 0, 96], [192, 32, 0], [128, 64, 128], [64, 128, 96], [64, 160, 0],
               [0, 64, 0], [192, 128, 224], [64, 32, 0], [0, 192, 128], [64, 128, 224], [192, 160, 0],
               [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 0, 3], [1, 0, 4], [1, 0, 5], [1, 0, 6], [1, 0, 7], [1, 0, 8], [1, 0, 9], [1, 0, 10], [1, 0, 11], [1, 0, 12], [1, 0, 13], [1, 0, 14], [1, 0, 15], [1, 0, 16], [1, 0, 17], [1, 0, 18], [1, 0, 19], [1, 0, 20], [1, 0, 21], [1, 0, 22], [1, 0, 23], [1, 0, 24], [1, 0, 25], [1, 0, 26], [1, 0, 27], [1, 0, 28], [1, 0, 29], [1, 0, 30], [1, 0, 31], [1, 0, 32], [1, 0, 33], [1, 0, 34], [1, 0, 35], [1, 0, 36], [1, 0, 37], [1, 0, 38], [1, 0, 39], [1, 0, 40], [1, 0, 41], [1, 0, 42], [1, 0, 43], [1, 0, 44], [1, 0, 45], [1, 0, 46], [1, 0, 47], [1, 0, 48], [1, 0, 49], [1, 0, 50], [1, 0, 51], [1, 0, 52], [1, 0, 53], [1, 0, 54], [1, 0, 55], [1, 0, 56], [1, 0, 57], [1, 0, 58], [1, 0, 59], [1, 0, 60], [1, 0, 61], [1, 0, 62], [1, 0, 63], [1, 0, 64], [1, 0, 65], [1, 0, 66], [1, 0, 67], [1, 0, 68], [1, 0, 69], [1, 0, 70], [1, 0, 71], [1, 0, 72], [1, 0, 73], [1, 0, 74], [1, 0, 75], [1, 0, 76], [1, 0, 77], [1, 0, 78], [1, 0, 79], [1, 0, 80], [1, 0, 81], [1, 0, 82], [1, 0, 83], [1, 0, 84], [1, 0, 85], [1, 0, 86], [1, 0, 87], [1, 0, 88], [1, 0, 89], [1, 0, 90],
               ]

    def __init__(self, **kwargs):
        super(COCOStufferDataset, self).__init__(img_suffix='.jpg', seg_map_suffix='_instanceTrainIds.png', **kwargs)
