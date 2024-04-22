# Prepare Dataset

## Training data
Please prepare the [CC12M dataset](https://github.com/google-research-datasets/conceptual-12m). All the images should be stored in a folder. A metafile (csv or tsv file) that contains the image_id and its corresponding caption is needed.
```shell
image_id, caption
00001.jpg, a boy is running on the beach,
00002.jpg, The bride was wearing a chic lace.
... 
```

- IMPORTANT UPDATE:
We provide the script for filtering cc12m subset and constructing cross-image pairs from scratch:

1. Using multi-processing (e.g. 32 processes) to filter cc12m dataset using Top-K frequently appeared entities. Feel free to modify the entities in [data_process_cc12m.py](../datasets/filter_cc12m_subset.py).
```shell 
cd datasets
python data_process_cc12m.py --mode filter --srcdir /path/to/your/cc12m.csv --processor 32
```
This will generate 32 sub-files in subset/ directory. 

2. Next, merge these sub-files into a single metafile (and optionally delete the sub-files by passing --remove_subfiles=True).
```shell
python data_process_cc12m.py --mode merge --dstdir /path/to/your/cc12m/subsets/ --remove_subfiles True
```

3. Construct cross-image pairs based on the filtered data. 
```shell
python data_process_cc12m.py --mode makepair --metafile /path/to/your/cc12m_filtered_subset.csv
```
The generated metafile is automatically saved to /path/to/your/cc12m_filtered_subset_pair.csv. This metafile can be used for training the model.

4. Modify the root path and metafile path in configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage1.yml
```shell
data:
    train:
        root_dir: '/path/to/your/cc12m_images/'
        meta_file: '/path/to/your/cc12m_filtered_subset_pair.csv'
```


One may also try different [image-caption datasets](https://github.com/rom1504/img2dataset) (e.g. YFCC, RedCaps) by providing the images and the corresponding meta-file.

## Evaluation
1. Follow the official websites to prepare [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc), [PASCAL Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context), [COCO](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k), and [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k).
2. For COCO dataset, convert it to semantic segmentation format following [GroupViT](https://github.com/NVlabs/GroupViT). 
```shell
python convert_dataset/convert_coco_object.py /path/to/your/coco/ -o /path/to/output/coco/
```
3. Change the image dirs in segmentation/configs/_base_/datasets/*.py.
- [PASCAL VOC](../segmentation/configs/_base_/datasets/pascal_voc12.py)
```shell
data_root = '/path/to/your/VOCdevkit/VOC2012'
```
- [PASCAL CONTEXT](../segmentation/configs/_base_/datasets/pascal_context.py)
```shell
data_root = '/path/to/your/pascal_context/VOCdevkit/VOC2010/'
```
- [COCO Object](../segmentation/configs/_base_/datasets/coco.py)
```shell
data_root = '/path/to/your/coco/'
```
- [COCO STUFF](../segmentation/configs/_base_/datasets/coco_stuff.py)
```shell
data_root = '/path/to/your/coco/'
```
- [ADE20K](../segmentation/configs/_base_/datasets/ade20k.py)
```shell
data_root = '/path/to/your/ADEChallengeData2016/'
```

4. To enable zero-shot classification evaluation, please prepare the validation set of [ImageNet](https://www.image-net.org/). The metafile of the validation set is already provided [here](../imagenet_info/val.csv). Modify the image path in configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage1.yml
```shell
val:
    root_dir: '/path/to/your/cc12m_images/'
```