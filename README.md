# Learning Open-Vocabulary Semantic Segmentation Models From Natural Language Supervision

This repository is the official implementation of [Learning Open-Vocabulary Semantic Segmentation Models From Natural Language Supervision](https://arxiv.org/abs/2301.09121) at CVPR 2023. Our transformer-based model, termed as OVSegmentor, is pre-trained on image-text pairs without using any mask annotations. After training, it can segment objects of arbitrary categories via zero-shot transfer. 


<div align="center">
<img src="figs/model.png" width="100%">
</div>

## Requirements
* Python 3.9
* [torch=1.11.0+cu113](https://pytorch.org/)
* [torchvision=0.14.1](https://pytorch.org/)
* [apex=0.1](https://github.com/NVIDIA/apex)
* [mmcv-full=1.3.14](https://github.com/open-mmlab/mmcv)
* [mmsegmentation=0.18.0](https://github.com/open-mmlab/mmsegmentation)
* [clip=1.0](https://github.com/openai/CLIP)

We recommand installing apex with cuda and c++ extensions

To install the other requirements:

```setup
pip install -r requirements.txt
```

## Prepare datasets
For training, we construct CC4M by filtering CC12M with a total number of 100 frequently appearred entities. The researchers are encouraged to prepare CC12M dataset from the [source](https://github.com/google-research-datasets/conceptual-12m) or using [img2dataset](https://github.com/rom1504/img2dataset). Note that, some url links may not be available any longer. The file structure should follow:

```shell
CC12M
├── 000002a0c848e78c7b9d53584e2d36ab0ac14785.jpg
├── 000002ca5e5eab763d95fa8ac0df7a11f24519e5.jpg
├── 00000440ca9fe337152041e26c37f619ec4c55b2.jpg
...
```
We provide the meta-file for CC4M at [here](https://drive.google.com/file/d/1ENpsWndAkWc0UZJvdJJDicPxpzrPugve/view?usp=share_link) for data loading. One may also try different [image-caption datasets](https://github.com/rom1504/img2dataset) (e.g. YFCC, RedCaps) by providing the images and the corresponding meta-file. The meta-file is a json file containing each filename and its caption in a single line.
```shell
{"filename": "000002ca5e5eab763d95fa8ac0df7a11f24519e5.jpg", "caption": "A man's hand holds an orange pencil on white"}
{"filename": "000009b46e38a28790f481f36366c781e03e4bbd.jpg", "caption": "Cooking is chemistry, except you CAN lick the spoon!"}
...
```
For evaluation, please follow the official websites to prepare [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc), [PASCAL Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context), [COCO](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k) converted to semantic seg format following [GroupViT](https://github.com/NVlabs/GroupViT), and [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k). Remember to change the image dirs in segmentation/configs/_base_/datasets/*.py.

To enable zero-shot classification evaluation, please prepare the validation set of [ImageNet](https://www.image-net.org/) with its corresponding meta-file. 

## Other preparations
1. The visual encoder is initialised with [DINO](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth). Edit the checkpoint path in the config file.
2. Pre-trained BERT model and nltk_data should be downloaded automatically.

## Training
To train the model(s) in the paper, we separate the training process as a two-stage pipeline. The first stage is a 30-epoch training with image-caption contrastive loss and masked entity completion loss, and the second-stage 10-epoch training further adds the cross-image mask consistency loss. 

For the first stage training on a single node with 8 A100 (80G) GPUs, we recommand to use slurm script to enable training:

```train
cd OVSegmentor
./tools/run_slurm.sh
```
Or simply use torch.distributed.launch as:

```train
./tools/run.sh
```

After that, please specify the checkpoint path from the 1st stage training in the config file used in the 2nd stage training (e.g. configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage2.yml). During cross-image sampling, we sample another image that share the same entity with the current image. This is achieved by (1) identifying the visual entity for the image. (2) Perform sampling over the valid candidates. We offer the pre-processed [class_label.json](https://drive.google.com/file/d/15s0Pwn11bkB-RqGmpzf7z6lYPOd1sIZF/view?usp=share_link) and [sample_list.json](https://drive.google.com/file/d/10sA94ZawsgL0E01im9-5xZciWnsCZOQz/view?usp=share_link).

We also provide our pre-trained 1st stage checkpoint from [here](https://drive.google.com/file/d/19Kpeh5iTgGSr5mzf4n0j5hqxGDgG-Wxi/view?usp=share_link).

Then, perform the second stage training. 
```train
./tools/run_slurm_stage2.sh
```
We adjust a few hyperparameters in 2nd stage to stablize the training process.

## Evaluation

To evaluate the model on PASCAL VOC, please specify the checkpoint path in tools/test_voc12.sh, and run:

```eval
./tools/test_voc12.sh
```
For PASCAL Context, COCO Object, and ADE20K, please refer to tools/.

The performance may vary 3%~4% due to different cross-image sampling. 

## Model Zoo

The pre-trained models can be downloaded from here:

| Model name  | Visual enc | Text enc      | Group tokens  | PASCAL VOC  | PASCAL Context | COCO Object | ADE20K | Checkpoint |
| ------------------ |------------------ |------------------ |---------------- | -------------- | -------------- | -------------- | -------------- | -------------- |
| OVSegmentor    | ViT-B|  BERT-Base|   8         |      53.8 |20.4       |  25.1         |      5.6       |    [download](https://drive.google.com/file/d/10F3b3FNzPdDx8LuKdjc1BzbSLMrPLvnc/view?usp=share_link)       |
| OVSegmentor |ViT-S | Roberta-Base   |     8         |     44.5| 18.3       | 19.0         |      4.3       |   [download](https://drive.google.com/file/d/10F3b3FNzPdDx8LuKdjc1BzbSLMrPLvnc/view?usp=share_link)      |
| OVSegmentor    | ViT-B|  BERT-Base|   16         |      Todo | Todo       | Todo         |      Todo       |   Todo       |

## Citation
If this work is helpful for your research, please consider citing us.
```
@article{xu2023learning,
  title={Learning Open-vocabulary Semantic Segmentation Models From Natural Language Supervision},
  author={Xu, Jilan and Hou, Junlin and Zhang, Yuejie and Feng, Rui and Wang, Yi and Qiao, Yu and Xie, Weidi},
  journal={arXiv preprint arXiv:2301.09121},
  year={2023}
}
```

## Acknowledgements
This project is built upon [GroupViT](https://github.com/NVlabs/GroupViT). Thanks to the contributors of the great codebase.
