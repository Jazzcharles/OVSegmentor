# Prepare Model

## Preparing pretrained DINO model
1. Download DINO pretrained weights and specify the model path.
```shell
wget https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```

2. Change the configs/default.yml by specifying the checkpoint path.
```shell
imgnet_pretrained_checkpoint: '/path/to/your/dino_vitbase16_pretrain.pth'
```

Pre-trained BERT model and nltk_data should be downloaded automatically.