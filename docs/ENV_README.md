# Prepare environment

1. Create a conda virtual environment

```shell
conda create --name ovseg python==3.10.4
```
2. Install [torch==1.11.0+cu113]((https://pytorch.org/))
```shell
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
3. Install [mmcv-full==1.3.14](https://mmcv.readthedocs.io/en/latest/get_started/installation.html) and [mmsegmentation==0.18.0](https://mmsegmentation.readthedocs.io/en/latest/get_started.html#)

```shell 
pip install -U openmim
mim install mmcv-full==1.3.14
pip install mmsegmentation==0.18.0
```

4. Install clip
```shell
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

5. Install other dependencies
```shell
pip install -r requirements.txt
```