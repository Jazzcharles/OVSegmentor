PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=29500 \
    main_pretrain.py --cfg configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage2.yml