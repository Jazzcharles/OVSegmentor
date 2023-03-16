PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p video -N1 -n8 --gres=gpu:8 --job-name=4m --quotatype=auto --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage1.yml
