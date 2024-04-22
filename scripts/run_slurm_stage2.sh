PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Gvlab-S1 -N1 -n1 --gres=gpu:1 --job-name=4m_stage2 --quotatype=auto --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage2.yml
