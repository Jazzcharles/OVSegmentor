PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p video -N1 -n1 --gres=gpu:1 --job-name=test_ade20k --quotatype=auto --cpus-per-task=10 \
python -u -m main_seg \
    --cfg configs/test_ade20k.yml \
    --resume /mnt/petrelfs/xujilan/exps/cc12m_100/test_voc12_bs256x1/best_miou.pth \