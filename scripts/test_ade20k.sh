PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Gvlab-S1 -N1 -n1 --gres=gpu:1 --job-name=test_ade20k --quotatype=auto --cpus-per-task=10 \
python -u -m main_seg \
    --cfg configs/test_ade20k.yml \
    --resume /mnt/petrelfs/xujilan/exps/cc12m_100/best_miou.pth \