srun -p Gvlab-S1 -n1 -N1 --gres=gpu:1 --job-name=test_voc_context --quotatype=auto --cpus-per-task=10 \
python -u -m main_seg \
    --cfg configs/test_voc12.yml \
    --resume /mnt/petrelfs/xujilan/exps/cc12m_100/best_miou.pth \