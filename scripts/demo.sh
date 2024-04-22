srun -p Gvlab-S1 -n1 -N1 --gres=gpu:1 --job-name=demo --quotatype=auto --cpus-per-task=10 \
python -u -m main_demo \
    --cfg configs/test_voc12.yml \
    --resume /mnt/petrelfs/xujilan/exps/cc12m_100/best_miou.pth \
    --vis input_pred_label \
    --vocab aeroplane bicycle bird boat bottle bus car cat chair cow table dog horse motorbike person plant sheep sofa train monitor \
    --image_folder ./visualization/input/ \
    --output_folder ./visualization/output/ \
