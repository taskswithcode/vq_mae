JOB_DIR=train_output_dir
#IMAGENET_DIR=birds_augmented/datasets/birds/
#IMAGENET_DIR=discrete_imagenet/
IMAGENET_DIR=/home/acc/final_save/datasets/imagenet1k/ILSVRC/Data/CLS-LOC/
MODEL_TYPE=mae_vit_base_patch16


#rm -rf $JOB_DIR
#mkdir -p $JOB_DIR

python main_pretrain.py \
    --output_dir ${JOB_DIR} \
    --batch_size 32 \
    --model ${MODEL_TYPE} \
    --mask_ratio 0.05 \
    --epochs 20 \
    --accum_iter 2 \
    --resume  train_output_dir/checkpoint-0.pth \
    --warmup_epochs 1 \
    --blr 6e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
