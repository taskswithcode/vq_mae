JOB_DIR=dbfinal_tuned_output
LOG_DIR=final_log
#IMAGENET_DIR=birds_augmented/datasets/birds/ft
#IMAGENET_DIR=discrete_imagenet/ft/
IMAGENET_DIR=/home/acc/final_save/datasets/imagenet1k/ILSVRC/Data/CLS-LOC/ft
MODEL_TYPE=vit_base_patch16


#rm -rf $JOB_DIR
mkdir -p $JOB_DIR

#CKPT=m15_train_output_dir/checkpoint-11.pth
#CKPT=m05_train_output_dir/checkpoint-15.pth
CKPT=train_output_dir/checkpoint-1.pth
#CKPT=final_tuned_output_cls4/checkpoint-16.pth
#CKPT=final_tuned_output_cls5/checkpoint-20.pth

#python main_finetune.py \
#    --output_dir ${JOB_DIR} \
#    --batch_size 48 \
#    --model ${MODEL_TYPE} \
#    --finetune ${CKPT} \
#    --epochs 200 \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05\
#    --drop_path 0.1 \
#    --reprob 0.25\
#    --mixup 0.8 \
#    --cutmix 1.0 \
#    --cls_token \
#    --log_dir ${LOG_DIR} \
#    --data_path ${IMAGENET_DIR}

#global pool - performs better than CLS above
#python main_finetune.py \
#    --output_dir ${JOB_DIR} \
#    --batch_size 48 \
#    --model ${MODEL_TYPE} \
#    --resume ${CKPT} \
#    --epochs 200 \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05\
#    --drop_path 0.1 \
#    --reprob 0.25\
#    --mixup 0.8 \
#    --cutmix 1.0 \
#    --log_dir ${LOG_DIR} \
#    --data_path ${IMAGENET_DIR}

#Attempting accum iter  to increase batch size
python main_finetune.py \
    --accum_iter 2 \
    --output_dir ${JOB_DIR} \
    --batch_size 48 \
    --model ${MODEL_TYPE} \
    --finetune ${CKPT} \
    --epochs 200 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05\
    --drop_path 0.1 \
    --reprob 0.25\
    --mixup 0.8 \
    --cutmix 1.0 \
    --log_dir ${LOG_DIR} \
    --data_path ${IMAGENET_DIR}

#python main_finetune.py \
#    --output_dir ${JOB_DIR} \
#    --batch_size 48 \
#    --model ${MODEL_TYPE} \
#    --finetune ${CKPT} \
#    --epochs 200 \
#    --blr 5e-4 --layer_decay 0.65 \
#    --weight_decay 0.05\
#    --drop_path 0.1 \
#    --reprob 0.25\
#    --mixup 0.8 \
#    --cutmix 1.0 \
#    --log_dir ${LOG_DIR} \
#    --data_path ${IMAGENET_DIR}
