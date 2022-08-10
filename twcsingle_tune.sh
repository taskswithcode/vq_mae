option=${1?"Specify 1 for start or 2 for resume"}
JOB_DIR=final_tuned_noeval_trans
LOG_DIR=final_log
#IMAGENET_DIR=birds_augmented/datasets/birds/ft
#IMAGENET_DIR=discrete_imagenet/ft/
#IMAGENET_DIR=/home/acc/final_save/datasets/imagenet1k/ILSVRC/Data/CLS-LOC/ft
IMAGENET_DIR=/home/ubuntu/main/datasets/imagenet1k/ft
MODEL_TYPE=vit_base_patch16


#rm -rf $JOB_DIR
mkdir -p $JOB_DIR

#CKPT=m15_train_output_dir/checkpoint-11.pth
#CKPT=m05_train_output_dir/checkpoint-15.pth
#CKPT=train_output_dir/checkpoint-1.pth
#CKPT=final_tuned_output_cls4/checkpoint-16.pth
#CKPT=final_tuned_output_cls5/checkpoint-20.pth
#CKPT=dbfinal_tuned_output/checkpoint-5.pth
CKPT=train_output_dir/checkpoint-1.pth
#CKPT=final_tuned_accum_22/checkpoint-13.pth

#global pool - performs better than CLS above
#Attempting accum iter  to increase batch size
ACCUM_ITER=22
BATCH_SIZE=48
EPOCHS=100
BLR=1e-3
LAYER_DECAY=.75
WEIGHT_DECAY=.05
DROP_PATH=.1
#Note cutmix is set to zero since it is not relevant when training on discretized images
#https://arxiv.org/abs/1905.04899
#REPOROB is a noop
REPROB=.25
#MIXUP=.8
MIXUP=0
CUTMIX=0
SMOOTHING=.1
if [ $option -eq 1 ]
then
	echo "Starting fine tuning"
python main_finetune.py \
    --accum_iter ${ACCUM_ITER} \
    --output_dir ${JOB_DIR} \
    --batch_size ${BATCH_SIZE} \
    --model ${MODEL_TYPE} \
    --finetune ${CKPT} \
    --epochs ${EPOCHS} \
    --blr ${BLR}\
    --layer_decay ${LAYER_DECAY} \
    --weight_decay ${WEIGHT_DECAY} \
    --drop_path ${DROP_PATH} \
    --reprob ${REPROB} \
    --mixup ${MIXUP} \
    --cutmix ${CUTMIX} \
    --log_dir ${LOG_DIR} \
    --smoothing ${SMOOTHING} \
    --data_path ${IMAGENET_DIR}


else
	echo "Resuming fine tuning"
python main_finetune.py \
    --accum_iter ${ACCUM_ITER} \
    --output_dir ${JOB_DIR} \
    --batch_size ${BATCH_SIZE} \
    --model ${MODEL_TYPE} \
    --resume ${CKPT} \
    --epochs ${EPOCHS} \
    --blr ${BLR}  \
    --layer_decay ${LAYER_DECAY} \
    --weight_decay ${WEIGHT_DECAY} \
    --drop_path ${DROP_PATH} \
    --reprob ${REPROB} \
    --mixup ${MIXUP} \
    --cutmix ${CUTMIX} \
    --log_dir ${LOG_DIR} \
    --smoothing ${SMOOTHING} \
    --data_path ${IMAGENET_DIR}
fi

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
