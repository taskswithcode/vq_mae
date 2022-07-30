#python main_finetune.py --eval --resume ft_model/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /home/acc/datasets/imagenet1k/ILSVRC/Data/CLS-LOC
#python main_finetune.py --eval --resume ft_model/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /home/acc/datasets/imagenetv2 
#python main_finetune.py --eval --resume tuned_output/checkpoint-49.pth --model vit_base_patch16 --batch_size 16 --data_path  birds_augmented/datasets/birds/ft
