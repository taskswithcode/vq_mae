python main_finetune.py --eval --resume final_tuned_noevaltrans0/checkpoint-1.pth --model vit_base_patch16 --batch_size 16 --data_path  /home/ubuntu/main/datasets/imagenet1k/ft
#python main_finetune.py --eval --resume ft_model/mae_finetuned_vit_base.pth --model vit_base_patch16 --batch_size 16 --data_path /home/acc/datasets/imagenetv2 
#python main_finetune.py --eval --resume tuned_output/checkpoint-49.pth --model vit_base_patch16 --batch_size 16 --data_path  birds_augmented/datasets/birds/ft
#python main_finetune.py --eval --resume tuned_output/checkpoint-49.pth --model vit_base_patch16 --batch_size 16 --data_path  birds_augmented/datasets/birds/ft
