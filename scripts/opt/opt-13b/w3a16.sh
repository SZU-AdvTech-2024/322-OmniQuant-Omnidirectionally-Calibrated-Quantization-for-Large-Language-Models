CUDA_VISIBLE_DEVICES=0 python main.py \
--model facebook/opt-13b --eval_ppl \
--epochs 20 --output_dir ./log/opt-13b-w3a16 \
--wbits 3 --abits 16 --lwc --let