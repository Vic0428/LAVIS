CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 evaluate.py --cfg-path lavis/projects/blip2/eval/vqav2_zeroshot_opt_eval_lambda3.yaml