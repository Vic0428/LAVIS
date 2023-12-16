This repo is based on the LAVIS library (https://github.com/salesforce/LAVIS). We added our token-wise cross-attention pruning method and SOTA pruning methods in lavis/models/blip2_models/blip2_pruning.py:
* Magnitude based pruning baseline - magnitude_based_baseline()
* SpAtten pruning baseline - self_attention_pruning()
* Our cross-attention pruning method - cross_attention_pruning()
* Our cross-attention pruning method with image weight applied - cross_attention_pruning_with_image_weight()

To implement different pruning methods, we can change the model configuration parameter "token_pruning" in lavis/projects/blip2/eval/opt2.7b_quantization/vqav2_fp16.yaml:
* Magnitude based pruning baseline - token_pruning: "topk"
* SpAtten pruning baseline - token_pruning: "self_attention_based"
* Our cross-attention pruning method - token_pruning: "cross_attention_based"
* Our cross-attention pruning method with image weight applied- token_pruning: "cross_attention_with_image_weight"
  
To implement different pruning ratio for each method, we can change the model configuration parameter "token_pruning_level" in lavis/projects/blip2/eval/opt2.7b_quantization/vqav2_fp16.yaml:
* token_pruning_level: 2 (keep 1/2 of the tokens)
* token_pruning_level: 4 (keep 1/4 of the tokens)
* tokn_pruning_levle : 8 (keep 1/8 of the tokens)
  
We used the VQA dataset to test the model accuracy under different pruning method. 
To download the dataset used for our model: python lavis/datasets/download_scripts/download_coco.py

To run our pruning methods: CUDA_VISIABLE_DEVICES=0 python evaluate.py --cfg-path lavis/projects/blip2/eval/opt2.7b_quantization/vqav2_fp16.yaml



           
