"""
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

Integration tests for BLIP2 models.
"""

import pytest
import torch
from lavis.models import load_model, load_model_and_preprocess
from lavis.models.blip2_models.Qformer import BertSelfAttention
from PIL import Image
import requests

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sample image
#raw_image = Image.open("docs/_static/merlion.png").convert("RGB")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')


class TestBlip2:
    def test_blip2_opt2p7b(self):
        # loads BLIP2-OPT-2.7b caption model, without finetuning on coco.
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device
        )

        # model, vis_processors, text_processors = load_model_and_preprocess(name="blip2_opt", model_type="pretrain_opt2.7b", is_eval=True, device=device)

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # question = "what city is this?"
        # question = text_processors['eval'](question)

        # answer = model.predict_answers(samples={"image": image, "text_input": question}, inference_method="generate")
        # print("answer",answer)

        # generate caption
        #for m in model.modules():
            #print("m:", m)  
        #print("qformer:", model.Qformer)

        caption = model.generate({"image": image})


        print("caption", caption)

        #assert caption == ["the merlion fountain in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)
        all_attn_probs = []
        for name, layer in model.Qformer.named_modules():
            # BertModel->BertEncoder->BertLayer -> BertAtteention -> BertSelfAttention
            if isinstance(layer, BertSelfAttention) and layer.is_cross_attention == True:
                print(f'{name}, attention_probs={layer.attn_probs}, attn_probs_shape={layer.attn_probs.shape}')
                all_attn_probs.append(layer.attn_probs)

        # TODO: try with atten_score (normalize score) 
        # TODO: atten_prob (instead of sum)     (variance?)
        # TODO: baseline (song han (query-token self-attention)) [TODO 1]
        # TODO (idea 3): for each image patch, rank each query tokens
        #       - Each image patch, vote in uniform way [TODO 2]
        #       - Each image patch, vote based on token importance (based on image self-attention) [TODO: 3]
        # TODO (idea 4): spatial relationship along sequence dimension ?

        # (layer_dim, bs_dim, head_dim, query_dim, key_dim) 
        all_attn_probs = torch.stack(all_attn_probs, dim=0)
        importance = torch.zeros(all_attn_probs.shape[3], device = 'cuda:0')
        for i in range(all_attn_probs.shape[-1]):
            r = torch.argsort(torch.argsort(all_attn_probs[:,:,:,:,i]))
            importance += torch.sum(r, dim=(0,1,2))
        rank_arg= torch.argsort(importance, descending=True)
        print('importance_score:', importance)
        print('rank_arg:', rank_arg)



        # sum_attn_probs = torch.sum(all_attn_probs, dim=(2, 4))
        #sum_attn_probs = torch.sum(all_attn_probs, dim=(2, 3))
        #for layer_id in range(sum_attn_probs.shape[0]):
            #print(f"layer_id={layer_id}, sum_attn_probs={sum_attn_probs[layer_id, :, :]}")
        #print(all_attn_probs.shape)
        #print(sum_attn_probs.shape)
            #for m in layer.modules():
                #if isinstance(m, BertSelfAttention) and m.is_cross_attention == True:
                    #print('qformer_m:', m)


        #assert len(captions) == 3

    def test_blip2_opt2p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt2.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        # image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
	#img_url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
	#raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        
	# generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a mermaid spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)



        assert len(captions) == 3

    def test_blip2_opt6p7b(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt", model_type="pretrain_opt6.7b", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a statue of a merlion in front of a water fountain"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_opt6p7b_coco(self):
        # loads BLIP2-OPT-2.7b caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type="caption_coco_opt6.7b",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["a large fountain spraying water into the air"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xl(self):
        # loads BLIP2-FLAN-T5XL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["marina bay sands, singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

    def test_blip2_flant5xxl(self):
        # loads BLIP2-FLAN-T5XXL caption model,
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5",
            model_type="pretrain_flant5xxl",
            is_eval=True,
            device=device,
        )

        # preprocess the image
        # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        # generate caption
        caption = model.generate({"image": image})

        assert caption == ["the merlion statue in singapore"]

        # generate multiple captions
        captions = model.generate({"image": image}, num_captions=3)

        assert len(captions) == 3

#test
t = TestBlip2()
t.test_blip2_opt2p7b()

