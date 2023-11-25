import logging
from transformers import OPTForCausalLM, BitsAndBytesConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from lavis.models.eva_vit import convert_weights_to_fp16
import torch
import torch.nn as nn
from lavis.models.blip2_models.blip2_cs242 import init_logger


logger = init_logger(__name__)

"""
Quantization functions
"""
def load_8bit_opt_model(opt_model):
    logger.info("Quantize OPT model into int8")
    opt_model = OPTForCausalLM.from_pretrained(opt_model, 
                                               load_in_8bit=True)
    return opt_model

def load_4bit_opt_model(opt_model):
    logger.info("Quantize OPT model into int4")
    bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                    )
    opt_model = OPTForCausalLM.from_pretrained(
        opt_model,
        quantization_config=bnb_config
    )
    return opt_model

def inplace_quantize_fp16_model(model):
    convert_weights_to_fp16(model)

def quantize_8bit_model(model):
    bnb_config = BnbQuantizationConfig(
                    load_in_8bit=True,
                    skip_modules=[])
    return load_and_quantize_model(model, bnb_quantization_config=bnb_config)

def quantize_4bit_model(model):
    bnb_config = BnbQuantizationConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    skip_modules=[])
    return load_and_quantize_model(model, bnb_quantization_config=bnb_config)
