import logging
from transformers import OPTForCausalLM, BitsAndBytesConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from lavis.models.eva_vit import convert_weights_to_fp16
import torch
import torch.nn as nn
import colorlog


"""
Initialize logger
"""
def init_logger():
    handler = colorlog.StreamHandler()

    logger = colorlog.getLogger(__name__)
    handler.setFormatter(colorlog.ColoredFormatter('%(green)s%(levelname)s:%(name)s:\t%(message)s'))
    logger.addHandler(handler)
    return logger
logger = init_logger()

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

def inplace_quantize_fp16_qformer(qformer):
    convert_weights_to_fp16(qformer)

def quantize_int8_qformer(qformer):
    return torch.ao.quantization.quantize_dynamic(
        qformer,
        {nn.Conv1d, nn.Conv2d, nn.Linear},
        dtype=torch.qint8
    )
