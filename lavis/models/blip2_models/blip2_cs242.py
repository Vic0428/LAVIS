from dataclasses import dataclass
import colorlog

@dataclass
class CS242Config:
    llm_quantization: str
    token_pruning: str
    token_pruning_level: int

"""
Initialize logger
"""
def init_logger(name):
    handler = colorlog.StreamHandler()

    logger = colorlog.getLogger(name)
    handler.setFormatter(colorlog.ColoredFormatter('%(green)s%(levelname)s:%(name)s:\t%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(colorlog.INFO)
    return logger