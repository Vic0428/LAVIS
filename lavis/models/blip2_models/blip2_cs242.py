from dataclasses import dataclass

@dataclass
class CS242Config:
    llm_quantization: str
    token_pruning: str
    token_pruning_level: int