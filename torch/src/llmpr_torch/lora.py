"""LoRA configuration shared by smoke tests and training."""
from __future__ import annotations

TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def attach_lora(model, *, rank: int = 16, alpha: int = 32):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(TARGET_MODULES),
    )
    adapted = get_peft_model(model, config)
    matched = tuple(adapted.targeted_module_names)
    missing = [name for name in TARGET_MODULES
               if not any(module.endswith(name) for module in matched)]
    if missing:
        raise RuntimeError(f"LoRA target modules were not matched: {missing}")
    return adapted
