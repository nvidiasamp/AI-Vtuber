import os
os.environ['NVTE_ASYNC_SAVE'] = '0'

# filesystem_async.pyをモンキーパッチして同期保存に変更
import multiprocessing
import queue

# Managerの代わりにシンプルなQueueを使うようにパッチ
import megatron.core.dist_checkpointing.strategies.filesystem_async as fa_module

original_get_queue = fa_module._get_write_results_queue

def patched_get_queue():
    return queue.Queue()

fa_module._get_write_results_queue = patched_get_queue

# 続けてmerge_lora実行
from pathlib import Path
import torch
from nemo.collections.llm.peft.lora import LoRAMerge
from nemo.collections.llm.peft.api import _load_base_model_and_lora, _setup_trainer_and_restore_model_and_adapter, _save_merged_weight
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from lightning.pytorch import Trainer
from nemo.lightning import MegatronStrategy

lora_checkpoint_path = '/output/nike_lora/2026-06-23_03-01-18/checkpoints/nike_lora--val_loss=0.1842-epoch=0-consumed_samples=600.0-last'
output_path = '/output/nike_lora_merged'

print("Trainerセットアップ中...")
trainer = Trainer(
    devices=1,
    accelerator="cpu",
    strategy=MegatronStrategy(
        ddp="pytorch",
        setup_optimizers=False,
        plugins=bf16_mixed(),
    ),
)

print("モデルとLoRAロード中...")
model, lora = _load_base_model_and_lora(lora_checkpoint_path)
_setup_trainer_and_restore_model_and_adapter(Path(lora_checkpoint_path), trainer, model, lora)

print("LoRAマージ中...")
lora_merge = LoRAMerge()
merged_model = lora_merge(trainer.strategy.megatron_parallel)
merged_weights = {k: v for k, v in merged_model.sharded_state_dict().items() if ".adapter." not in k}

print("保存中...")
_save_merged_weight(output_path, merged_weights, model, trainer)
print("完了！")
