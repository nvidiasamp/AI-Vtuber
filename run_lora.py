from nemo.collections.llm import api
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Config1P5B, Qwen2Model
from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.gpt.data.chat import ChatDataModule
from nemo import lightning as nl
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from megatron.core.optimizer.optimizer_config import OptimizerConfig

def main():
    trainer = nl.Trainer(
        devices=1,
        max_steps=300,
        accelerator="gpu",
        strategy=nl.MegatronStrategy(tensor_model_parallel_size=1),
        precision="bf16-mixed",
        log_every_n_steps=10,
        val_check_interval=100,
        limit_val_batches=5,
        enable_checkpointing=True,
    )

    data = ChatDataModule(
        dataset_root="/workspace/nemo_finetune/data",
        global_batch_size=2,
        micro_batch_size=1,
        seq_length=2048,
    )

    model = Qwen2Model(Qwen2Config1P5B())

    peft = LoRA(
        target_modules=["linear_qkv", "linear_proj"],
        dim=32,
        alpha=64,
    )

    logger = nl.NeMoLogger(
        log_dir="/output",
        name="nike_lora",
        ckpt=nl.ModelCheckpoint(
            save_last=True,
            save_on_train_epoch_end=True,
            save_optim_on_train_end=False,
            monitor="val_loss",
            save_top_k=1,
        ),
    )

    optim = nl.MegatronOptimizerModule(
        config=OptimizerConfig(
            optimizer="adam",
            lr=5e-5,
            weight_decay=0.01,
            bf16=True,
        )
    )

    resume = nl.AutoResume(
        restore_config=RestoreConfig(path="/models/qwen2.5-1.5b-nemo"),
        resume_if_exists=False,
    )

    api.finetune(
        model=model,
        data=data,
        trainer=trainer,
        peft=peft,
        log=logger,
        optim=optim,
        resume=resume,
    )

if __name__ == "__main__":
    main()
