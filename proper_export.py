import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nemo.collections.llm.peft.lora import LoRAMerge
from nemo.collections.llm.peft.api import _load_base_model_and_lora, _setup_trainer_and_restore_model_and_adapter
from nemo.collections.llm.recipes.precision.mixed_precision import bf16_mixed
from lightning.pytorch import Trainer
from nemo.lightning import MegatronStrategy

ckpt_path = '/output/nike_lora/2026-06-23_03-01-18/checkpoints/nike_lora--val_loss=0.1842-epoch=0-consumed_samples=600.0-last'
output_path = '/models/qwen2.5-1.5b-nike-v2'

trainer = Trainer(
    devices=1,
    accelerator="gpu",
    strategy=MegatronStrategy(ddp="pytorch", setup_optimizers=False, plugins=bf16_mixed()),
)

print("モデルとLoRAロード・マージ中...")
model, lora = _load_base_model_and_lora(ckpt_path)
_setup_trainer_and_restore_model_and_adapter(Path(ckpt_path), trainer, model, lora)
lora_merge = LoRAMerge()
merged_model = lora_merge(trainer.strategy.megatron_parallel)

print("NeMoマージ済み重みを収集中...")
nemo_sd = {}
for k, v in merged_model.named_parameters():
    if '.adapter.' not in k:
        nemo_sd[k] = v.detach().cpu().to(torch.bfloat16)

# NeMo設定取得
megatron_config = trainer.strategy.megatron_parallel.config
head_num = megatron_config.num_attention_heads
num_query_groups = megatron_config.num_query_groups
heads_per_group = head_num // num_query_groups
head_size = megatron_config.kv_channels
qkv_total_dim = head_num + 2 * num_query_groups
print(f"head_num={head_num}, num_query_groups={num_query_groups}, head_size={head_size}")

def split_qkv(linear_qkv):
    w = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    hidden_size = w.size(-1)
    q_slice = torch.cat([
        torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
        for i in range(num_query_groups)
    ])
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
    q = w[q_slice].reshape(-1, hidden_size)
    k = w[k_slice].reshape(-1, hidden_size)
    v = w[v_slice].reshape(-1, hidden_size)
    return q, k, v

def split_qkv_bias(bias):
    b = bias.reshape([qkv_total_dim, head_size])
    q_slice = torch.cat([
        torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
        for i in range(num_query_groups)
    ])
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))
    return b[q_slice].reshape(-1), b[k_slice].reshape(-1), b[v_slice].reshape(-1)

def split_fc1(linear_fc1):
    half = linear_fc1.shape[0] // 2
    return linear_fc1[:half], linear_fc1[half:]

print("HFモデルにマッピング中...")
base_hf = AutoModelForCausalLM.from_pretrained('/models/qwen2.5-1.5b-instruct', torch_dtype=torch.bfloat16)
hf_sd = base_hf.state_dict()

def nemo_key(k):
    # "0.module." プレフィックスを除去
    return k.replace('0.module.', '').replace('.to_wrap', '')

nemo_sd_clean = {nemo_key(k): v for k, v in nemo_sd.items()}

nan_count = 0
updated = 0
for i in range(28):  # 28 layers
    prefix = f'decoder.layers.{i}'
    hf_prefix = f'model.layers.{i}'

    # linear_proj → o_proj
    key = f'{prefix}.self_attention.linear_proj.weight'
    if key in nemo_sd_clean:
        v = nemo_sd_clean[key]
        if torch.any(torch.isnan(v)):
            nan_count += 1
        hf_sd[f'{hf_prefix}.self_attn.o_proj.weight'] = v
        updated += 1

    # linear_qkv → q/k/v_proj
    key = f'{prefix}.self_attention.linear_qkv.weight'
    if key in nemo_sd_clean:
        v = nemo_sd_clean[key]
        q, k, v_proj = split_qkv(v)
        for tensor, name in [(q, 'q_proj'), (k, 'k_proj'), (v_proj, 'v_proj')]:
            if torch.any(torch.isnan(tensor)):
                nan_count += 1
            hf_sd[f'{hf_prefix}.self_attn.{name}.weight'] = tensor
        updated += 3

    # bias
    key = f'{prefix}.self_attention.linear_qkv.bias'
    if key in nemo_sd_clean:
        v = nemo_sd_clean[key]
        q_b, k_b, v_b = split_qkv_bias(v)
        hf_sd[f'{hf_prefix}.self_attn.q_proj.bias'] = q_b
        hf_sd[f'{hf_prefix}.self_attn.k_proj.bias'] = k_b
        hf_sd[f'{hf_prefix}.self_attn.v_proj.bias'] = v_b
        updated += 3

    # layernorm
    key = f'{prefix}.self_attention.linear_qkv.layer_norm_weight'
    if key in nemo_sd_clean:
        hf_sd[f'{hf_prefix}.input_layernorm.weight'] = nemo_sd_clean[key]
        updated += 1

    key = f'{prefix}.mlp.linear_fc1.layer_norm_weight'
    if key in nemo_sd_clean:
        hf_sd[f'{hf_prefix}.post_attention_layernorm.weight'] = nemo_sd_clean[key]
        updated += 1

    # mlp fc1 → gate/up proj
    key = f'{prefix}.mlp.linear_fc1.weight'
    if key in nemo_sd_clean:
        gate, up = split_fc1(nemo_sd_clean[key])
        hf_sd[f'{hf_prefix}.mlp.gate_proj.weight'] = gate
        hf_sd[f'{hf_prefix}.mlp.up_proj.weight'] = up
        updated += 2

    # mlp fc2 → down proj
    key = f'{prefix}.mlp.linear_fc2.weight'
    if key in nemo_sd_clean:
        hf_sd[f'{hf_prefix}.mlp.down_proj.weight'] = nemo_sd_clean[key]
        updated += 1

# embedding / norm / lm_head
vocab_size = 151936
if 'embedding.word_embeddings.weight' in nemo_sd_clean:
    w = nemo_sd_clean['embedding.word_embeddings.weight'][:vocab_size]
    hf_sd['model.embed_tokens.weight'] = w
    updated += 1

if 'decoder.final_layernorm.weight' in nemo_sd_clean:
    hf_sd['model.norm.weight'] = nemo_sd_clean['decoder.final_layernorm.weight']
    updated += 1

if 'output_layer.weight' in nemo_sd_clean:
    w = nemo_sd_clean['output_layer.weight'][:vocab_size]
    hf_sd['lm_head.weight'] = w
    updated += 1

print(f"更新済みパラメータ: {updated}個, NaN検出: {nan_count}個")

print("HFモデルに重みをロードして保存中...")
base_hf.load_state_dict(hf_sd)
tokenizer = AutoTokenizer.from_pretrained('/models/qwen2.5-1.5b-instruct')
base_hf.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

# config修正
import json
with open(f'{output_path}/config.json') as f:
    config = json.load(f)
config['torch_dtype'] = 'bfloat16'
with open(f'{output_path}/config.json', 'w') as f:
    json.dump(config, f, indent=2)

print(f"保存完了: {output_path}")
