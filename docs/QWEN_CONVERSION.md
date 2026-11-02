# Qwen2.5 / Qwen3 → GPA：转换、FineWeb 续训与完整导出

v0.4 提供独立的 Hugging Face 因果 LM 封装。支持原生 `model_type=qwen2`
（包含 Qwen2.5）和 `qwen3`；不是将已有 RSA checkpoint 直接改名成 GPA。
相关代码不依赖私有 RSA 训练目录。

## 模型和继承范围

所有 decoder 层的 token Softmax Attention 都替换为 GPA，保持原来的：

* Embedding、LM head 及原模型的权重共享设置。
* 每层 Q/K/V/O 投影与 bias、原始 GQA 的 KV 投影宽度。
* Qwen3 的 Q/K RMSNorm、各层输入/输出 Norm 与 MLP。

转换器复用原参数对象，并检查对象身份和参数版本没有改变。新增 GPA 参数独立初始化，
不声称保留原 Softmax Attention 的函数或初始 loss；需要语料适配。
Qwen3 的 `num_heads*head_dim` 可不同于 `hidden_size`，实现不假设二者相等。

GQA 的 K/V 投影输出按组扩展到 query heads；GPA 状态因此按 query heads 存储。
每层新增 A/B、计数 n、**所有有向级对的完整 R**、route MLP 和双通道定标。
配置沿用 v0.3 配对写入、分层 decay、稠密兼容度乘积读取；没有换成 RSA 的相邻稀疏读取。
原始 QKV 短卷积以恒等通路初始化，独立因果控制卷积以均值窗口初始化。
`--quadratic-features` 可选；默认不修改继承的 Q/K 角特征。

默认每层两级 slot=`head_dim`，使 `Dk=M` 的分级调度可用；例如 native head_dim128
对应128,128。可通过 `--slots 64,64,64,64` 等自定义。非方形状态仍正确，
但会回退到原 packed/chunk 路径，不能假称同样有 staged 提速。
稠密关系容量随 `L*(L-1)*M²` 增长，不建议直接照搬 RSA 的高层数低秩预算。
`--last-slots` 可独立设置最后一个 decoder 层，例如前层 `--slots 64,64,64,64`，
末层 `--last-slots 128,128,128,128,128,128,128,128`。此时仍是完整稠密 R，
显存和速度成本与 RSA 的稀疏关系不同；优先先验证较小配置。

默认不加显式 RoPE；`--keep-rope` 保留原 Qwen 的 RoPE 模块、频率和半维旋转约定。
这是架构选项，并非“去 RoPE 一定更好”的结论。转换后的大模型尚没有在本项目取得
v0.3 小合成模型的检索成绩；旧表不能迁移成 Qwen-GPA 的性能声明。

`GPAQwenForCausalLM` 使用自己的 decoder loop，既不调用 token attention，也不建立
T×T 因果掩码。暂仅支持无 padding 的等长输入（训练时文本打包，生成可 batch1）；
含0的 padding mask 会明确报错，不会悄悄污染记忆。

## 安装

```bash
git clone https://github.com/fisalt/GPA.git
cd GPA
python -m pip install -e '.[hf,test]'
# CUDA 训练另需与 PyTorch/CUDA 匹配的公开 FLA/Triton：
python -m pip install flash-linear-attention
# 只有需要 ZeRO 时才安装：
# python -m pip install -e '.[deepspeed]'
```

HF 集成要求 PyTorch>=2.6、Transformers>=4.51；基础 GPA 核心仍可单独运行。
较新的 Transformers 对恢复优化器的 torch.load 有安全版本要求，因此未声明支持旧版 torch 续训。
建议转换、训练与评测使用同一 Transformers 版本；未声称跨主版本 checkpoint 互载数值等价。
完整 checkpoint 采用 FP32 master 权重，CUDA 使用 BF16 autocast 计算。
不要将完整 GPA checkpoint 的所有参数强制加载成 BF16：加载函数会明确拒绝该操作，
避免将学习到的 FP32 前置状态和控制参数静默舍入。ZeRO 的参数/优化器精度遵循其配置。

## 转换

```bash
python scripts/convert_qwen_to_gpa.py \
  --model Qwen/Qwen3-0.6B --output runs/qwen3_gpa_base \
  --backend fla --execution staged

# 或另一个独立目录：
python scripts/convert_qwen_to_gpa.py \
  --model Qwen/Qwen2.5-0.5B --output runs/qwen25_gpa_base \
  --backend fla --execution staged --keep-rope
```

`--model` 也接受本地完整原生 Qwen 权重。转换输出包含模型、tokenizer、
`conversion.json` 和 `config.json` 中的 `auto_map`。已有非空输出目录不会被覆盖。
CPU 检查可用 `--backend torch --execution packed`；这不是长序列高效训练后端。

## FineWeb / FineWeb-Edu 训练

默认数据源是官方 [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)。
也支持本地 parquet/json/jsonl。只做普通全 token 的 next-token CE，不混合检索合成数据，
没有 key 对齐、蒸馏或其他辅助损失。EOS 分隔文档，每个训练 block 使用新的 GPA 起始记忆；
文档可以跨 block，此处没有跨 batch 的状态携带或截断 BPTT。

```bash
python scripts/train_gpa_fineweb.py \
  --model runs/qwen3_gpa_base --output runs/qwen3_gpa_fineweb \
  --data-files '/path/to/fineweb-edu/data/**/*.parquet' \
  --length 8192 --microbatch 1 --tokens-per-update 524288 \
  --tuning lora --lora-rank 32 --lr 1e-4 --max-steps 10000

# 不指定 --data-files 就从官方数据集流式读取。
# 分布式训练：
torchrun --nproc_per_node=8 scripts/train_gpa_fineweb.py \
  --model runs/qwen3_gpa_base --output runs/qwen3_gpa_fineweb_8gpu \
  --data-files '/path/to/data/*.parquet' --length 8192 \
  --microbatch 1 --tokens-per-update 524288 --deepspeed ds_configs/stage3.json
```

默认微批次1、每次更新524,288个**输入 token**；监督目标略少，因为每 block 丢弃首位置的
next-token target。梯度累积自动按 `world*microbatch*length` 计算，不能整除时明确报错；
例如六卡可显式 `--accumulation 10`，实际 token 数会写入运行记录。
默认学习率1e-4、梯度裁剪1、余弦日程、非重入 activation checkpoint。
LM head 的 CE 按512个位置分块并 checkpoint，不保留整个 T×vocab 的训练 logits。
因此有 labels 时返回 loss，`outputs.logits=None`；推理时正常返回 logits。

本地目录 `--data-files /path/to/fineweb-directory` 会递归寻找 parquet。
通配符由脚本展开；`**` 使用递归 glob。数据集本身不按 rank 二次切分，
仅由 Accelerate 对连续 batch 分片；强制 DataLoader workers=0，避免重复或双重分片。
恢复时 Trainer 按保存步数重放/跳过流，不声称保存了精确的外部数据流游标。
数据文件、顺序、seed、world size 与软件版本变化会改变恢复样本顺序。

### 默认哪些权重训练

`--tuning lora` 使用项目内的轻量 LoRA：
`Linear(x)=W x + (alpha/r) B A x`，初始 B=0。
这是原生 LoRA，不是 PEFT 的 adapter-only 文件格式。

| 参数 | 默认策略 |
|---|---|
| Q/K/V/O 与 MLP gate/up/down | 原权重冻结，训练 LoRA A/B，rank32、alpha64 |
| 新 GPA 状态前缀、所有 R、decay/beta、route、中心和控制参数 | 完整训练 |
| 新短卷积 | 完整训练 |
| Embedding、所有 Norm（包括 Qwen3 Q/K Norm） | 完整训练 |
| 独立且不共享的 LM head | 冻结；共享 head 随 Embedding 训练 |
| 当前模式不用的位置矩参数、空参数 | 冻结，避免 DDP unused parameter 问题 |

`--freeze-embedding` / `--freeze-norm` 可以关闭额外训练；`--tuning full` 是全参适配，
`--tuning gpa-only` 冻结原线性投影（Embedding/Norm 仍由上述开关控制）。
输出目录的 `trainable_audit.json` 列出每个可训练参数的名称、形状、dtype 和数量，
`training_run.json` 记录实际 token 数、分片方式和语料来源。
审计是 ZeRO/Accelerate 接管前的参数策略；引擎的运行 dtype 以其配置为准。

### 完整保存、恢复与合并

所有 `checkpoint-*` 都是**完整 GPA 模型**，包含原权重、LoRA、新状态、Embedding/Norm，
另外有 Trainer 的优化器、调度器和 RNG 状态。没有“只保存 LoRA 却漏掉 GPA”的路径。
ZeRO-3 配置启用了保存时 gather；不能关闭后仍假称 `from_pretrained` 可直接加载。
大模型保存需要足够 CPU RAM 和磁盘空间；先加载完整模型再建立 ZeRO 初始化钩子，
单机不一定能用仅分片容量加载任意大的基座。

```bash
python scripts/train_gpa_fineweb.py \
  --model runs/qwen3_gpa_base --output runs/qwen3_gpa_fineweb \
  --resume runs/qwen3_gpa_fineweb/checkpoint-1000 \
  --data-files '/path/to/data/*.parquet' --length 8192 --max-steps 10000

python scripts/convert_qwen_to_gpa.py \
  --model runs/qwen3_gpa_fineweb/final --output runs/qwen3_gpa_merged --merge-lora
```

恢复严格加载所有参数，不重置 GPA 或 LoRA。模型配置中保存 rank/alpha/目标层，
构造时自动恢复相同结构。合并将 `(alpha/r) BA` 加入对应原投影，移除 LoRA 模块，
保留全部 GPA 状态与其他已训练权重，导出可独立加载的完整模型。
`--resume` 会恢复原优化器/调度器；若要继承权重但重新设置学习率日程，用
`--model checkpoint目录 --output 新目录`，不传 `--resume`。
共享 Embedding/LM head 的 checkpoint 只存一份权重；Trainer 可能打印
`lm_head.weight` alias 缺失提示，完整加载和共享关系仍需通过保存/恢复测试验证。

## tmux 与一键脚本

```bash
mkdir -p logs
tmux new-session -d -s gpa_qwen3 \
  'MODEL=Qwen/Qwen3-0.6B NPROC_PER_NODE=1 DATA_FILES="/path/to/data/*.parquet" bash scripts/run_qwen_gpa_fineweb.sh > logs/gpa_qwen3.log 2>&1'
tail -f logs/gpa_qwen3.log

# 多卡 + ZeRO-3：
# NPROC_PER_NODE=8 DEEPSPEED=ds_configs/stage3.json bash scripts/run_qwen_gpa_fineweb.sh
# 16K/32K：调整 LENGTH；内存仍不足时减微批次或增大 checkpoint/CE 分块控制。
```

脚本支持 MODEL、CONVERTED、OUTPUT、LENGTH、MICROBATCH、TOKENS_PER_UPDATE、LR、
MAX_STEPS、LORA_RANK、SLOTS、LAST_SLOTS、KEEP_ROPE、QUADRATIC_FEATURES、DEEPSPEED 和 RESUME。
不自动停止任何已有训练。环境镜像、conda 和代理由调用者设置，不硬编码服务器地址。

## AutoModel 和状态缓存

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = 'runs/qwen3_gpa_merged'
# gpa 包仍需安装；auto_map 会带上 HF 包装代码，不复制整个依赖库。
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True).cuda().eval()
tok = AutoTokenizer.from_pretrained(path)
inputs = tok('Explain associative memory briefly.', return_tensors='pt').to('cuda')
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    answer = model.generate(**inputs, max_new_tokens=64, use_cache=True)
print(tok.decode(answer[0], skip_special_tokens=True))
```

缓存保存 A/B/n/R、固定 chunk offset/frozen 快照以及短卷积尾部，不保存增长的 KV 序列。
自定义 cache 路径不会创建 HF `DynamicCache`；beam 重排同时重排所有状态和尾部。
不自动 detach。训练中的 activation checkpoint 与可变缓存不混用。
不支持 padding、speculative cache rollback 或 torch.compile 全图缓存。

## 验证范围

`tests/test_hf_qwen.py` 覆盖两种 native 架构、GQA、Qwen3 非标准 head_dim、
可选 RoPE、全序列/分段缓存一致性、无 token attention、完整 LoRA/auto_map 保存重载、
梯度、合并和 greedy/beam 生成。`tests/test_finetune_data.py` 检查文本打包与单次分布式分片。

```bash
python -m pytest -q tests/test_hf_qwen.py tests/test_finetune_data.py
python scripts/smoke_qwen_conversion.py --output runs/conversion_smoke
```

端到端 smoke 使用随机 tiny Qwen 和本地文本 fixture，包含转换、三步训练，
再从中途 step2 恢复到第三步，并逐项核对与未中断训练权重一致，最后合并和 auto_map 生成。
它验证工程链路，不是 FineWeb 大模型已收敛或 NIAH 性能证明。
实际运行版本、结果和未测试项见 [VALIDATION_V4.md](VALIDATION_V4.md)。
