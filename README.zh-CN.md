# GPA 代码整理版

**v0.4 新增 Qwen2.5／Qwen3 全层转换为 GPA，以及 FineWeb 适配训练。**
支持继承原投影/MLP/Embedding/Norm、原生 LoRA、完整 checkpoint、恢复训练、合并导出，
以及带状态/短卷积缓存的 `auto_map` 生成。见[转换与训练说明](docs/QWEN_CONVERSION.md)。
本轮不把下面 tiny 模型的检索成绩当作转换后 Qwen 大模型的成绩。

v0.3 更新两阶段 GDN 的 decay/control、配对关系写入及**全连接兼容度乘积读取**，
另外实现衰减计数 KV 备选模型 `PaperDenseGPA`。不采用 RSA 的相邻/R32 稀疏读取。旧版与原始结果保留。
公式、来源与新对照实验见 [v0.3 说明](docs/DENSE_V3.md)。

当前选出的 49.4 万参数模型仅用答案 CE，从随机初始化训练。独立 T256 测试中，
四 token 标识符整串生成正确率 **93.36%**，同宽同深 Transformer 为 **96.09%**；
单值、MQAR 与覆盖更新两者均为 100%。这是合成小实验，尚非官方 RULER 或百分百检索。
同权重的新分级调度将 GPA 训练步耗时在 T256/T1024 分别缩短约 **1.99×/2.82×**，
Transformer 仍更快。详见[本轮完整报告](docs/REPORT_V3.zh-CN.md)，含公式、各项收益、
失败实验、效率及复现命令。直接复现使用 `bash scripts/run_selected_v3.sh`。

[English](README.md) · [完整公式](docs/EQUATIONS.md) · [验证记录](docs/VALIDATION.md) · [上传说明](docs/RELEASE.md)

v0.2 新增各级/各条边的 GDN 打包调度、Triton 稠密 route 前后向、固定形状
CUDA Graph，以及混合合成检索 LM 实验。见[效率与实验记录](docs/OPTIMIZATION.md)。

历史 v0.2 单 A100 完整测试 **58 项通过**。工程加速有效，但当时的小模型检索**尚未达到
Transformer 对照水平**；不能把算子正确/变快等同于模型质量达标。
中文总结见[单卡实验报告](docs/REPORT.zh-CN.md)。可选 query 对齐方案结果有好有坏，未设为默认。

本目录独立于正在训练的 Qwen/RSA 代码，不改动现有训练。包含：

1. 五个命题逐项对应的、小规模可核对的 PyTorch 实现。
2. 单级采用两阶段 GDN、多级采用**完整稠密动态关系矩阵**的 GPA 核心。
3. 可学习初始状态、计数、残差存储、双通道读取、可选位置矩与缓存。
4. 默认 PyTorch 参考后端和可选 FLA/Triton GDN 分块后端，含前后向测试。

五个命题参考实现基于作者确认的 **2026-10-24 `neurips_2026.tex`**；v0.3 又核对了
**2026-11-02** 源稿与最新 RSA 代码。源稿哈希与实现边界见
[PROVENANCE.md](docs/PROVENANCE.md)。

## 最关键的结构变化

存储仍按残差逐级进行，但**读取不再是“前级指导后级”**：

* 每级维护 `A_i: Dk×M_i`、`B_i: M_i×Dv` 两阶段状态，起始状态可学习。
* `slot_i = center_i @ k_i` 为直接线性投影；M_i 可以不同于 Dk。
* 对所有 `i != j`，维护完整 `R_ij: M_i×M_j`，包括后级到前级。
  R 不是仅可学习的静态连接，也不是之前相邻连接的低秩替代。
* 每个时刻先完成各级写入，然后计算各级本地 route，再同步聚合所有
  入边消息。消息使用同一快照，不依赖读取循环的先后次序。
* 最终仍有 `qout_i = route_i W_rs_i + (q/16) W_qs_i`，再读取第二阶段状态。
* R 的更新由写入分配决定；Q 只参与读，不会反过来改变存储状态。

完整符号、方向、形状、更新与读取式见 [EQUATIONS.md](docs/EQUATIONS.md)。
本版 g 和 beta 均为 `[B,T,H]` 标量门。标量门不存在沿 K 轴/V 轴之分；
不能把这种结论推广到 vector gate，所以接口明确拒绝向量 gate。

## 五个命题的代码位置

| 命题 | `src/gpa/propositions.py` 中的实现 | 验证重点 |
|---|---|---|
| 1：冗余与计数 | `ExactKVCache` | 只合并联合 K,V **位模式完全相同**的条目，加 log(count) 后与原注意力一致；同 K 不同 V 不合并。 |
| 2：内容与位置分离 | `DecoupledRoPESequence` | 重建一致；不隐瞒其仍保留 O(T) token 索引。 |
| 3：位置压缩 | `PositionalMomentCache` | 受控小角度的一阶近似、Taylor 余项、负权重报错；不宣称任意长位置无损。 |
| 4：层间容量 | `analyze_interlayer_similarity` | 检查匹配碰撞与分离条件，条件不成立时不认证容量界。 |
| 5：多级残差 | `ResidualCodebook` | 残差分解、索引组合数与不同重建数的区别、带前提的收缩误差界。 |

另外实现了 `counted_slot_step`，用于核对原稿的计数归一化 soft centroid 更新。
它与工程版 GDN 分开，**不声称 GDN 严格等价于五个命题中全部无损操作**。
命题诊断代码可能做小规模枚举/两两比较，不属于线性复杂度运行路径。

## 安装与快速测试

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python examples/smoke.py
python examples/cache_decode.py
python examples/propositions.py
```

```python
import torch
from gpa import GPAConfig, DenseGPA

model = DenseGPA(GPAConfig(
    heads=4, key_dim=32, value_dim=48,
    slots=(64, 64, 64, 128),
))
q = torch.randn(2, 16, 4, 32)
k = torch.randn_like(q)
v = torch.randn(2, 16, 4, 48)
out, state = model(q, k, v)
out.square().mean().backward()
```

这是接收已投影 Q/K/V 的 attention core，不是完整预训练 LM。调用者需要
提供投影、FFN 和 generation adapter。接口**不自动忽略 padding**；应使用
无 padding 的等长流或独立缓存。新文档用 `state=None`；连续 chunk 必须
传回 state。默认不会 detach，若要截断反传请显式 `state.detach()`。

### CUDA / FLA

```bash
python -m pip install -e '.[test,cuda]'
python -m pytest -q tests/test_cuda.py
python examples/smoke.py --device cuda --backend fla --chunk-size 64
python scripts/benchmark.py --device cuda --backend fla --length 1024 --backward
```

FLA 为外部依赖，未复制其源码。本版只把 GDN scan 交给 FLA/Triton，
**没有把整个稠密多级模型描述成一个完全融合的 Triton kernel**，也没有
未经测量就宣称与旧 RSA/GDN 等速。真实测试环境与结果见验证记录。

## 两种执行语义必须区分

等 slot 的分块模式可设置 `execution="packed"`，把各级与各条 R 边分别
并入 head 维；`execution="triton"` 再融合稠密消息聚合、归一化与 route
softmax，前后向都有实现。四级每块由 20 次 GDN scan 调用减为 3 次，
没有删边或改块边界。不等 slot 或逐 token 模式仍走参考路径。

```python
cfg = GPAConfig(heads=4, key_dim=32, value_dim=32, slots=(16,)*4,
                chunk_size=64, backend="fla", execution="triton")
model = DenseGPA(cfg).cuda()  # 可保留 FP32 主权重，计算时用 bf16 autocast
```

* `chunk_size=1, backend="torch"`：逐 token 更新后计算残差的完整定义。
* `chunk_size>1`：为方便分块并行，用块起点冻结的 A/B 计算残差和写入
  分配，是明确标注的**块滞后近似**；其余 GDN、R、count 仍因果更新。
  可选 `backend="fla"`。不声称它与逐 token 残差完全等价。

块边界由缓存中的 offset 和 frozen snapshot 维护，不随外部 forward 的
切分改变，因此 prefill、任意切段和逐 token decode 遵循同一分块定义。
训练缓存保留梯度；跨块反传不自动截断。

默认 `positional_mode="gated"` 不使用显式 RoPE；`"moments"` 增加位置矩；
`"none"` 关闭 decay/位置矩。三种模式都保留计数。

## 复杂度与限制

固定级数、slot、维度及 chunk 大小时，对序列长度 T 线性。稠密 R 的成本
为 `sum_{i!=j} M_i M_j`，即等宽时 **O(L²M²)**，不是免费连接。训练的完整
计算图/激活也不是 O(1) 显存。固定缓存形状不等于无限精度地记住任意序列；
计数的数值范围仍增长。新增模型尚不能据此宣称复现论文准确率或 NIAH 满分。

## 打包与上传

### 简单训练与效率对比

```bash
python scripts/compare_efficiency.py --cuda-graph --output efficiency.json
python scripts/train_retrieval.py --steps 400 --output runs/gpa
python scripts/train_retrieval.py --attention transformer --steps 400 --output runs/transformer
python scripts/train_retrieval.py --resume runs/gpa/checkpoint-400.pt --steps 2000 \
    --easy-curriculum --cuda-graph --output runs/gpa_curriculum
python scripts/analyze_retrieval.py runs/gpa_curriculum/checkpoint-2000.pt --plot
```

小 LM 为 width=128、两层，混合单值、随机四 token 标识符、MQAR 和覆盖更新
任务，测试流使用独立种子。它不是 Qwen，也不是官方 RULER NIAH3；整串正确率
必须单独报告，不能只看 token accuracy。Transformer 对照宽度、深度与短卷积
相同，但参数量不同。训练使用 `read_scale=8、beta_init=.6、decay_init=.0001`，
与 attention core 默认初始化明确区分。首次编译/capture 不计入稳态计时。

```bash
python scripts/build_release.py
```

输出 `dist/gpa-attention-0.3.0-source.zip` 和 SHA256；包内另有逐文件校验清单。
不包含权重、训练数据、论文、机器地址、环境和私有日志。本次不自动公开上传；
可由作者按 [RELEASE.md](docs/RELEASE.md) 上传到 `fisalt/GPA` 的 Release。
发布前请权利人选择许可证，本次未擅自给项目赋予开源许可证。
