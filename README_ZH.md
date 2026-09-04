# ComfyUI-VDN-H3 — MiniMax-H3 的 VDN-H3

<img width="1039" height="505" alt="VDN-H3" src="https://github.com/user-attachments/assets/ab4c1691-bff5-46fe-8b3e-635429b0700f" />

**[English](README.md)**

这是 [OpenVDN VDN-H3](https://github.com/OpenVDN/vdn-minimax-h3) 发布版混合注意力架构在 ComfyUI 原生 MiniMax-H3 模型上的移植。

VDN-H3 在局部帧窗口内保留精确 softmax 注意力，并用双向 Video Delta Attention 线性分支覆盖窗口外的长距离时序上下文。本仓库直接读取官方 VDN stage 目录，不修改 ComfyUI 核心文件。

## 保留的发布架构

本移植按照检查点中的 `model_spec.json` 执行，包括：

- MiniMax-H3 的 text/video/audio 打包布局；
- 按帧或按 chunk 对齐的 softmax 窗口；
- `none` / `rows` / `columns` / `both` anchor 模式；
- 线性分支共享 softmax 分支的原始、QKNorm 前、RoPE 前 Q/K/V；
- 检查点指定的 Q/K/V 可分离短卷积；
- beta、逐帧 KDA alpha 和检查点指定的 delta rule；
- 正向与反向状态扫描；
- 可选 text state 和 alpha boundary bridge；
- branch RMSNorm、output gate、`to_out_linear`；
- 可选 softmax gate；
- 当窗口覆盖整个 clip 时使用完整稠密注意力，并关闭不存在的线性补集。

默认情况下，架构参数全部来自检查点。Advanced 节点只有在显式选择 `architecture_mode=override` 后才会覆盖部分参数；这些设置属于消融实验，不再声称与训练时检查点完全一致。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Saganaki22/ComfyUI-VDN-H3
```

将官方 VDN stage 保持原目录结构下载到 `ComfyUI/models/vdn/`：

```bash
hf download OpenVDN/vdn-minimax-h3 \
  --include "stage-dmd-step-250/*" \
  --local-dir <ComfyUI>/models/vdn
```

stage 应保持官方结构，包括 `model_spec.json`、`linear_branch/` 和 `adapters/`。

官方当前发布：

- `stage-dmd-step-250`：VDN-H3 8-step stage，包含 Turbo/DMD adapter；
- `stage-b-step-2000`：VDN-H3 50-step stage，包含 Stage-B/default adapter。

**模型/检查点权重不是 Apache-2.0。** 下载或使用前请阅读下方“许可证与来源”。

## 节点

### Apply VDN-H3 (MiniMax-H3 Hybrid Attention)

`MODEL -> MODEL`

| 输入 | 含义 |
|---|---|
| `vdn_checkpoint` | `models/vdn/` 下的官方 stage 目录 |
| `apply_turbo_adapter` | stage 存在 Turbo/DMD adapter 时应用；发布的 8-step DMD stage 需要开启 |
| `strength` | adapter 强度；`1.0` 为发布设置 |
| `lora_mode` | 仅 `merge`：通过 ComfyUI 原生 `ModelPatcher` 权重补丁生命周期应用 adapter |
| `branch_weights` | `stream` 或 `resident` |
| `attention_backend` | 默认 `grouped`，或可选 `flex`（失败时回退 grouped） |
| `verbose` | 输出额外布局/adapter 日志 |

旧的 VDN `bypass` LoRA 模式已移除。VDN 不再安装、遍历或修复可变的 `module.forward` LoRA bypass 链。adapter 正确性不应依赖注入/弹出顺序、clone 顺序、其它 wrapper provider 或 Continuum chunk 生命周期。

### Apply VDN-H3 Advanced

增加独立的 Stage-B/Turbo 强度、可选 fast kernels 和显式架构消融。

`architecture_mode=checkpoint` 为默认模式，此时不会使用下列消融字段。只有选择 `architecture_mode=override` 后才会应用：

- `window_radius`
- `window_chunk`
- `anchor_frames`
- `text_state`
- `linear_branch`

如果 override 与检查点训练架构不同，节点会在控制台明确记录差异。

`fast_kernels` 可用 `torch.compile` 融合部分线性分支热点。算法保持一致，但 BF16 的舍入位置可能因 kernel 融合而改变，因此不承诺逐位一致；编译失败会自动回退 eager。

## Adapter 生命周期与量化基座

普通 VDN LoRA 目标通过 ComfyUI 的 `ModelPatcher.add_patches` 注册。权重 backup/restore、clone、load/offload 均由 ComfyUI 管理。

对于 fused/quantized MiniMax-H3 模块，VDN 不自行永久反量化或替换模块，而是交给当前 ComfyUI 的 `convert_weight` / `set_weight` 抽象处理。CI 有合成量化权重回归测试，验证补丁与恢复路径；不同真实 GPU 量化布局仍需要实际渲染验证。

Q/K/V adapter 转换通过可变 rank 的 block-diagonal 融合精确保留各自的 LoRA rank 和 `alpha/rank` scale。缺失、不完整或形状错误的 adapter 会在应用前失败，不会静默跳过。

## Curve / pruned MiniMax-H3 基座

部分 MiniMax-H3 检查点将完整 time embedding 折叠为 `adaln_t_table`。完整宽度的 AdaLN LoRA 一般不能无损投影到这个较小的 curve basis。

因此本节点不会丢弃这些学习到的 adapter 权重，也不会用近似投影。对于 curve/pruned 基座，它恢复与基座匹配的 dense time-embedder 输入，然后运行原始低秩 AdaLN delta，同时保持基座 curve projection 本身不变。

要做到这一点，需要与 curve 基座匹配的 dense time embedder。节点会寻找：

1. VDN stage 目录中的 `dense_time_embedder.safetensors`；或
2. `models/diffusion_models` 中安装的匹配 dense MiniMax-H3 检查点。

可从匹配的 dense H3 检查点提取这个小 companion：

```bash
python tools/extract_h3_time_embedder.py \
  <path-to-dense-h3.safetensors> \
  <ComfyUI>/models/vdn/<stage>/dense_time_embedder.safetensors
```

如果无法确认兼容的 dense embedder，节点会明确失败。它不会静默丢失 AdaLN adapter 参数。

## Branch 权重驻留

`branch_weights=stream`

- 不把完整 VDN branch 注册为常驻附加模型；
- 需要时按 block 从 stage 文件解析；
- safetensors 映射仅在单次加载期间存在，不保留无限生命周期的全局 mmap handle。

`branch_weights=resident`

- 将 branch tensor 包装成单独的 Comfy `ModelPatcher`；
- 通过 additional model 注册，让 ComfyUI 管理 device、load 和 offload；
- 不再使用旧的未追踪全局 GPU branch cache。

量化 VDN branch 文件当前必须使用 `stream`。`resident` 会明确失败，而不是偷偷反量化。

## 组合与生命周期

VDN 通过 Comfy object patch 替换 `diffusion_model.blocks.*.attn.forward`。如果其它扩展已经拥有相同 object-patch 目标，VDN 会拒绝叠加，而不是构造不确定的 forward 链。

普通模型权重 LoRA/patch 走 Comfy 的另一套权重生命周期；VDN 不遍历、不重排这些 provider 的 `module.forward`。

测试包含重复 `ModelPatcher` clone/load/unload，以及 pseudo-Continuum 序列：每个 chunk 都先执行曾经真实崩溃的 conditioning 路径 `preprocess_text_embeds -> token_refiner.fc1`，再执行合成 transformer forward。测试检查：

- 输出稳定；
- base 权重恢复；
- forward owner 不被 VDN 改写；
- 不会发生 2x/3x adapter 累积。

这是 CPU 结构回归测试，不等同于真实 GPU Continuum 渲染验证。

## Attention backend

- `grouped`：便携默认路径；相同窗口的帧以 grouped dense SDPA 执行。
- `flex`：环境支持时使用 PyTorch FlexAttention；单次失败只对该调用回退 grouped，不会修改共享 VDN 状态。
- full coverage：直接走 ComfyUI 普通优化后的完整 attention，线性分支关闭，因为不存在窗口外补集。

历史性能数据见 [Benchmarks.md](Benchmarks.md)。除非明确标注，否则其中数字来自本次生命周期重构之前，不应当作当前分支的性能验证。

## 验证状态

CI 分为两个独立 lane：

1. **Pinned Comfy + official oracle**
   - ComfyUI `6c53f8c9a06d95f3d847009ceaae55c624169247`
   - OpenVDN `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`
   - 直接导入 OpenVDN 源码，在小尺寸 CPU 张量上比较实现；
   - 同时覆盖独立数学、adapter 转换、ModelSpec/checkpoint、curve、量化补丁和生命周期。
2. **Current Comfy main smoke**
   - 每次 CI 检出当前 Comfy `master`，验证包导入和节点注册。

直接 oracle 覆盖 window bounds/anchors、frame statistics、所有支持的 delta rule、正反向 scan、alpha bridge、feature/short-conv 以及完整 `BidirectionalLinearBranch`。当前 pinned suite 为 **70 tests passed**。

CI 不下载大型模型，也不执行 GPU 渲染。因此绿色 CI 证明的是实现/结构/lifecycle contract，不证明真实渲染质量、显存峰值或端到端速度。

## 许可证与来源

**源码：** 本仓库源码使用 Apache License 2.0，起源于 Saganaki22 的 ComfyUI-VDN-H3，并移植/改编 OpenVDN 发布的 VDN-H3 架构与算法。见 `LICENSE` 和 `NOTICE`。

**OpenVDN：** OpenVDN 源代码仓库为 Apache-2.0。其 NOTICE 单独说明：VDN-H3 模型权重属于 MiniMax-H3 的衍生权重，按 MiniMax-H3 Community License Agreement 分发。

**模型/检查点权重：** 本仓库不会重新授权 MiniMax-H3 或 VDN-H3 权重。下载和使用这些权重仍受对应 MiniMax-H3 许可证、地域/资格等限制约束。

来源：

- OpenVDN VDN-H3: https://github.com/OpenVDN/vdn-minimax-h3
- 原始 ComfyUI 移植: https://github.com/Saganaki22/ComfyUI-VDN-H3
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- MiniMax-H3: https://huggingface.co/Comfy-Org/MiniMax-H3
- VDN-H3 weights: https://huggingface.co/OpenVDN/vdn-minimax-h3