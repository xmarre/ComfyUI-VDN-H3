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

默认情况下，架构参数全部来自检查点。Advanced 节点只有在显式选择 `architecture_mode=override` 后才会覆盖部分参数；这些设置属于消融实验，不声称与训练时检查点完全一致。

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
| `lora_mode` | `merge` 或 `bypass`；见下方“LoRA adapter 模式” |
| `branch_weights` | `stream` 或 `resident`；它控制的是 **VDN 线性分支**，不是 LoRA 应用方式 |
| `attention_backend` | 默认 `grouped`，或可选 `flex`（失败时回退 grouped） |
| `verbose` | 输出额外布局/adapter 日志 |

### Apply VDN-H3 Advanced

增加独立的 Stage-B/Turbo 强度、可选 fast kernels 和显式架构消融。

`architecture_mode=checkpoint` 为默认模式。只有选择 `architecture_mode=override` 后才会应用 `window_radius`、`window_chunk`、`anchor_frames`、`text_state`、`linear_branch` 等消融字段。

## LoRA adapter 模式

`lora_mode` 与 `branch_weights` 解决的是两类不同显存问题，两者可以自由组合。

### `lora_mode=merge`

- 通过正常的 `ModelPatcher.add_patches()` 注册 Stage-B/Turbo adapter；
- backup/restore、load/offload、自定义权重转换与重新量化全部由 Comfy 管理；
- 这是当前参考/eager adapter 路径，也是输出质量验证时更保守的选择；
- 对量化基座，eager 反量化 -> 合并补丁 -> 重新量化可能产生较大的临时显存峰值。

### `lora_mode=bypass` — 安全的 runtime 低显存模式

为了兼容已有 workflow，仍保留 `bypass` 这个名字，但它**不再使用旧的 `BypassForwardHook` 实现**。

新的 runtime 模式：

- 使用 Comfy 公共的 `ModelPatcher.add_weight_wrapper()` / `weight_function` 生命周期；
- 不替换、不遍历、不拼接、不恢复任何 LoRA 目标的 `module.forward`；
- 不安装 VDN `PatcherInjection`，也不使用 `_vdn_live_hooks`；
- 常驻 base parameter 保持未合并状态；
- LoRA A/B 仍为低秩 tensor，不保留第二份完整尺寸的 patched weight；
- 每次只构造当前 layer 的临时 compute weight，并用原地 `addmm_` 累积 `B @ A`，避免再分配一份完整尺寸 delta；
- 对同一权重的 Stage-B 和 Turbo 项聚合到一个 runtime wrapper；
- wrapper 的复制、安装和移除全部由正常的 `ModelPatcher` clone/load/offload 生命周期处理。

这样保留低显存 adapter 选项，同时不再引入曾经导致 Continuum 在 chunk 2 第一次 transformer 调用之前递归崩溃的跨 provider `module.forward` 链。

对于 fused/quantized MiniMax-H3 模块，仍以 Comfy 的 cast 路径为准。运行时 weight wrapper 可能使被 patch 的 INT8 layer 在该调用中回退到反量化 compute 路径，而不是继续使用 fused INT8 kernel。因此这是显存与速度之间的实际取舍，必须在真实 workflow 上测量。

**输出质量说明：** 历史 bypass 测试使用的是旧的 forward-hook 实现，并且在 8-step DMD stage 上观察过质量差异。那些结果不能证明新的 weight-wrapper runtime 模式也有同样行为。在完成匹配 GPU 渲染之前，Stage-DMD 的质量对照仍以 `merge` 为参考，`bypass` 视为需要实测的低显存路径。

## Curve / pruned MiniMax-H3 基座

部分 MiniMax-H3 检查点将完整 time embedding 折叠为 `adaln_t_table`。完整宽度的 AdaLN LoRA 一般不能无损投影到较小的 curve basis。

因此本节点不会丢弃这些学习到的 adapter 权重，也不会用近似投影。对于 curve/pruned 基座，它恢复与基座匹配的 dense time-embedder 输入，然后运行原始低秩 AdaLN delta，同时保持 base curve projection 本身不变。这个精确 AdaLN 路径在普通 adapter 选择 `merge` 或 `bypass` 时都会使用。

节点会寻找：

1. VDN stage 目录中的 `dense_time_embedder.safetensors`；或
2. `models/diffusion_models` 中安装的匹配 dense MiniMax-H3 检查点。

可从匹配的 dense H3 检查点提取：

```bash
python tools/extract_h3_time_embedder.py \
  <path-to-dense-h3.safetensors> \
  <ComfyUI>/models/vdn/<stage>/dense_time_embedder.safetensors
```

如果无法确认兼容的 dense embedder，节点会明确失败，不会静默丢失 AdaLN adapter 参数。

## VDN Branch 权重驻留

这与 `lora_mode` 完全独立。

`branch_weights=stream`

- 不把完整 VDN linear branch 注册为常驻附加模型；
- 需要时按 block 从 stage 文件解析；
- safetensors 映射仅在单次加载期间存在。

`branch_weights=resident`

- 将 branch tensor 包装成单独的 Comfy `ModelPatcher`；
- 通过 additional model 注册，让 ComfyUI 管理 device、load 和 offload；
- 不再使用旧的未追踪全局 GPU branch cache。

量化 VDN branch 文件当前必须使用 `stream`；`resident` 会明确失败，而不是偷偷反量化。

因此低内存配置可以组合 `lora_mode=bypass` + `branch_weights=stream`：前者控制 adapter 是否合并，后者控制 VDN linear branch 是否常驻。

## 组合与生命周期

VDN 为 **VDN 混合注意力变换本身** 通过 Comfy object patch 替换 `diffusion_model.blocks.*.attn.forward`。如果其它扩展已经占用同一个 attention object-patch 目标，VDN 会明确拒绝叠加。

这与 LoRA runtime 模式是两回事。`lora_mode=bypass` 不会改写任何 LoRA 目标的 `module.forward`，而是使用 weight wrapper。

测试覆盖重复 `ModelPatcher` clone/load/unload，以及 pseudo-Continuum 序列：每个 chunk 都先执行真实崩溃路径的结构替身 `preprocess_text_embeds -> token_refiner.fc1`，再执行合成 transformer forward。runtime 模式还要求：

- 没有 VDN LoRA injection；
- `module.forward` owner 不变；
- 常驻 base weight 不变；
- 每个目标权重只有一个聚合 runtime wrapper；
- 重复 clone 输出稳定；
- 不发生 2x/3x adapter 累积；
- 强度变化始终从真实 base 重新计算。

这是 CPU 结构回归，不等同于真实 GPU Continuum、显存峰值或端到端速度验证。

## Attention backend

- `grouped`：便携默认路径；相同窗口的帧以 grouped dense SDPA 执行。
- `flex`：支持时使用 PyTorch FlexAttention；单次失败只对该调用回退 grouped，不修改共享 VDN 状态。
- full coverage：走 ComfyUI 普通完整 attention，线性补集关闭。

历史性能数据见 [Benchmarks.md](Benchmarks.md)。除非明确标注，否则旧数字来自本次 lifecycle/runtime-adapter 重构之前。

## 验证状态

CI 分为两个 lane：

1. **Pinned Comfy + official oracle**
   - ComfyUI `6c53f8c9a06d95f3d847009ceaae55c624169247`
   - OpenVDN `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`
   - 直接导入 OpenVDN 源码，在小尺寸 CPU tensor 上比较实现；
   - 同时覆盖独立数学、adapter 转换、ModelSpec/checkpoint、curve、量化/自定义权重和 lifecycle。
2. **Current Comfy main smoke**
   - 检出当前 Comfy `master`，验证包导入和节点注册。

绿色 CI 证明实现/结构/lifecycle contract，不证明真实渲染质量、峰值显存或端到端速度。

## 许可证与来源

**源码：** 本仓库源码使用 Apache License 2.0，起源于 Saganaki22 的 ComfyUI-VDN-H3，并移植/改编 OpenVDN 发布的 VDN-H3 架构与算法。见 `LICENSE` 和 `NOTICE`。

**OpenVDN：** OpenVDN 源代码仓库为 Apache-2.0。其 NOTICE 单独说明：VDN-H3 模型权重属于 MiniMax-H3 的衍生权重，按 MiniMax-H3 Community License Agreement 分发。

**模型/检查点权重：** 本仓库不会重新授权 MiniMax-H3 或 VDN-H3 权重。下载和使用仍受对应 MiniMax-H3 许可证、地域/资格等限制约束。

来源：

- OpenVDN VDN-H3: https://github.com/OpenVDN/vdn-minimax-h3
- 原始 ComfyUI 移植: https://github.com/Saganaki22/ComfyUI-VDN-H3
- ComfyUI: https://github.com/Comfy-Org/ComfyUI
- MiniMax-H3: https://huggingface.co/Comfy-Org/MiniMax-H3
- VDN-H3 weights: https://huggingface.co/OpenVDN/vdn-minimax-h3
