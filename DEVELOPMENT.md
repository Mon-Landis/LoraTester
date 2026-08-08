# 开发环境

## 本机基线

| 项目 | 当前值 |
|---|---|
| ComfyUI | `0.29.2`，提交 `3221224`，2026-07-31 |
| ComfyUI 路径 | `D:\ComfyUI\ComfyUI_windows_portable\ComfyUI` |
| 插件开发路径 | `D:\ComfyUI\LoraTester` |
| 便携 Python | `3.13.14` |
| Torch | `2.13.0+cu130` |
| CUDA / GPU | CUDA 13.0 / NVIDIA GeForce RTX 2080 Ti |
| Pillow / NumPy | `12.3.0` / `2.5.1` |
| 系统 Python | `3.14.4`，不作为 ComfyUI 兼容性基准 |
| Node.js / npm | 当前未安装；纯 Python 节点不需要 |

开发、测试和预览均应使用便携版 Python。它启用了隔离路径，直接运行 `-c` 时不会自动加入当前工作目录；仓库脚本已经显式处理插件路径。

## 采样节点使用的 ComfyUI 接口

- `folder_paths.get_filename_list("loras")`：节点下拉列表。
- `folder_paths.get_full_path_or_raise("loras", name)`：安全解析 LoRA 文件。
- `comfy.utils.load_torch_file(..., safe_load=True, return_metadata=True)`：加载并缓存 LoRA。
- `comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)`：分别应用到 MODEL 与 CLIP。
- `clip.tokenize()`、`clip.encode_from_tokens_scheduled()`：在节点内编码追加触发词后的正负提示词。
- `nodes.common_ksampler()` 或等价的 `comfy.sample` 调用：固定 seed、noise、latent 和采样参数。
- `vae.decode(samples["samples"])`：节点内解码；五维输出需按标准 `VAEDecode` 逻辑压平成 IMAGE 批次。
- 本合成器的 `plan.tasks` 和 `CompositionSession.submit()`：流式拼图并复用 B 轴图。

LoRA 原始权重范围在 ComfyUI 标准加载器中允许负值。本项目的梯度倍率保持 `0.25 / 0.5 / 0.75 / 1.0`，因此负的最高权重会形成对应的负梯度。主节点已经使用与标准加载器一致的 `-100 .. 100` 输入范围。每组还提供 `min_strength`，默认 `0`，运行时要求 `min_strength < max_strength`；实际权重为 `min + (max - min) * multiplier`，所以中心格也会应用非零下限及其触发词。

## 开发联接

以下脚本只在目标不存在时创建目录联接；不会覆盖已有自定义节点目录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\link_to_comfy.ps1
```

本机执行策略禁止直接运行 `.ps1`，因此使用上面的进程级 `Bypass`；脚本不会修改系统或用户级执行策略。

目标为：

```text
D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\LoraTester
  -> D:\ComfyUI\LoraTester
```

根 `__init__.py` 当前注册 `LoraTesterSampler` 与 `LoraTesterStyle` 两个节点。

## 稳定节点字段

主采样节点包含：

```python
"show_lora_details": (
    "BOOLEAN",
    {"default": True},
)
```

稳定字段定义位于 `lora_tester/node_contract.py`。颜色模式字段为 `color_mode`，候选值固定为 `black / white / custom`。

## 前端边界

当前功能可完全使用 Python `INPUT_TYPES` 实现。只有在需要以下能力时才补 Node.js LTS 和 Web 前端扩展：

- 根据 `lora_count` 动态隐藏 B/C 输入；
- 展开式自定义样式编辑器；
- 颜色选择器之外的背景图和区域框交互预览；
- 执行中的 69 项任务进度矩阵。

没有这些需求时，引入前端构建链只会增加安装和版本兼容成本。

## ComfyUI 启动级检查

只加载本插件并使用内存数据库，避免与正在运行的 ComfyUI 抢占数据库锁：

```powershell
Set-Location D:\ComfyUI\ComfyUI_windows_portable\ComfyUI
..\python_embeded\python.exe main.py --quick-test-for-ci --disable-all-custom-nodes --whitelist-custom-nodes LoraTester --database-url "sqlite:///:memory:"
```

当前检查结果为插件导入 `0.0 seconds`、退出码 `0`，节点映射数量为 2；真实 `INPUT_TYPES` 调用也已通过。
