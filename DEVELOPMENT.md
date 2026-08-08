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

主节点的三个 LoRA 槽位都使用 `*_min_strength` 与 `*_max_strength` 成对输入。节点构造 `LoraSpec` 时会校验最低权重严格小于最高权重；不要只在前端限制范围，后端校验是工作流/API 调用的最终边界。

## 前端扩展

`web/lora_tester.js` 当前已经提供以下行为：

- 根据 `lora_count` 动态隐藏 B/C 输入；
- 隐藏/恢复最小权重、最大权重、触发词和文件选择输入时保持节点尺寸稳定；
- 对颜色模式、背景适配、装饰器和详情开关进行 English/简体中文显示翻译；
- 加载已有工作流时按序列化的 `lora_count` 恢复可见分组。

前端扩展只改变显示状态和标签，不改变序列化字段值。后续若增加以下能力，再评估引入 Node.js 构建链：

- 展开式自定义样式编辑器；
- 颜色选择器之外的背景图和区域框交互预览；
- 执行中的 69 项任务进度矩阵。

当前脚本为原生 ES 模块，无需 npm 构建步骤。

## 测试与发布

所有测试应使用 ComfyUI 便携版 Python，而不是系统 Python：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
```

该脚本会调用 `D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe`。若直接执行测试，使用同一个解释器；系统 Python 可能没有 Torch。

建议的本地发布顺序：

1. 在独立 worktree 或备份分支修改并运行测试。
2. 检查 `git diff --check` 和 `git status`，确认没有生成无关文件。
3. 提交后快进合并到 `main`，确认 ComfyUI 的 `custom_nodes\LoraTester` 仍指向主体目录。
4. 重启 ComfyUI，确认节点输入契约和前端动态分组正常。
5. 推送 `main` 后再在生产实例中验证真实 LoRA 采样。

## ComfyUI 启动级检查

只加载本插件并使用内存数据库，避免与正在运行的 ComfyUI 抢占数据库锁：

```powershell
Set-Location D:\ComfyUI\ComfyUI_windows_portable\ComfyUI
..\python_embeded\python.exe main.py --quick-test-for-ci --disable-all-custom-nodes --whitelist-custom-nodes LoraTester --database-url "sqlite:///:memory:"
```

当前检查结果为插件导入 `0.0 seconds`、退出码 `0`，节点映射数量为 2；真实 `INPUT_TYPES` 调用也已通过。
