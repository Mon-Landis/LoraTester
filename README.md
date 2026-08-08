# ComfyUI LoRA Tester

这是一个基于标准 KSampler 流程的 ComfyUI 自定义节点。主节点直接接收 `MODEL / CLIP / VAE / LATENT`，在节点内完成提示词编码、LoRA 叠加、采样、VAE 解码和最终拼图；不再需要外接 `CLIP Text Encode` 与 `VAE Decode`。

视觉原则是把结果做成可追踪的实验样张：图片是第一视觉层级，严格矩阵、区域框、侧边权重轴和少量 A/B/C 功能色负责说明关系。同一区域不逐格重复标注权重，中心原图不加标签；左下三 LoRA 混合区的说明贴紧对应图片下边缘并占用下方空格，最底排说明与底轴共用下沿空间，因为这些组合无法仅靠二维轴唯一表达。它借鉴编辑式工业信息设计，但不使用官方素材，也不把结果做成黑黄警告条或通用军事 HUD。

## 已支持

- 1 LoRA：5 个唯一任务、5 个位置。
- 2 LoRA：25 个唯一任务、25 个位置。
- 3 LoRA：69 个唯一任务、73 个位置；4 张 B 独立图在负 x/负 y 轴复用。
- 黑底白字、白底黑字、自定义三种颜色模式。
- 自定义背景颜色或图片、背景适配方式、区域框、文字颜色、A/B/C 功能色、间距和字体；背景可直接使用路径、PIL、NumPy 或单张 Comfy IMAGE 张量。
- 可注册 `StyleDecorator`，在背景和前景阶段增加项目自己的风格层。
- 支持 PIL、NumPy、ComfyUI `[B,H,W,C]` Torch 张量。
- 支持严格任务 ID、顺序队列和流式提交，不需要长期保存全部源图。
- 默认用区域侧轴标注 LoRA 名称和四档实际权重，左右侧的名称与数值统一竖向显示；自动字号根据整张矩阵的像素跨度缩放，也可用样式配置手动覆盖。
- 主区域使用约两倍于普通单元格的自动分隔；左下三混区不画统一外框，仅为 8 张实际图片分别保留 ABC 混合功能色描边。
- `show_lora_details` 控制底部 A/B/C 原始名称和最低/最高权重。
- 每个采样任务固定复用同一个 seed、原始 latent 和采样参数，并从输入的基础 MODEL/CLIP 重新应用 LoRA，不会继承上一格权重。
- 节点进度条显示整轮任务进度；当前图片的采样步数会折算到 5/25/69 次总任务中，不会在每张图片开始时重新归零。
- 每组 LoRA 提供文件、触发词、最低权重和最高权重；最低权重默认为 0 且必须低于最高权重。每档实际权重按最低值到最高值均分；触发词按 A/B/C 顺序，在实际权重非零时加到正面提示词开头。
- 逐张解码并提交给合成器，不在内存中同时保留 69 张源图。

## ComfyUI 节点

插件注册两个节点，均位于 `Lora Tester` 分类：

- `LoRA Tester (KSampler)`：主采样与拼图节点。`lora_count` 默认为 1，可选 1–3；分别执行 5、25、69 次采样。
- `LoRA Tester Style`：可选样式节点。只有主节点的 `color_mode` 为 `custom` 时使用，可配置背景颜色或单张背景 `IMAGE`、文字与边框颜色、A/B/C 功能色、间距、字体和装饰器。

主节点会根据 `lora_count` 自动调整 LoRA 配置区：1 只显示 A，2 显示 A/B，3 显示 A/B/C。隐藏只影响界面和节点高度，不会清空 B/C 已保存的文件、触发词或权重；加载已有工作流时也会恢复对应数量。插件通过 ComfyUI 原生 `locales` 机制提供 English 与简体中文节点名、输入名、说明和分类翻译；前端扩展另为颜色模式、背景适配、装饰器和详情开关提供本地化显示文本，但序列化值保持不变。切换 ComfyUI 语言后刷新页面即可生效。

主节点只接受 batch size 为 1 的 latent，并输出一张 `IMAGE`。`max_canvas_megapixels` 默认为 150，可在高级选项中调整；它只限制最终画布，不改变生成图。三 LoRA 输出张量很大，调整上限前应先确认系统内存足够。

## 节点输入与权重规则

每个 LoRA 槽位都有文件、触发词、最低权重和最高权重四项输入：

| 输入 | 默认值 | 说明 |
|---|---:|---|
| `lora_count` | `1` | 使用 A、A+B 或 A+B+C；未使用的槽位只在界面隐藏，不会清除已保存的值。 |
| `*_name` | 首个可用文件 | 从 ComfyUI 的 `models/loras` 列表选择文件。 |
| `*_trigger` | 空字符串 | 实际权重非零时，按 A、B、C 顺序加到正面提示词开头。 |
| `*_min_strength` | `0` | 强度梯度起点，必须严格小于最高权重。 |
| `*_max_strength` | `1` | 强度梯度终点；模型和 CLIP 使用相同权重。 |

四档梯度倍率固定为 `0.25 / 0.5 / 0.75 / 1.0`，每个位置的实际权重计算为：

```text
actual = min_strength + (max_strength - min_strength) * multiplier
```

因此最低权重非零时，中心位置也会应用各 LoRA 的最低权重并注入对应触发词；最低权重为零时，中心位置就是未应用 LoRA 的底模原图。输出侧轴和底栏显示实际权重范围，底栏开关由 `show_lora_details` 控制。

| LoRA 数量 | 唯一采样任务 | 画布占位 | 结构 |
|---:|---:|---:|---|
| 1 | 5 | 5 | 原图加一条四档单 LoRA 横轴。 |
| 2 | 25 | 25 | 两条单 LoRA 轴和一个 4×4 两两混合区域。 |
| 3 | 69 | 73 | A/B、A/C、B/C 三个两两区域、两条 B 独立轴复用图，以及 8 个三 LoRA 混合位置。 |

同一轮中的所有任务复用 seed、latent、采样器、调度器、步数、CFG 和 denoise；节点从基础 MODEL/CLIP 重新应用当前任务的 LoRA，不会把上一格的 LoRA 权重带入下一格。

## 安装、更新与重启

将插件放在 ComfyUI 的 `custom_nodes` 目录下。Windows 开发环境可以使用目录联接：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\link_to_comfy.ps1
```

脚本只在目标路径不存在时创建联接，不会覆盖已有插件目录。修改 Python 节点、`locales` 或 `web` 文件后，需要重启 ComfyUI 后端；随后在浏览器中刷新页面，才能加载新的输入定义和前端脚本。更新 Git 工作区后也应执行同样的重启流程。

若从 GitHub 获取代码：

```powershell
Set-Location D:\ComfyUI
git clone https://github.com/Mon-Landis/LoraTester.git
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\LoraTester\scripts\link_to_comfy.ps1
```

正式运行前，请确认 LoRA 文件位于 ComfyUI 配置的 `models/loras` 目录，并在节点下拉框中刷新到文件名。

本机开发目录已经通过目录联接安装到：

```text
D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes\LoraTester
  -> D:\ComfyUI\LoraTester
```

## 合成器 Python API

```python
from lora_tester import LoraComparisonCompositor, StyleConfig

compositor = LoraComparisonCompositor.from_values(
    ["A.safetensors", "B.safetensors", "C.safetensors"],
    [0.8, 1.0, 2.0],
    image_width=1024,
    image_height=1024,
    lora_min_weights=[0.2, 0.75, 0.5],
    trigger_words=["alpha", "beta", "charlie"],
    show_lora_details=True,
    style=StyleConfig.black(),
)

session = compositor.start()
for task in session.pending_tasks:
    image = sample_and_decode(task.weights, task.prompt_additions)
    session.submit(image, task_id=task.task_id)

result_pil = session.finalize()
```

任务对象还提供 `multipliers`、`active_slots`、实际 `weights`、触发词列表和全部贴图坐标。节点侧应以 `plan.tasks` 作为唯一采样队列，不应按 73 个位置生成图片；重复占位会在 `CompositionSession.submit()` 阶段复用同一任务结果。

如采样结果已经是 ComfyUI IMAGE 批次：

```python
from lora_tester import compose_comfy_batch

result_tensor = compose_comfy_batch(compositor, decoded_image_batch)
```

完整三 LoRA 队列示例位于 [previews/3_lora_manifest.json](previews/3_lora_manifest.json)。

## 自定义样式

```python
style = StyleConfig.custom(
    background_color="#101715",
    background_image="background.png",
    background_fit="cover",  # cover / contain / stretch / tile
    background_opacity=0.7,
    panel_color="#202A27",
    placeholder_color="#E7E9E2",
    text_color="#F2F3EF",
    muted_text_color="#9DA9A4",
    frame_color="#697771",
    accent_colors=("#D8F04A", "#45BFC5", "#F08866"),
    decorator="technical",
)
```

图片尺寸表示每个生成图片区域的尺寸，不包含轴标、三混标签带、区域间隔和底栏。三 LoRA 以 1024 像素方图合成时输出张量已经很大；默认有画布像素保护，必要时应在正式节点中增加联系表缩放选项，而不是无条件提高限制。

## 预览与测试

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\render_previews.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_environment.ps1
```

生成的留白预览位于 [previews](previews)。其中 `3_lora_axes_640x800.png` 使用 `LoraX / 0.9`、`LoraY / 3`、`LoraZZZ / 2` 和单格 `640 × 800`，用于检查完整分辨率下的轴标签、特殊格标题与底栏。

最低权重预览使用 `LoraX: 0.2–0.9`、`LoraY: 0.75–3`、`LoraZZZ: 0.5–2`，可用于确认中心位置、轴刻度、触发词对应的实际权重和底栏 `MIN / MAX` 描述。
