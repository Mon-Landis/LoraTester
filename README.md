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
- 可用 `LoRA Stack` 保存多组 LoRA 文件、触发词和强度，并用 `LoRA Stack Splitter` 生成全部非空组合。
- 可用动态增长的 `LoRA Stack Lister` 合并多个独立 Stack；连接一个输入后自动显示下一个输入槽。
- `Style Combination Tester` 以 Prompt 为 Y 轴、LoRA/画师组合为 X 轴生成比较矩阵；左侧固定显示 `Prompt 1 / Prompt 2 / ...`，顶部显示 `BASE / A / B / A+B / ...`，底栏分别列出不同的 LoRA 文件、触发词、画师 Tag 与强度配置。同一文件使用不同强度或触发词时会分配不同字母，不会合并。BASE 与第一列效果图默认至少间隔单图宽度的 `1/8`，也可用 `control_gap` 显式覆盖。
- LoRA 文件下拉框提供稳定的“画师 Tag 模式”占位项；选择后，该项不加载 LoRA，触发词字段改为画师 Tag，强度字段改为画师权重。旧工作流仍保存原有字段和值。
- Node 2.0 与旧 LiteGraph 共用一套动态布局：打开工作流、分页切回、缩小后再增大数量时都会重新同步可见项；其他设备缺少已保存 LoRA 时，下拉框仍可打开并保留缺失值供替换。
- 隐藏槽位中的跨设备缺失 LoRA 不会阻断低数量工作流：后端只校验当前 `lora_count` 内的槽位，隐藏值仍保存在工作流中供以后替换；当前数量内的缺失 LoRA 仍会明确报错。

## ComfyUI 节点

插件注册八个节点，位于 `Lora Tester`、`Stacks` 与 `Artist Tags` 子分类：

- `Style Component Tester`：主采样与拼图节点。`lora_count` 默认为 1，可选 1–3；分别执行 5、25、69 次采样。
- `LoRA Tester Style`：可选样式节点。只有主节点的 `color_mode` 为 `custom` 时使用，可配置背景颜色或单张背景 `IMAGE`、文字与边框颜色、A/B/C 功能色、间距、字体和装饰器。
- `LoRA Stack`：最多配置 16 组 LoRA 文件、触发词和强度，输出 `LORA_STACK`；所有启用项都必须选择文件。
- `LoRA Stack Splitter`：将一个 Stack 拆成所有非空组合并输出 `LORA_STACK_LIST`。例如 ABC 会得到 A、B、C、AB、AC、BC、ABC。
- `LoRA Stack Lister`：按连接顺序合并多个 `LORA_STACK`，动态显示最多 16 个输入槽。
- `Style Combination Tester`：最多使用 16 行正面提示词，与 `LORA_STACK_LIST` 组成 XY 比较矩阵，并保留无 LoRA/画师组合的 BASE 对照列；“独立画师 Tag”会应用到 BASE 和所有 Stack 测试格。
- `Artist Tag Template`：输出 `ARTIST_TAG_TEMPLATE`，用两条表达式分别定义权重为 1 和非 1 时的画师 Tag 写法。
- `Anima Artist Mixer Configuration`：输出 `ANIMA_ARTIST_MIXER_CONFIG`；默认参数按参考工作流设为 `1.6 / true / base_anchored / true / false / 0.0`。

`Style Component Tester` 会根据 `lora_count` 自动调整 LoRA/画师配置区：1 只显示 A，2 显示 A/B，3 显示 A/B/C。`LoRA Stack` 和 `Style Combination Tester` 也会按数量隐藏未启用的配置组；`LoRA Stack Lister` 会在连接后显示下一个输入槽。隐藏只影响界面和节点高度，不会清空已保存的值。插件通过 ComfyUI 原生 `locales` 机制为原有采样和样式节点提供 English 与简体中文翻译；前端扩展另为枚举值、详情开关和动态输入提供显示逻辑，但序列化值保持不变。

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

### 低权重组合的质量边界

混合测试中的低权重格不是“接近底模的保证质量”格。只要某个 LoRA 的实际权重非零，节点就会同时对 MODEL 和 CLIP 应用该权重，并把它的触发词加入正向提示词；画师模式项也会按当前画师权重渲染。于是当 LoRA 和画师权重都处在轴的低端时，生成结果可能同时带有微弱的模型补丁、仍然被编码的触发词条件和微弱的画风条件，落在底模与目标风格之间，甚至明显低于 BASE 基线。这是底模、LoRA 训练分布和提示词编码共同造成的正常模型行为，不表示节点串用了上一格权重、误把 LoRA 送入画师 Mixer 或拼图逻辑出错。

如果需要把 BASE 作为稳定质量参照，应使用 `min_strength = 0` 的 BASE 格；低权重格应作为“过渡区/失真风险区”观察，而不是质量下限。只有实际权重为零时，对应 LoRA 才不会应用到当前测试格，且其触发词也不会加入提示词；节点仍可能在本轮开始时预读已配置槽位的 CPU state dict，这不改变当前格的 MODEL/CLIP 结果。

| LoRA 数量 | 唯一采样任务 | 画布占位 | 结构 |
|---:|---:|---:|---|
| 1 | 5 | 5 | 原图加一条四档单 LoRA 横轴。 |
| 2 | 25 | 25 | 两条单 LoRA 轴和一个 4×4 两两混合区域。 |
| 3 | 69 | 73 | A/B、A/C、B/C 三个两两区域、两条 B 独立轴复用图，以及 8 个三 LoRA 混合位置。 |

同一轮中的所有任务复用 seed、latent、采样器、调度器、步数、CFG 和 denoise；节点从基础 MODEL/CLIP 重新应用当前任务的 LoRA，不会把上一格的 LoRA 权重带入下一格。

## 画师 Tag 与模型模板

内置模型判断读取 ComfyUI 已解析的模型配置，不根据 checkpoint 文件名猜测：

| 模型状态 | 默认模板 | 输入宽容与最终示例 |
|---|---|---|
| `model.model.model_config.unet_config["image_model"] == "anima"` | `@{tag}` / `(@{tag}:{weight})` | `fkey`、`@fkey`、`@@fkey` 都会得到 `@fkey`；`(@fkey:3)` 配合新权重 `1.2` 会得到 `(@fkey:1.2)`。保留用户输入的空格或下划线。 |
| 其他模型（Pony、Animagine、Illustrious、NoobAI、CKN 等 Danbooru Tag 系） | `{tag}` / `({tag}:{weight})` | 去除多余 `@` 与旧权重，转为小写并把连续空白换成下划线；`Artist Name`、权重 `1.2` 得到 `(artist_name:1.2)`。 |
| 手动接入 `Artist Tag Template` | 用户的两条模板 | 只统一去除多余 `@` 和旧括号权重，不擅自改变大小写或空格。模板仅允许 `{tag}`、`{weight}` 字段。 |

[Anima Base/Aesthetic/Turbo 官方模型卡](https://huggingface.co/circlestone-labs/Anima)明确要求画师 Tag 使用 `@` 前缀；[Anima 2.9B 模型卡](https://huggingface.co/Gazingstars123/Anima-2.9B)明确沿用这一提示习惯。因此 Anima 内置模板不会产生 `@@artist`，也不会把用户习惯的空格强制改成下划线。画师模式的一项可输入逗号、中文逗号或换行分隔的多个画师，每个画师使用该项当前权重。

## Anima Artist Mixer 路由

当某一张对比图实际进入 `anima_artist_mixer` 路由时，输出对比图会在该格图片下方显示 `Anima Artist Mixer`；原生提示词、Mixer 缺失回退、关闭高级开关、非 Anima 或单画师格不会显示该标注。为避免覆盖下一行，只有存在潜在多画师路由时才预留标注带，实际未走 Mixer 的单元格保持空白。

本项目可选兼容 [Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer)，只使用它的 `AnimaArtistPack` 与 `AnimaArtistAdapterMixer` 注册节点。每个测试格独立判断，行为如下：

| 当前测试格 | 执行结果 |
|---|---|
| 非 Anima | 使用内置/手动模板原生拼入提示词；即使连接 Mixer 配置也忽略。 |
| Anima，合计 0 或 1 个画师 | 原生提示词编码，不查询也不调用外部 Mixer。 |
| Anima，高级设置中的 `Use Anima Artist Mixer` 已关闭 | 强制原生提示词编码；即使存在多个画师且外部项目已加载也不查询、不调用 Mixer。 |
| Anima，合计至少 2 个画师，外部项目未加载 | 保留原提示词并原生编码；节点界面显示兼容性警告。 |
| Anima，合计至少 2 个画师，外部项目已加载 | 仅将画师模式项和当前采样节点的“独立画师 Tag”字段加入 `artist_chain`；正向提示词、通用前缀和 LoRA 触发词中的 `@tag` 原样保留在 `base_prompt`。先调用 `AnimaArtistPack.pack()`，再把返回值及配置传给 `AnimaArtistAdapterMixer.patch()`，采样使用它返回的 MODEL 与正向 CONDITIONING。 |

直接采样器的双条目布局共有 25 张测试图。使用 Anima、Mixer 已加载且高级开关开启时，实际节点执行测试得到：

| 独立画师 Tag 数 | 测试条目 | Mixer 调用数 |
|---:|---|---:|
| 0 | 1 个画师模式 + 1 个 LoRA | 0 / 25 |
| 0 | 2 个 LoRA | 0 / 25 |
| 1 | 1 个画师模式 + 1 个 LoRA | 20 / 25，仅画师模式权重非零的图片调用 |
| 1 | 2 个 LoRA | 0 / 25 |
| 2 | 1 个画师模式 + 1 个 LoRA | 25 / 25 |
| 2 | 2 个 LoRA | 25 / 25 |

“合计画师数”包含当前测试格的显式画师模式项，以及当前采样节点独立字段中的画师项。普通提示词中的 `@tag` 不再参与计数或自动抽取。例如直接采样器的测试项 `@test_artist`（权重 `0.25`）加独立字段 `@independent_artist` 和正向提示词 `@prompt_artist, portrait` 时，外部输入为：

```text
artist_chain:
(@test_artist:0.25)
@independent_artist

base_prompt:
@prompt_artist, portrait
```

外部项目缺失时不会删除原提示词中的画师项，因此回退不会吞掉用户内容。外部项目存在但模型不是 Anima 时也不会调用。高级 Mixer 开关默认开启；关闭后同时抑制缺失 Mixer 的前端警告并强制后端使用原生编码。配置节点的 `enabled=false` 或 `strength=0` 也会强制原生编码。配置节点没有接入外部项目的 `advanced_options`，因此不会开启 Anchor-Q 或跨运行 warm-cache。组合测试器会在开始前输出模型族、画师混合状态、Mixer 可用/开关/启用/生效状态和需要 Mixer 的组合，并在每张图采样前输出当前 Stack、LoRA、画师项和路由模式。

组合测试器同样逐格路由：独立字段为 1 个画师时，BASE 原生编码；某个 Stack 再含 1 个画师时，只有该 Stack 列进入 Mixer。独立字段自身已有至少 2 个画师时，BASE 和全部 Stack 列都会进入 Mixer。字段为空时完全保持旧工作流行为；关闭高级 Mixer 开关时所有列都强制原生编码。

### 测试详情日志

两个采样节点的高级设置中都有 `Log Test Details`（默认开启）。开启后，每个测试格会分行输出：LoRA 文件名与实际权重、画师 Tag 与权重、最终渲染 Tag、原生/Mixer 路由，以及缓存状态。`cache=run-local:miss` 表示本次执行首次从文件加载，`cache=run-local:hit` 表示命中本次执行的 CPU state-dict LRU；`Patched model: column reuse` 表示组合测试器在同一 Stack 列复用了已经应用 LoRA 的 MODEL/CLIP。画师项的 `cache=external:lazy-per-sample` 表示外部 Mixer 会在当前采样步首次需要时建立并复用 embedding，`cache=none` 表示原生提示词编码，不代表跨测试格缓存。关闭开关后不会输出本项目的逐格详情和组合预检日志，但外部节点自身的日志仍由外部项目控制。

两个采样器的独立字段都支持逗号、中文逗号或换行分隔，也支持 `fkey`、`@fkey`、`(@fkey:0.5)` 等形式。组合测试器中的该字段应用到 BASE 和所有 Stack 测试格。字段为空时保持旧工作流的原生提示词行为。

## 缓存与执行顺序

- ComfyUI 的 `load_torch_file()` 默认把 LoRA state dict 加载到 CPU；MODEL/CLIP patch 的加载、卸载与显存调度继续由 ComfyUI ModelPatcher 管理。本项目不调用 `empty_cache()`，避免打断 ComfyUI 的动态显存策略。
- 每个直接测试格在成功、采样异常或用户中断时，都会在 `finally` 中通过 ComfyUI 的 `unload_model_and_clones()` 释放临时 MODEL/CLIP clone；组合测试器还会在每个提示词行释放临时 Artist Mixer route，并在每个非 BASE 列结束时释放列级 LoRA clone。基础输入对象不会被直接作为卸载目标，clone 组内的设备调度仍由 ComfyUI 管理。
- AIMDO/DynamicVRAM 会在多个测试格之间共享动态 pinned host buffer。旧版执行路径可能让 host buffer 在后半轮测试中逐渐逼近预留上限，于是日志出现 `hostbuf_grow requested ... beyond reserved host buffer`；只要采样仍继续完成，这通常是主机缓冲区的非致命压力告警，不代表最终图片被改写。显式按格/列释放 clone 可以避免本节点把临时 patcher 长期留在当前执行中；本项目仍不主动调用全局 `gc.collect()`、`empty_cache()` 或修改 ComfyUI 的 RAM/VRAM 上限。
- LoRA state-dict LRU 只缓存本轮 CPU 文件读取，执行结束、异常和中断都会清空；它不持有 MODEL、CLIP、conditioning、latent、图片或外部 Mixer 对象。若在升级后仍持续出现 host buffer 告警，应先确认其它节点没有保留 DynamicVRAM clone，再单独检查 ComfyUI/AIMDO 版本和启动参数。
- 主采样器只在单次执行内保留最近 3 个 LoRA 的 CPU LRU，正好覆盖单节点最多三项；正常结束、异常和用户中断后都会清空。缓存键包含规范化路径，并比较文件大小、修改时间和创建/变更时间；同路径文件被覆盖后会自动重新加载。
- `Style Combination Tester` 使用运行内最多 3 项的 CPU LRU，并按 Stack 列执行：同一组合只应用一次 LoRA MODEL/CLIP patch、只编码一次负面提示词，再依次采样该列的所有提示词行。正常结束、异常和用户中断都会清空该 LRU。
- 外部 Mixer 的 GPU 侧画师 embedding、混合 context 与 Anchor 状态由其 ModelPatcher cleanup/中断路径清理。实际源码中 `build_artist_embedding_sum()` 会按提示词、权重、模型 patch 和输入 tensor 签名缓存加权后的画师 embedding sum，`_mixed_context_cache` 则缓存当前采样运行的混合 context；这些缓存会在运行、模型 patch 或输入改变时失效。本项目不复制或延长这些缓存的生命周期，也不再额外保留跨测试格的画师向量。
- ComfyUI 自身仍可缓存整个未变化节点的输出；直接采样器与 `LoRA Stack` 的 `IS_CHANGED` 还会把当前有效 LoRA 文件 stat 指纹加入输出缓存签名，因此原路径文件被覆盖也会重新执行。本项目不额外缓存最终大图、latent、VAE 解码图、正向 conditioning 或 patched MODEL，避免 CPU/显存泄漏和陈旧内容复用。
- 单画师 post-Adapter 效果的“单位向量差分乘权重”验证记录在 [`audit/anima_artist_linearity.md`](audit/anima_artist_linearity.md)。该等式对现有 `(@artist:weight)` 语义不成立，因此本版本不实现跨测试格的画师效果缓存。

## 可读审计报告

运行 `python scripts/generate_artist_audit.py` 会使用生产路由函数生成 [`audit/artist_routing_report.md`](audit/artist_routing_report.md) 和对应的 JSON 原始数据。报告覆盖独立字段解析、Anima/Danbooru 模板、Mixer 存在/缺失、普通提示词和 LoRA 触发词不抽取、组合测试预检等分支；它不伪造图片结果，适合审核节点实际会送入的 `artist_chain`、`base_prompt` 和最终路由。

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

生成的留白预览位于 [previews](previews)。其中 `3_lora_axes_640x800.png` 使用 `LoraX / 0.9`、`LoraY / 3`、`LoraZZZ / 2` 和单格 `640 × 800`，用于检查完整分辨率下的轴标签、特殊格标题与底栏。`multi_prompt_stack_matrix.png` 使用三个原始 LoRA 的七种非空组合，检查 Prompt 竖轴、BASE 对照间隔、混合状态标题和原始 LoRA 底栏。

最低权重预览使用 `LoraX: 0.2–0.9`、`LoraY: 0.75–3`、`LoraZZZ: 0.5–2`，可用于确认中心位置、轴刻度、触发词对应的实际权重和底栏 `MIN / MAX` 描述。
