# ComfyUI LoRA Tester

[简体中文](README.md) | [English](README_EN.md)

面向 ComfyUI 的 LoRA、画师 Tag 与通用 XY 参数测试节点。采样器直接接收 `MODEL / CLIP / VAE / LATENT`，在节点内完成提示词编码、模型/CLIP 处理、采样、VAE 解码和带标注的对比图合成。

![多提示词与风格组合对比矩阵](previews/multi_prompt_stack_matrix.png)

## 核心能力

- 通用 `XY Test Sampler`：组合任意两个 `XY_AXIS`，输出带标注的 `comparison_sheet` 和按行优先排列的原始图片批次 `raw_images`。
- 提示词轴：拆分长文本、统一前置或后置追加提示词，并单独传递画师 Tag。
- 种子轴：解析显式种子列表，或根据来源种子确定性生成一组随机种子。
- 风格轴：从风格组合、组合拆分和列表汇总节点构造轴，统一表示 LoRA 与画师 Tag，顶部显示“权重-代号”组合，底部只列出代号对应的来源信息。
- 通用轴合成：提示词列表、风格组合、风格组合列表和种子列表均可经 `Axis Composer` 转换为方向无关的 `XY_AXIS`。
- 专用 LoRA 测试：保留 1 至 3 个 LoRA 的权重梯度与混合布局。
- Anima 兼容：按模型配置选择画师 Tag 模板，并可选接入 Anima Artist Mixer 处理多画师组合。
- 可定制输出：黑色、白色或自定义主题，支持背景图、字体、颜色、间距、装饰器、分类表格和文字说明。
- 双语节点界面：通过 ComfyUI 原生 `locales` 提供简体中文与 English。

## 推荐工作流

最常用的“提示词 Y 轴 + 种子 X 轴”连接方式：

```mermaid
flowchart LR
  P1["Multi Prompt Input"] --> P2["Global Prompt Append"]
  P2 --> P3["Axis Composer"]
  S1["Seed List / Random Seeds"] --> S2["Axis Composer"]
  P3 -->|axis → y_axis| XY["XY Test Sampler"]
  S2 -->|axis → x_axis| XY
  M["MODEL / CLIP / VAE / LATENT"] --> XY
  XY --> O1["comparison_sheet"]
  XY --> O2["raw_images"]
```

画风横向测试可使用：

```text
Style Stack -> Style Stack Splitter / Style Stack Lister -> Axis Composer -> axis -> x_axis
Multi Prompt Input -> Global Prompt Append                    -> Axis Composer -> axis -> y_axis
```

所有轴节点的输出均名为 `axis`，轴本身不绑定方向，可自由接入采样器的 `x_axis` 或 `y_axis`。`axis_title` 是整条轴的总标题；每行或每列顶端显示的文字来自各个 `AxisEntry.label`。`Axis Composer` 的 `include_base` 可将风格 BASE 放入单独分组，使基线列与其余测试列之间自动留出间隔。`Prompt Axis`、`Style Axis` 和 `Seed Axis` 仍作为对应数据类型的快捷构造器提供。

### 种子列表的使用

在 `Seed List / Random Seeds` 节点中选择“指定列表”，然后在“种子列表”里输入十进制非负整数；可用英文逗号、空格或换行分隔，例如 `1, 42, 123456`。输入顺序就是轴上的顺序，每个值对应一行或一列。将节点输出接入 `Seed Axis` 的 `seed_list`，或直接接入 `Axis Composer` 的 `source`，再把生成的 `axis` 接到 `XY Test Sampler` 的 `x_axis` 或 `y_axis`。

选择“确定性随机生成”时，“随机种子数量”控制轴长度；相同的“生成器种子”会得到相同的种子序列，便于复现。将“生成器种子”设为 `-1` 会在每次执行时随机选择新的生成器种子，因此生成的序列不再可复现。

## 安装

在 ComfyUI 的 `custom_nodes` 目录克隆仓库：

```powershell
Set-Location D:\ComfyUI\ComfyUI_windows_portable\ComfyUI\custom_nodes
git clone https://github.com/Mon-Landis/LoraTester.git
..\..\python_embeded\python.exe -s -m pip install -r .\LoraTester\requirements.txt
```

其他安装方式使用运行 ComfyUI 的 Python 执行：

```bash
python -m pip install -r ComfyUI/custom_nodes/LoraTester/requirements.txt
```

安装或更新 Python、`web`、`locales` 文件后，重启 ComfyUI 后端并刷新浏览器。LoRA 文件需要位于 ComfyUI 已配置的 `models/loras` 目录。

仓库外开发可使用目录联接；脚本只在目标不存在时创建联接，不会覆盖已有目录：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\link_to_comfy.ps1
```

## 节点概览

| 分类 | 节点 | 用途 |
|---|---|---|
| `Lora Tester/XY` | `XY Test Sampler` | 对两个轴做笛卡尔积采样，输出拼接图与原始图片批次。 |
| `Lora Tester/XY/Prompt` | `Multi Prompt Input` | 通过数量控制和独立输入框构造多提示词列表。 |
| `Lora Tester/XY/Prompt` | `Global Prompt Append` | 为全部提示词统一前置/后置文本，并追加独立画师 Tag。 |
| `Lora Tester/XY/Prompt` | `Prompt Axis` | 将提示词列表直接转换为方向无关的 `XY_AXIS`。 |
| `Lora Tester/XY/Style` | `Style Stack` | 配置最多 16 个 LoRA 或画师 Tag 风格项。 |
| `Lora Tester/XY/Style` | `Artist Tag Text Parser` / `画师 Tag-文本解析` | 将画师提示词文本解析为含权重的风格组合。 |
| `Lora Tester/XY/Style` | `Artist Tag Replacer` / `画师替换` | 精确匹配画师并替换为画师或 LoRA，支持替换权重与倍率。 |
| `Lora Tester/XY/Style` | `Style Stack Splitter` | 生成风格组合的全部非空组合。 |
| `Lora Tester/XY/Style` | `Style Stack Flattener` / `风格组合平铺` | 将组合平铺为独立单项，可选择在首位加入原始组合。 |
| `Lora Tester/XY/Style` | `Style Stack Lister` | 动态合并最多 16 个独立风格组合。 |
| `Lora Tester/XY/Style` | `Style Axis` | 将风格组合列表直接转换为带分组和详情表的 `XY_AXIS`。 |
| `Lora Tester/XY/Seed` | `Seed List / Random Seeds` | 解析种子列表或确定性生成随机种子。 |
| `Lora Tester/XY/Seed` | `Seed Axis` | 将种子列表直接转换为方向无关的 `XY_AXIS`。 |
| `Lora Tester/XY/Axis` | `Axis Composer` | 将任一受支持的原始数据源或完整轴转换为通用 `axis`。 |
| `Lora Tester/XY/Axis` | `Axis Content Preview` / `轴内容预览` | 将任意完整轴展开为保留分组与缩进的可读文本，同时原样传递轴。 |
| `Lora Tester` | `Style Component Tester` | 专用的 1 至 3 LoRA 权重与混合测试器。 |
| `Lora Tester` | `LoRA Tester Style` | 为采样器提供自定义视觉样式。 |
| `Lora Tester/Artist Tags` | `Artist Tag Template` | 覆盖普通权重和加权画师 Tag 的格式。 |
| `Lora Tester/Artist Tags` | `Anima Artist Mixer Configuration` | 配置可选的多画师 Anima 路由。 |
| `Lora Tester/Deprecated` | `Style Combination Tester` | 兼容旧工作流的提示词水平测试器；新工作流请使用通用 XY 节点。 |

旧节点 `Style Combination Tester` 已标记为废弃，但注册键和输入仍保留，已有工作流可以继续加载。其内部复用通用 XY 采样核心；不要在新工作流中继续依赖该节点。

## XY 行为与限制

- 每个轴最多 64 个条目。两轴不能同时修改同一个参数，例如同时提供两个种子轴或两个提示词轴；采样器会在加载模型前拒绝冲突。
- 已内置 `prompt`、`seed`、`lora_stack`、`steps`、`cfg`、`sampler_name`、`scheduler` 和 `denoise` 参数处理器。当前界面提供提示词、种子和风格数据源及快捷轴构造器；`Axis Composer` 提供一致的通用入口。
- 轴合成后只允许整轴级操作，不提供单元素二次修改。内部已提供保留分组的轴拼接与带参数冲突检查的交叉合并函数，供后续节点使用。
- 当轴来源可被前端识别时，对应的基础采样控件会被禁用；后端校验始终是最终边界。
- `raw_images` 为 ComfyUI `[N,H,W,C]` IMAGE 批次，顺序是行优先 `(y, x)`，不受分组间距和底部详情布局影响。
- 输入 latent 必须只有一个样本。轴条目过多或 latent 空间过大时会产生非阻断警告；最终拼接图默认受 `150 MP` 的 `max_canvas_megapixels` 限制。
- 原始图片批次会占用 CPU 内存。矩阵规模和单图分辨率较大时，应先减小轴长度或 latent 尺寸，再考虑提高画布上限。
- `extra_footer_text` 非空时会在所有轴详情后增加 `NOTES`；风格轴底部只显示代号来源表，提示词正文不会重复显示。

### AnimaFlow XY 测试器

`AnimaFlow XY 测试器`（`Lora Tester/XY`）是独立的 f 采测试入口，依赖 [Comfyui-anima-sampler](https://github.com/KeithZ117/Comfyui-anima-sampler) 的 `AnimaFlowCorrectiveSampler`。现有提示词、风格/画师、种子、组合轴和平铺后的风格轴继续使用 `XY_AXIS`；输出仍为对比图和行优先的原始图片批次。

- 求解器、调度器、CFG 模式的选项和数值范围、默认值来自当前外部节点定义，不复制求解器或调度器实现。
- `flow_settings` 可直接连接外部 `Anima Flow Settings`。每个单元格调用外部节点的公开 `sample()`；LoRA/画师路由和图表合成复用 XY 核心，VAE 每格只解码一次。
- 新增 `AnimaFlow 参数轴`（`Lora Tester/XY/Axis`）：选择参数后输入每行一个值或逗号分隔的值。例如 `flow_solver` 填写 `flow_euler, flow_heun`，`flow_shift` 填写 `3, 5`。高级参数列表来自外部 Settings 节点；布尔参数使用 `true/false/1/0`。
- 参数轴覆盖基础配置；高级参数轴只覆盖对应设置，保留其余已连接的设置。可进行交叉合并，也可先接入轴内容预览。
- 原生 `sampler_name`、`scheduler` 不能自动翻译为 Flow 参数；误接时明确报错。普通 XY 测试器也不会接受 Flow 专用参数。
- 缺少依赖或接口不兼容时保留兼容壳，工作流仍可识别节点；显示红色依赖提示和项目链接，执行被阻止，绝不回退到 KSampler。安装/更新依赖后重启 ComfyUI 并刷新浏览器。
- f 采保持自己的步进预览和进度语义，XY 层报告完成格数，不干预其内部模型调用次数。

推荐连线：`画师文本解析 -> 风格组合平铺 -> Style Axis -> AnimaFlow XY 测试器.x_axis`，提示词轴接 `y_axis`；比较 Flow 参数时，将 `AnimaFlow 参数轴` 接任一轴。

### 轴内容预览

将任意轴节点的 `axis` 输出接入 `轴内容预览`（`Axis Content Preview`）。支持提示词轴、风格轴、种子轴、组合轴和采用 `XY_AXIS` 类型的自定义轴。执行后，节点内显示可选择、复制和滚动的只读多行文本，不会将结构压平；`预览文本语言` 可选择中文或英文。

- 总览包含轴标题、分组数、总项数和参数名称。
- 按原顺序展开各组、各项、标签、详情标签和全部参数；BASE 显示为无参数覆盖。
- 提示词展开正文、前置 / 后置文本、合并提示词和独立画师 Tag；LoRA / 画师项展开文件、触发词 / Tag、权重与画师模板。
- 种子保留完整整数；其他参数与嵌套数据按结构展开，详情表和文本单独列出。
- 输出 `axis` 为原轴，输出 `text` 为完整格式化字符串，可串接采样器或文本保存节点。预览结果保存在工作流属性中，保存 / 重载后可继续检查；修改输入后重新执行即可更新。

例如：`Style Axis -> 轴内容预览 -> XY Test Sampler`。本节点只检查轴数据，不加载模型或生成图片；单独连接该输出节点可只运行上游轴构造链路。

### 画师文本解析与替换

`画师 Tag-文本解析` 接收专门的画师文本，例如 `@wlop, (@ask_(askzy):0.55)`，输出一个 `LORA_STACK`：每个画师独立存为画师模式项，未指定权重时使用 `1.0`。支持逗号、中文逗号和换行分隔，保留名称中的括号、顺序、重复项以及可选的画师模板。该字段只用于画师文本，不会从普通提示词中自动提取画师。

`画师替换` 接收一个 `LORA_STACK`；配置一个完整的匹配画师名（可带或不带 `@`），选择替换 LoRA 文件或画师 Tag 模式，填写替换触发词 / 画师 Tag，再设置强度与模式：

- `替换`：匹配项的新权重直接使用配置的强度。
- `倍率`：匹配项的新权重为原权重乘以配置的倍率。例如原权重 `0.3`、倍率 `2`，输出权重 `0.6`。
- 精确匹配、区分大小写，并替换全部匹配项；不会误匹配普通 LoRA 触发词或画师名的子串。无匹配时原样返回。
- 替换物、触发词和强度控件复用 `Style Stack` 构造器的同一代码路径；替换成画师时填写画师 Tag，替换成 LoRA 时填写该文件的触发词。
- 保留未匹配项、顺序与画师模板；一项包含多个画师时只替换匹配画师，其余画师保留原权重。

可连接为 `画师 Tag-文本解析 -> 画师替换 -> 风格组合平铺 -> Style Axis / Axis Composer`。解析输出直接接入轴时代表完整画师组合；通过平铺节点可分别测试每个画师。

### 风格组合平铺

将 `Style Stack` 接入 `风格组合平铺`（`Style Stack Flattener`），其 `lora_stack_list` 输出可直接接入 `Style Axis` 或 `Axis Composer`。

- 默认关闭 `加入原始组合`：`a:1.2,b:0.3,c:1` 输出 `[a:1.2]`、`[b:0.3]`、`[c:1]` 三项。
- 开启开关：输出 `[a:1.2,b:0.3,c:1]`、`[a:1.2]`、`[b:0.3]`、`[c:1]` 四项，原始组合在首位。
- `权重模式` 默认选择 `继承`，子项保留原权重；`归一化` 将每个子项权重设为 `1`；`双行` 对原权重非 `1` 的项相邻输出权重 `1`、原权重两个子项，原权重为 `1` 时只输出一次。
- 例如 `a:1.2,b:0.3,c:1` 在双行模式下输出 `[a:1]`、`[a:1.2]`、`[b:1]`、`[b:0.3]`、`[c:1]`。加入原始组合时，仍在首位输出未修改的 `[a:1.2,b:0.3,c:1]`，共六项。
- 权重模式不影响加入的原始组合；每项保留原始顺序、触发词和画师 Tag 模板。不生成两两组合，也不自动去重。

## 提示词与画师 Tag

`Global Prompt Append` 只把“独立画师 Tag”字段送入画师处理链。普通提示词和 LoRA 触发词中的 `@tag` 会保留在原提示词中，不会被自动抽取。

内置模型判断读取 ComfyUI 已解析的模型配置，而不是 checkpoint 文件名：

| 模型 | 默认格式 |
|---|---|
| Anima | `@{tag}` / `(@{tag}:{weight})` |
| 其他 Danbooru Tag 系模型 | `{tag}` / `({tag}:{weight})` |
| 连接 `Artist Tag Template` | 使用节点中定义的两条模板 |

项目可选兼容 [Anima-Artist-Mixer](https://github.com/An1X3R/Anima-Artist-Mixer)。仅当模型为 Anima、当前测试格至少包含两个显式画师项、Mixer 可用且开关/配置启用时，才调用外部 Mixer；其余情况使用原生提示词编码。外部项目缺失不会阻止本插件导入或执行。

更深入的边界与验证记录：

- [Anima 画师权重线性验证](audit/anima_artist_linearity.md)
- [画师路由审计报告](audit/artist_routing_report.md)

## 专用 LoRA 测试器

`Style Component Tester` 适合快速检查 1 至 3 个 LoRA 的单项权重和混合效果：

| LoRA 数量 | 唯一采样任务 | 画布位置 |
|---:|---:|---:|
| 1 | 5 | 5 |
| 2 | 25 | 25 |
| 3 | 69 | 73，部分单轴图片复用 |

每项使用 `min_strength` 与 `max_strength`，四档倍率为 `0.25 / 0.5 / 0.75 / 1.0`：

```text
actual = min_strength + (max_strength - min_strength) * multiplier
```

实际权重非零时，LoRA 会同时应用到 MODEL 与 CLIP，并将触发词加入正面提示词。低权重格不等于质量下限；稳定的底模参照应使用实际权重为零的 BASE。

![三 LoRA 专用测试布局](previews/3_lora_axes_640x800.png)

## 样式

`color_mode` 支持 `black / white / custom`。使用 `custom` 时连接 `LoRA Tester Style`，可配置背景颜色或单张背景 `IMAGE`、背景适配、文字/边框/A-B-C 功能色、间距、字体和装饰器。样式只影响拼接图，不改变 `raw_images`。

Python 侧也可通过 `register_style_decorator()` 注册实现 `draw_background()` 与 `draw_foreground()` 的装饰器。详细模块边界和 API 见 [开发与架构文档](DEVELOPMENT.md)。

## 开发与验证

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_tests.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\render_previews.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\check_environment.ps1
```

- [开发与架构文档](DEVELOPMENT.md)
- [预览与测试清单](previews/)
- [三 LoRA 任务清单示例](previews/3_lora_manifest.json)
