import { app } from "../../scripts/app.js";

const TARGET_NODE = "LoraTesterSampler";
const STACK_NODE = "LoraStack";
const STACK_SPLITTER_NODE = "LoraStackSplitter";
const STACK_LISTER_NODE = "LoraStackLister";
const MULTI_PROMPT_NODE = "MultiPromptSample";
const MAX_STACK_INPUTS = 16;
const MULTI_PROMPT_MIN_WIDTH = 480;
const HIDDEN_WIDGET_TYPE = "hidden";
const WIDGET_STATE = Symbol("loraTesterWidgetState");
const DOM_STATE = Symbol("loraTesterDomState");
const LOCALIZED_OPTION = Symbol("loraTesterLocalizedOption");
const STACK_INPUT_TEMPLATES = Symbol("loraTesterStackInputTemplates");
const GRAPH_SYNC_INSTALLED = Symbol("loraTesterGraphSyncInstalled");

const OPTION_LABELS = {
  LoraTesterSampler: {
    color_mode: {
      black: { en: "Black background / white text", zh: "黑底白字" },
      white: { en: "White background / black text", zh: "白底黑字" },
      custom: { en: "Custom style", zh: "自定义样式" },
    },
  },
  MultiPromptSample: {
    color_mode: {
      black: { en: "Black background / white text", zh: "黑底白字" },
      white: { en: "White background / black text", zh: "白底黑字" },
      custom: { en: "Custom style", zh: "自定义样式" },
    },
  },
  LoraTesterStyle: {
    background_fit: {
      cover: { en: "Cover", zh: "覆盖裁切" },
      contain: { en: "Contain", zh: "完整包含" },
      stretch: { en: "Stretch", zh: "拉伸填满" },
      tile: { en: "Tile", zh: "平铺" },
    },
    decorator: {
      none: { en: "None", zh: "无" },
      technical: { en: "Technical", zh: "技术信息风" },
    },
  },
};

const TOGGLE_LABELS = {
  LoraTesterSampler: {
    show_lora_details: {
      en: {
        label_on: "show names and min/max weights",
        label_off: "hide names and min/max weights",
      },
      zh: {
        label_on: "显示名称和最低/最高权重",
        label_off: "隐藏名称和最低/最高权重",
      },
    },
  },
  MultiPromptSample: {
    show_lora_details: {
      en: {
        label_on: "show LoRA names and strengths",
        label_off: "hide LoRA names and strengths",
      },
      zh: {
        label_on: "显示 LoRA 名称和强度",
        label_off: "隐藏 LoRA 名称和强度",
      },
    },
  },
};

const INPUT_LABELS = {
  LoraStack: {
    lora_count: { en: "LoRA Count", zh: "LoRA 数量" },
  },
  LoraStackSplitter: {
    lora_stack: { en: "LoRA Stack", zh: "LoRA 组合" },
  },
  MultiPromptSample: {
    model: { en: "Base Model", zh: "基础模型" },
    clip: { en: "CLIP", zh: "CLIP" },
    vae: { en: "VAE", zh: "VAE" },
    latent_image: { en: "Latent", zh: "潜空间图像" },
    lorastacks: { en: "LoRA Stack List", zh: "LoRA 组合列表" },
    prompt_count: { en: "Prompt Count", zh: "提示词数量" },
    prompt_prefix: { en: "Shared Prompt Prefix", zh: "通用正面提示词前缀" },
    negative_prompt: { en: "Shared Negative Prompt", zh: "通用负面提示词" },
    seed: { en: "Seed", zh: "随机种子" },
    steps: { en: "Steps", zh: "采样步数" },
    cfg: { en: "CFG", zh: "CFG" },
    sampler_name: { en: "Sampler", zh: "采样器" },
    scheduler: { en: "Scheduler", zh: "调度器" },
    denoise: { en: "Denoise", zh: "降噪强度" },
    color_mode: { en: "Color Mode", zh: "颜色模式" },
    show_lora_details: { en: "Show Original LoRAs", zh: "显示原始 LoRA 底栏" },
    control_gap: { en: "BASE Control Gap", zh: "BASE 对照列间距" },
    max_canvas_megapixels: { en: "Maximum Canvas (MP)", zh: "最大画布（百万像素）" },
    custom_style: { en: "Custom Style", zh: "自定义样式" },
  },
};

const OUTPUT_LABELS = {
  LoraStack: {
    lora_stack: { en: "LoRA Stack", zh: "LoRA 组合" },
  },
  LoraStackSplitter: {
    lora_stack_list: { en: "LoRA Stack List", zh: "LoRA 组合列表" },
  },
  LoraStackLister: {
    lora_stack_list: { en: "LoRA Stack List", zh: "LoRA 组合列表" },
  },
  MultiPromptSample: {
    comparison_sheet: { en: "Comparison Sheet", zh: "XY 对比图" },
  },
};

const LORA_GROUPS = [
  {
    minimumCount: 2,
    widgets: [
      "lora_b_name",
      "lora_b_trigger",
      "lora_b_min_strength",
      "lora_b_max_strength",
    ],
  },
  {
    minimumCount: 3,
    widgets: [
      "lora_c_name",
      "lora_c_trigger",
      "lora_c_min_strength",
      "lora_c_max_strength",
    ],
  },
];

const STACK_ITEM_GROUPS = Array.from({ length: 16 }, (_, index) => ({
  minimumCount: index + 1,
  widgets: [
    `lora_${index + 1}_name`,
    `lora_${index + 1}_trigger`,
    `lora_${index + 1}_strength`,
  ],
}));

const PROMPT_GROUPS = Array.from({ length: 16 }, (_, index) => ({
  minimumCount: index + 1,
  widgets: [`positive_prompt_${index + 1}`],
}));

function widgetElements(widget) {
  return [widget?.element, widget?.inputEl, widget?.el, widget?.container].filter(
    (element, index, values) => element?.style && values.indexOf(element) === index,
  );
}

function widgetOptionTargets(widget) {
  return [widget?.options, widget?._state?.options].filter(
    (options, index, values) => options && values.indexOf(options) === index,
  );
}

function activeLanguage() {
  let locale = "";
  try {
    locale =
      app.extensionManager?.setting?.get?.("Comfy.Locale") ??
      app.ui?.settings?.getSettingValue?.("Comfy.Locale") ??
      "";
  } catch {
    // Settings may not be ready while a workflow is being restored.
  }
  if (!locale) locale = navigator.languages?.[0] ?? navigator.language ?? "en";
  return String(locale).toLowerCase().startsWith("zh") ? "zh" : "en";
}

function installOptionLabels(widget, labels) {
  if (!widget || !labels) return;
  for (const options of widgetOptionTargets(widget)) {
    if (options[LOCALIZED_OPTION]) continue;
    const originalGetOptionLabel = options.getOptionLabel;
    options.getOptionLabel = function (value) {
      const translated = labels[String(value)]?.[activeLanguage()];
      if (translated) return translated;
      if (typeof originalGetOptionLabel === "function") {
        return originalGetOptionLabel.call(this, value);
      }
      return value == null ? "" : String(value);
    };
    options[LOCALIZED_OPTION] = true;
  }
}

function installToggleLabels(widget, labels) {
  if (!widget || !labels) return;
  const translated = labels[activeLanguage()] ?? labels.en;
  for (const options of widgetOptionTargets(widget)) {
    options.label_on = translated.label_on;
    options.label_off = translated.label_off;
    options.on = translated.label_on;
    options.off = translated.label_off;
  }
}

function installWidgetTranslations(node, nodeName) {
  const optionLabels = OPTION_LABELS[nodeName] ?? {};
  const toggleLabels = TOGGLE_LABELS[nodeName] ?? {};
  for (const widget of node.widgets ?? []) {
    installOptionLabels(widget, optionLabels[widget.name]);
    installToggleLabels(widget, toggleLabels[widget.name]);
  }
}

function preserveUnavailableLoraValues(node) {
  for (const widget of node.widgets ?? []) {
    if (!/^lora_(?:[abc]|\d+)_name$/.test(String(widget.name ?? ""))) continue;
    const value = widget.value;
    if (value == null || value === "") continue;
    for (const options of widgetOptionTargets(widget)) {
      const values = options.values;
      if (!Array.isArray(values) || values.includes(value)) continue;
      try {
        // Replacing the array also invalidates Node 2.0's reactive option cache.
        options.values = [...values, value];
      } catch {
        // Older widgets can expose a read-only property backed by a mutable array.
        try {
          values.push(value);
        } catch {
          // A fully frozen option list cannot be extended, but must not block UI sync.
        }
      }
    }
  }
}

function localizedInputLabel(nodeName, inputName) {
  const language = activeLanguage();
  const fixed = INPUT_LABELS[nodeName]?.[inputName];
  if (fixed) return fixed[language] ?? fixed.en;

  let match;
  if (nodeName === STACK_NODE) {
    match = /^lora_(\d+)_(name|trigger|strength)$/.exec(inputName);
    if (match) {
      const fields = {
        name: { en: "File", zh: "文件" },
        trigger: { en: "Trigger Words", zh: "触发词" },
        strength: { en: "Strength", zh: "强度" },
      };
      return `LoRA ${match[1]} ${fields[match[2]][language] ?? fields[match[2]].en}`;
    }
  }
  if (nodeName === STACK_LISTER_NODE) {
    match = /^stack_(\d+)$/.exec(inputName);
    if (match) return language === "zh" ? `LoRA 组合 ${match[1]}` : `LoRA Stack ${match[1]}`;
  }
  if (nodeName === MULTI_PROMPT_NODE) {
    match = /^positive_prompt_(\d+)$/.exec(inputName);
    if (match) return language === "zh" ? `正面提示词 ${match[1]}` : `Positive Prompt ${match[1]}`;
  }
  return null;
}

function installNodeLabels(node, nodeName) {
  for (const widget of node.widgets ?? []) {
    const label = localizedInputLabel(nodeName, String(widget.name ?? ""));
    if (!label) continue;
    widget.label = label;
    if (
      nodeName === MULTI_PROMPT_NODE &&
      /^(?:prompt_prefix|negative_prompt|positive_prompt_\d+)$/.test(widget.name)
    ) {
      for (const options of widgetOptionTargets(widget)) options.placeholder = label;
      const renderedElements = [...widgetElements(widget)];
      for (const container of document.querySelectorAll("[node-id][node-type]")) {
        if (
          container.getAttribute("node-id") !== String(node.id) ||
          container.getAttribute("node-type") !== MULTI_PROMPT_NODE ||
          container.querySelector("label")?.textContent?.trim() !== label
        ) {
          continue;
        }
        renderedElements.push(container);
      }
      for (const element of new Set(renderedElements)) {
        const input = element.matches?.("textarea, input")
          ? element
          : element.querySelector?.("textarea, input");
        if (!input) continue;
        input.placeholder = label;
        input.setAttribute("aria-label", label);
      }
    }
  }
  for (const input of node.inputs ?? []) {
    const label = localizedInputLabel(nodeName, String(input.name ?? ""));
    if (label) input.label = label;
  }
  for (const output of node.outputs ?? []) {
    const labels = OUTPUT_LABELS[nodeName]?.[String(output.name ?? "")];
    const label = labels?.[activeLanguage()] ?? labels?.en;
    if (label) output.label = label;
  }
}

function restoreHiddenOption(widget) {
  // Visibility for these optional groups is owned by their count widget.
  // A restored Node 2.0 store can otherwise preserve this extension's stale true.
  for (const options of widgetOptionTargets(widget)) delete options.hidden;
}

function setElementVisible(element, visible) {
  if (!element[DOM_STATE]) {
    element[DOM_STATE] = Object.fromEntries(
      ["display", "visibility", "height", "width", "position", "left"].map((property) => [
        property,
        {
          value: element.style.getPropertyValue(property),
          priority: element.style.getPropertyPriority(property),
        },
      ]),
    );
  }
  const original = element[DOM_STATE];
  if (visible) {
    for (const [property, state] of Object.entries(original)) {
      if (state.value) element.style.setProperty(property, state.value, state.priority);
      else element.style.removeProperty(property);
    }
    return;
  }
  element.style.setProperty("display", "none", "important");
  element.style.setProperty("visibility", "hidden", "important");
  element.style.setProperty("height", "0px", "important");
  element.style.setProperty("width", "0px", "important");
  element.style.setProperty("position", "absolute", "important");
  element.style.setProperty("left", "-100000px", "important");
}

function captureVisibleState(widget) {
  const state = widget[WIDGET_STATE] ?? { hiddenByLoraTester: false };
  if (!state.hiddenByLoraTester) {
    state.type = widget.type;
    state.computeSize = widget.computeSize;
    state.hasOwnDraw = Object.prototype.hasOwnProperty.call(widget, "draw");
    state.draw = widget.draw;
    state.hidden = widget.hidden;
    state.computedHeight = widget.computedHeight;
    state.y = widget.y;
    state.lastY = widget.last_y;
  }
  widget[WIDGET_STATE] = state;
  return state;
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;

  // Workflow restore can replace a widget after this extension hid its predecessor.
  // The replacement inherits hidden options but not the symbol-backed restore state.
  if (visible && !widget[WIDGET_STATE]) {
    delete widget.hidden;
    for (const options of widgetOptionTargets(widget)) delete options.hidden;
    if (
      widget.type === HIDDEN_WIDGET_TYPE &&
      widget._state?.type &&
      widget._state.type !== HIDDEN_WIDGET_TYPE
    ) {
      widget.type = widget._state.type;
      delete widget.computeSize;
      delete widget.draw;
    }
    widgetElements(widget).forEach((element) => setElementVisible(element, true));
    return;
  }

  const state = captureVisibleState(widget);

  if (visible) {
    if (!state.hiddenByLoraTester) return;
    widget.type = state.type;
    if (state.computeSize === undefined) delete widget.computeSize;
    else widget.computeSize = state.computeSize;
    if (state.hasOwnDraw) widget.draw = state.draw;
    else delete widget.draw;
    widget.hidden = state.hidden;
    if (state.computedHeight === undefined) delete widget.computedHeight;
    else widget.computedHeight = state.computedHeight;
    if (state.y === undefined) delete widget.y;
    else widget.y = state.y;
    if (state.lastY === undefined) delete widget.last_y;
    else widget.last_y = state.lastY;
    restoreHiddenOption(widget);
    widgetElements(widget).forEach((element) => setElementVisible(element, true));
    state.hiddenByLoraTester = false;
    return;
  }

  if (!state.hiddenByLoraTester) {
    state.hiddenByLoraTester = true;
    widget.type = HIDDEN_WIDGET_TYPE;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
  }

  // Node 2.0 replaces the widget's private state while restoring workflows.
  // Reapply every target even when this widget was already hidden earlier.
  widget.hidden = true;
  widget.options ??= {};
  for (const options of widgetOptionTargets(widget)) options.hidden = true;
  widget.computedHeight = 0;
  widget.y = -100000;
  widget.last_y = -100000;
  widgetElements(widget).forEach((element) => setElementVisible(element, false));
}

function refreshReactiveCollection(node, property) {
  const collection = node[property];
  if (!Array.isArray(collection)) return;
  const snapshot = [...collection];
  // Node 2.0 wraps this property in a shallow reactive array. Reassigning the
  // same widget references does not invalidate its processed-widget snapshot.
  node[property] = [];
  node[property] = snapshot;
}

function resizeNodeToWidgets(node) {
  requestAnimationFrame(() => {
    const computed = node.computeSize?.();
    if (!computed || !Number.isFinite(computed[1])) return;
    const width = Math.max(node.size?.[0] ?? 0, computed[0] ?? 0);
    node.setSize?.([width, computed[1]]);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  });
}

function refreshWidgetViews(node) {
  requestAnimationFrame(() => {
    const canvas = app.canvas;
    if (!canvas) return;
    const selected = [...(canvas.selectedItems ?? [])];
    if (selected.some((item) => String(item?.id) === String(node.id))) {
      canvas.selectItems?.(selected, false);
    } else {
      canvas.selectNode?.(node, false);
    }
  });
}

function updateLoraGroups(node, value) {
  const count = Math.min(3, Math.max(1, Number.parseInt(value, 10) || 1));
  for (const group of LORA_GROUPS) {
    const visible = count >= group.minimumCount;
    for (const name of group.widgets) {
      setWidgetVisible(node.widgets?.find((widget) => widget.name === name), visible);
    }
  }
  refreshReactiveCollection(node, "widgets");
  node.graph?.incrementVersion?.();
  resizeNodeToWidgets(node);
}

function updateWidgetGroups(node, value, groups, maximum) {
  const count = Math.min(maximum, Math.max(1, Number.parseInt(value, 10) || 1));
  for (const group of groups) {
    const visible = count >= group.minimumCount;
    for (const name of group.widgets) {
      setWidgetVisible(node.widgets?.find((widget) => widget.name === name), visible);
    }
  }
  refreshReactiveCollection(node, "widgets");
  node.graph?.incrementVersion?.();
  resizeNodeToWidgets(node);
}

function installDynamicCount(node, widgetName, groups, maximum) {
  const countWidget = node.widgets?.find((widget) => widget.name === widgetName);
  if (!countWidget) return;
  if (countWidget.__loraTesterInstalled) {
    updateWidgetGroups(node, countWidget.value, groups, maximum);
    return;
  }
  countWidget.__loraTesterInstalled = true;
  const originalCallback = countWidget.callback;
  countWidget.callback = function (value, ...args) {
    countWidget.value = value;
    if (this && this !== countWidget) this.value = value;
    const result = originalCallback?.apply(this, [value, ...args]);
    updateWidgetGroups(node, value, groups, maximum);
    refreshWidgetViews(node);
    scheduleNodeUi(node);
    return result;
  };
  updateWidgetGroups(node, countWidget.value, groups, maximum);
}

function installDynamicLoraCount(node) {
  const countWidget = node.widgets?.find((widget) => widget.name === "lora_count");
  if (!countWidget) return;
  if (countWidget.__loraTesterInstalled) {
    updateLoraGroups(node, countWidget.value);
    return;
  }
  countWidget.__loraTesterInstalled = true;

  const originalCallback = countWidget.callback;
  countWidget.callback = function (value, ...args) {
    countWidget.value = value;
    if (this && this !== countWidget) this.value = value;
    const result = originalCallback?.apply(this, [value, ...args]);
    updateLoraGroups(node, value);
    refreshWidgetViews(node);
    scheduleNodeUi(node, TARGET_NODE);
    return result;
  };

  updateLoraGroups(node, countWidget.value);
}

function installMultiPromptLayout(node) {
  const width = node.size?.[0] ?? 0;
  if (width >= MULTI_PROMPT_MIN_WIDTH) return;
  const height = node.size?.[1] ?? node.computeSize?.()?.[1] ?? 200;
  node.setSize?.([MULTI_PROMPT_MIN_WIDTH, height]);
}

function inputIsConnected(input) {
  return input?.link != null || (Array.isArray(input?.linkIds) && input.linkIds.length > 0);
}

function updateStackListInputs(node) {
  const currentInputs = [...(node.inputs ?? [])];
  const templates = node[STACK_INPUT_TEMPLATES] ?? new Map();
  for (const input of currentInputs) {
    const match = /^stack_(\d+)$/.exec(String(input?.name ?? ""));
    if (match) templates.set(Number(match[1]), input);
  }
  node[STACK_INPUT_TEMPLATES] = templates;
  if (!templates.size) return;

  let lastConnected = 0;
  for (const [index, input] of templates) {
    if (inputIsConnected(input)) lastConnected = Math.max(lastConnected, index);
  }
  const visibleThrough = Math.min(MAX_STACK_INPUTS, Math.max(1, lastConnected + 1));
  const stackInputs = [...templates.entries()]
    .filter(([index]) => index <= visibleThrough)
    .sort(([a], [b]) => a - b)
    .map(([, input]) => input);
  const otherInputs = currentInputs.filter(
    (input) => !/^stack_(\d+)$/.test(String(input?.name ?? "")),
  );
  node.inputs = [...stackInputs, ...otherInputs];
  resizeNodeToWidgets(node);
  node.setDirtyCanvas?.(true, true);
}

function installDynamicStackList(node) {
  if (node.__loraTesterStackListInstalled) {
    updateStackListInputs(node);
    return;
  }
  node.__loraTesterStackListInstalled = true;
  const originalConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnectionsChange?.apply(this, args);
    updateStackListInputs(this);
    return result;
  };
  updateStackListInputs(node);
}

function nodeNameForUi(node) {
  return String(
    node?.constructor?.nodeData?.name ??
      node?.constructor?.comfyClass ??
      node?.comfyClass ??
      node?.type ??
      "",
  );
}

function supportsNodeUi(nodeName) {
  return (
    nodeName === TARGET_NODE ||
    nodeName === STACK_NODE ||
    nodeName === STACK_SPLITTER_NODE ||
    nodeName === STACK_LISTER_NODE ||
    nodeName === MULTI_PROMPT_NODE ||
    nodeName in OPTION_LABELS ||
    nodeName in INPUT_LABELS ||
    nodeName in OUTPUT_LABELS
  );
}

function applyNodeUi(node, nodeName = nodeNameForUi(node)) {
  if (!supportsNodeUi(nodeName)) return;
  preserveUnavailableLoraValues(node);
  installWidgetTranslations(node, nodeName);
  installNodeLabels(node, nodeName);
  if (nodeName === TARGET_NODE) installDynamicLoraCount(node);
  if (nodeName === STACK_NODE) {
    installDynamicCount(node, "lora_count", STACK_ITEM_GROUPS, 16);
  }
  if (nodeName === MULTI_PROMPT_NODE) {
    installMultiPromptLayout(node);
    installDynamicCount(node, "prompt_count", PROMPT_GROUPS, 16);
  }
  if (nodeName === STACK_LISTER_NODE) installDynamicStackList(node);
}

function scheduleNodeUi(node, nodeName = nodeNameForUi(node)) {
  if (!supportsNodeUi(nodeName)) return;
  queueMicrotask(() => {
    applyNodeUi(node, nodeName);
    requestAnimationFrame(() => {
      applyNodeUi(node, nodeName);
      requestAnimationFrame(() => applyNodeUi(node, nodeName));
    });
  });
}

function scheduleGraphNodeUi(graph) {
  for (const node of graph?._nodes ?? []) scheduleNodeUi(node);
}

app.registerExtension({
  name: "LoraTester.NodeUi",
  setup() {
    const canvas = app.canvas;
    if (!canvas?.setGraph || canvas[GRAPH_SYNC_INSTALLED]) return;
    canvas[GRAPH_SYNC_INSTALLED] = true;
    const originalSetGraph = canvas.setGraph;
    canvas.setGraph = function (...args) {
      const result = originalSetGraph.apply(this, args);
      scheduleGraphNodeUi(this.graph);
      return result;
    };
  },
  loadedGraphNode(node) {
    scheduleNodeUi(node);
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    const hasDynamicLoraCount = nodeData.name === TARGET_NODE;
    const hasDynamicStackCount = nodeData.name === STACK_NODE;
    const hasDynamicPromptCount = nodeData.name === MULTI_PROMPT_NODE;
    const hasDynamicStackList = nodeData.name === STACK_LISTER_NODE;
    const hasWidgetTranslations = nodeData.name in OPTION_LABELS;
    const hasNodeLabels =
      nodeData.name in INPUT_LABELS ||
      nodeData.name in OUTPUT_LABELS ||
      nodeData.name === STACK_LISTER_NODE ||
      nodeData.name === STACK_SPLITTER_NODE;
    if (!hasDynamicLoraCount && !hasDynamicStackCount && !hasDynamicPromptCount && !hasDynamicStackList && !hasWidgetTranslations && !hasNodeLabels) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      scheduleNodeUi(this, nodeData.name);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      scheduleNodeUi(this, nodeData.name);
      return result;
    };

    const originalOnAfterGraphConfigured = nodeType.prototype.onAfterGraphConfigured;
    nodeType.prototype.onAfterGraphConfigured = function (...args) {
      const result = originalOnAfterGraphConfigured?.apply(this, args);
      applyNodeUi(this, nodeData.name);
      return result;
    };
  },
});
