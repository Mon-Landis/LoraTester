import { app } from "../../scripts/app.js";

const TARGET_NODE = "LoraTesterSampler";
const HIDDEN_WIDGET_TYPE = "lora-tester-hidden";
const WIDGET_STATE = Symbol("loraTesterWidgetState");
const DOM_STATE = Symbol("loraTesterDomState");
const LOCALIZED_OPTION = Symbol("loraTesterLocalizedOption");

const OPTION_LABELS = {
  LoraTesterSampler: {
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
        label_on: "show names and max weights",
        label_off: "hide names and max weights",
      },
      zh: {
        label_on: "显示名称和最高权重",
        label_off: "隐藏名称和最高权重",
      },
    },
  },
};

const LORA_GROUPS = [
  {
    minimumCount: 2,
    widgets: ["lora_b_name", "lora_b_trigger", "lora_b_max_strength"],
  },
  {
    minimumCount: 3,
    widgets: ["lora_c_name", "lora_c_trigger", "lora_c_max_strength"],
  },
];

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

function restoreHiddenOption(widget, state) {
  for (const options of widgetOptionTargets(widget)) {
    if (state.hasOptionHidden) options.hidden = state.optionHidden;
    else delete options.hidden;
  }
}

function setElementVisible(element, visible) {
  if (!element[DOM_STATE]) {
    element[DOM_STATE] = {
      display: element.style.display,
      visibility: element.style.visibility,
      height: element.style.height,
      width: element.style.width,
      position: element.style.position,
      left: element.style.left,
    };
  }
  const original = element[DOM_STATE];
  if (visible) {
    Object.assign(element.style, original);
    return;
  }
  Object.assign(element.style, {
    display: "none",
    visibility: "hidden",
    height: "0px",
    width: "0px",
    position: "absolute",
    left: "-100000px",
  });
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
    const options = widget?._state?.options ?? widget.options ?? {};
    state.hasOptionHidden = Object.prototype.hasOwnProperty.call(options, "hidden");
    state.optionHidden = options.hidden;
  }
  widget[WIDGET_STATE] = state;
  return state;
}

function setWidgetVisible(widget, visible) {
  if (!widget) return;
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
    restoreHiddenOption(widget, state);
    widgetElements(widget).forEach((element) => setElementVisible(element, true));
    state.hiddenByLoraTester = false;
    return;
  }

  if (state.hiddenByLoraTester) return;
  state.hiddenByLoraTester = true;
  widget.type = HIDDEN_WIDGET_TYPE;
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  widget.hidden = true;
  widget.options ??= {};
  for (const options of widgetOptionTargets(widget)) options.hidden = true;
  widget.computedHeight = 0;
  widgetElements(widget).forEach((element) => setElementVisible(element, false));
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
  node.graph?.incrementVersion?.();
  resizeNodeToWidgets(node);
}

function installDynamicLoraCount(node) {
  const countWidget = node.widgets?.find((widget) => widget.name === "lora_count");
  if (!countWidget || countWidget.__loraTesterInstalled) return;
  countWidget.__loraTesterInstalled = true;

  const originalCallback = countWidget.callback;
  countWidget.callback = function (value, ...args) {
    countWidget.value = value;
    if (this && this !== countWidget) this.value = value;
    const result = originalCallback?.apply(this, [value, ...args]);
    updateLoraGroups(node, value);
    refreshWidgetViews(node);
    return result;
  };

  updateLoraGroups(node, countWidget.value);
}

app.registerExtension({
  name: "LoraTester.NodeUi",
  beforeRegisterNodeDef(nodeType, nodeData) {
    const hasDynamicLoraCount = nodeData.name === TARGET_NODE;
    const hasWidgetTranslations = nodeData.name in OPTION_LABELS;
    if (!hasDynamicLoraCount && !hasWidgetTranslations) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = originalOnNodeCreated?.apply(this, args);
      queueMicrotask(() => {
        installWidgetTranslations(this, nodeData.name);
        if (hasDynamicLoraCount) installDynamicLoraCount(this);
      });
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      queueMicrotask(() => {
        installWidgetTranslations(this, nodeData.name);
        if (hasDynamicLoraCount) {
          installDynamicLoraCount(this);
          const countWidget = this.widgets?.find((widget) => widget.name === "lora_count");
          updateLoraGroups(this, countWidget?.value);
        }
      });
      return result;
    };
  },
});
