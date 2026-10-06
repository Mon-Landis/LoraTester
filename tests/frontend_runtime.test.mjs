import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { createContext, runInContext } from "node:vm";

const source = readFileSync(new URL("../web/lora_tester.js", import.meta.url), "utf8")
  .replace(/^import .*;\r?\n/gm, "");

function runtime() {
  const microtasks = [];
  const frames = [];
  const observers = [];
  const state = { locale: "zh", extension: null, queries: 0 };
  const app = {
    extensionManager: { setting: { get: (name) => name === "Comfy.Locale" ? state.locale : false } },
    ui: { settings: { getSettingValue: () => false } },
    registerExtension: (extension) => { state.extension = extension; },
  };
  const document = {
    documentElement: {},
    querySelector: () => null,
    querySelectorAll: () => { state.queries += 1; return []; },
  };
  const context = createContext({
    app, document, console, navigator: { language: "zh" }, api: {}, ComfyWidgets: {},
    queueMicrotask: (callback) => microtasks.push(callback),
    requestAnimationFrame: (callback) => frames.push(callback),
    MutationObserver: class {
      constructor(callback) { this.callback = callback; observers.push(this); }
      observe() {}
    },
  });
  runInContext(source, context);
  return { context, state, frames, microtasks, observers, app };
}

function element() {
  let text = "";
  const attributes = new Map();
  return {
    style: { getPropertyValue: () => "", getPropertyPriority: () => "", setProperty() {}, removeProperty() {} },
    writes: 0,
    get textContent() { return text; },
    set textContent(value) { text = value; this.writes += 1; },
    getAttribute: (name) => attributes.get(name) ?? null,
    setAttribute: (name, value) => attributes.set(name, value),
    removeAttribute: (name) => attributes.delete(name),
    querySelectorAll: () => [],
    matches: () => false,
  };
}

function artistNode() {
  return {
    type: "LoraStack", id: 1, inputs: [], outputs: [],
    widgets: [
      { name: "lora_1_name", value: "__lora_tester_artist_tag__", options: {} },
      { name: "lora_1_trigger", value: "@wlop", options: {} },
      { name: "lora_1_strength", value: 1, options: {} },
    ],
  };
}

test("artist-mode labels settle instead of alternating between style and artist", () => {
  const fixture = runtime();
  fixture.context.node = artistNode();
  runInContext('installNodeLabels(node,"LoraStack"); artistModeLabels(node,"LoraStack")', fixture.context);
  const changes = runInContext('Array.from({length:5},()=>installNodeLabels(node,"LoraStack") || artistModeLabels(node,"LoraStack"))', fixture.context);
  assert.equal(changes.some(Boolean), false);
  assert.equal(fixture.context.node.widgets[1].label, "画师 1 Tag");
  fixture.context.node.widgets[0].value = "style.safetensors";
  runInContext('installNodeLabels(node,"LoraStack"); artistModeLabels(node,"LoraStack")', fixture.context);
  assert.equal(fixture.context.node.widgets[1].label, "风格 1 触发词");
  fixture.state.locale = "en";
  runInContext('installNodeLabels(node,"LoraStack"); artistModeLabels(node,"LoraStack")', fixture.context);
  assert.equal(fixture.context.node.widgets[1].label, "Style 1 Trigger Words");
});

test("settled reactive labels are not assigned again", () => {
  const fixture = runtime();
  let writes = 0;
  let label = "Style";
  fixture.context.widget = {
    get label() { return label; },
    set label(value) { label = value; writes += 1; },
    _state: {
      get label() { return label; },
      set label(value) { label = value; writes += 1; },
    },
  };
  runInContext('setWidgetLabel(widget,"Style")', fixture.context);
  assert.equal(writes, 0);
});

test("repeated lifecycle schedules share one three-pass refresh", () => {
  const fixture = runtime();
  fixture.context.node = artistNode();
  runInContext('globalThis.calls=0; applyNodeUi=()=>{globalThis.calls+=1}; for(let index=0;index<10;index+=1) scheduleNodeUi(node,"LoraStack")', fixture.context);
  assert.equal(fixture.microtasks.length, 1);
  fixture.microtasks.shift()();
  fixture.frames.shift()();
  fixture.frames.shift()();
  assert.equal(fixture.context.calls, 3);
  runInContext('scheduleNodeUi(node,"LoraStack")', fixture.context);
  assert.equal(fixture.microtasks.length, 1);
});

test("pending refresh passes observe edits without another scheduling burst", () => {
  const fixture = runtime();
  fixture.context.node = artistNode();
  runInContext('globalThis.seen=[]; applyNodeUi=node=>seen.push(node.widgets[0].value); scheduleNodeUi(node,"LoraStack")', fixture.context);
  fixture.microtasks.shift()();
  fixture.context.node.widgets[0].value = "other.safetensors";
  runInContext('scheduleNodeUi(node,"LoraStack")', fixture.context);
  fixture.frames.shift()();
  fixture.frames.shift()();
  assert.equal(fixture.microtasks.length, 0);
  assert.equal(fixture.context.seen[2], "other.safetensors");
});

test("cached preview output is installed only when its text changes", () => {
  const fixture = runtime();
  fixture.context.node = { type: "LoraTesterAxisPreview", properties: {}, widgets: [], setDirtyCanvas() {} };
  runInContext('globalThis.installs=0; installAxisPreview=node=>{installs+=1;node.widgets=[{name:"axis_preview_text",value:node.properties.loraTesterAxisPreviewText}]}; updateAxisPreview(node,{text:["tree"]}); updateAxisPreview(node,{text:["tree"]}); updateAxisPreview(node,{text:["new tree"]})', fixture.context);
  assert.equal(fixture.context.installs, 2);
  assert.equal(fixture.context.node.properties.loraTesterAxisPreviewText, "new tree");
});

test("execution notifications preserve prior handlers without full UI refresh", () => {
  const fixture = runtime();
  const OriginalNode = class {
    onExecuted() { return "original"; }
  };
  fixture.state.extension.beforeRegisterNodeDef(OriginalNode, { name: "LoraTesterXYSampler" });
  fixture.context.node = new OriginalNode();
  runInContext('globalThis.warningUpdates=0; updateAnimaRemapWarning=()=>{warningUpdates+=1}', fixture.context);
  assert.equal(fixture.context.node.onExecuted({}), "original");
  assert.equal(fixture.microtasks.length, 0);
  fixture.context.node.onExecuted({ lora_tester_anima_remap: [{ message: "warning" }] });
  fixture.context.node.onExecuted({ lora_tester_anima_remap: [{ message: "warning" }] });
  assert.equal(fixture.context.warningUpdates, 1);
});

test("unchanged mixer warning does not replace its DOM text", () => {
  const fixture = runtime();
  const warning = { name: "lora_tester_anima_mixer_warning", element: element(), options: {} };
  fixture.context.node = { type: "LoraTesterXYSampler", widgets: [warning], inputs: [], graph: { incrementVersion() {} }, setDirtyCanvas() {} };
  runInContext('updateMixerWarning(node,"LoraTesterXYSampler")', fixture.context);
  const writes = warning.element.writes;
  runInContext('updateMixerWarning(node,"LoraTesterXYSampler")', fixture.context);
  assert.equal(warning.element.writes, writes);
});

test("unchanged XY warning does not emit DOM mutations", () => {
  const fixture = runtime();
  const warning = { name: "lora_tester_xy_warning", element: element(), options: {}, __loraTesterWarningVisible: false };
  fixture.context.node = { type: "LoraTesterXYSampler", widgets: [warning], inputs: [], setDirtyCanvas() {} };
  runInContext('updateXyAxisState(node); updateXyAxisState(node)', fixture.context);
  assert.equal(warning.element.writes, 0);
});

test("progress text mutations do not scan all disabled XY controls", () => {
  const fixture = runtime();
  fixture.context.node = { id: 1 };
  runInContext('setXyDomOverride(node,"seed",true)', fixture.context);
  const baseline = fixture.state.queries;
  fixture.observers[0].callback([{ addedNodes: [{ nodeType: 3 }] }]);
  while (fixture.microtasks.length) fixture.microtasks.shift()();
  while (fixture.frames.length) fixture.frames.shift()();
  assert.equal(fixture.state.queries, baseline);
});

test("new control mounts coalesce XY overrides into one animation frame", () => {
  const fixture = runtime();
  fixture.context.node = { id: 1 };
  runInContext('setXyDomOverride(node,"seed",true)', fixture.context);
  const baseline = fixture.state.queries;
  const added = { nodeType: 1, matches: () => true, querySelector: () => null };
  for (let index = 0; index < 10; index += 1) fixture.observers[0].callback([{ addedNodes: [added] }]);
  assert.equal(fixture.microtasks.length, 0);
  assert.equal(fixture.frames.length, 1);
  fixture.frames.shift()();
  assert.equal(fixture.state.queries, baseline + 1);
});
