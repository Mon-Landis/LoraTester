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

test("single style name switches localize without changing boolean values", () => {
  const fixture = runtime();
  for (const nodeName of ["LoraTesterLoraStackAxis", "LoraTesterAxisComposer"]) {
    fixture.context.nodeName = nodeName;
    fixture.context.node = {
      type: nodeName,
      inputs: [],
      outputs: [],
      widgets: [{
        name: "show_single_style_name", type: "toggle", value: true,
        options: {}, _state: { options: {} },
      }],
    };
    runInContext('installNodeLabels(node,nodeName); installWidgetTranslations(node,nodeName)', fixture.context);
    const widget = fixture.context.node.widgets[0];
    assert.equal(widget.label, "直接展示单风格元素");
    assert.equal(widget.options.label_on, "显示单风格名称");
    assert.equal(widget.options.label_off, "显示代号和权重");
    assert.equal(widget._state.options.label_on, "显示单风格名称");
    assert.equal(widget.value, true);
    fixture.state.locale = "en";
    widget.value = false;
    runInContext('installNodeLabels(node,nodeName); installWidgetTranslations(node,nodeName)', fixture.context);
    assert.equal(widget.label, "Show Single Style Name");
    assert.equal(widget.options.label_on, "show single style name");
    assert.equal(widget.options.label_off, "show code and weight");
    assert.equal(widget.value, false);
    fixture.state.locale = "zh";
  }
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

test("stack mixer strength warning follows dependency availability and settles", () => {
  const fixture = runtime();
  const warning = { name: "lora_tester_anima_mixer_warning", element: element(), options: {} };
  fixture.context.node = {
    type: "LoraStackMixerStrength", inputs: [], widgets: [warning],
    graph: { incrementVersion() {} }, setDirtyCanvas() {},
  };
  runInContext('globalThis.resizes=0; resizeNodeToWidgets=()=>{resizes+=1}; updateMixerWarning(node,"LoraStackMixerStrength")', fixture.context);
  assert.equal(warning.__loraTesterWarningVisible, true);
  assert.match(warning.element.textContent, /未检测到 Anima Artist Mixer/);
  const writes = warning.element.writes;
  const resizes = fixture.context.resizes;
  runInContext('updateMixerWarning(node,"LoraStackMixerStrength")', fixture.context);
  assert.equal(warning.element.writes, writes);
  assert.equal(fixture.context.resizes, resizes);
  runInContext('app.extensionManager.nodeDefs=new Map([["AnimaArtistPack",{}],["AnimaArtistAdapterMixer",{}]]); updateMixerWarning(node,"LoraStackMixerStrength")', fixture.context);
  assert.equal(warning.__loraTesterWarningVisible, false);
  fixture.state.locale = "en";
  runInContext('app.extensionManager.nodeDefs.clear(); updateMixerWarning(node,"LoraStackMixerStrength")', fixture.context);
  assert.equal(warning.__loraTesterWarningVisible, true);
  assert.match(warning.element.textContent, /Stack Mixer strength is retained/);
  assert.equal(warning.serialize, undefined);
});

test("stack mixer strength passthrough preserves artists and axis metadata", () => {
  const fixture = runtime();
  const source = artistNode();
  source.widgets[1].value = "@first, @second";
  fixture.context.node = {
    id: 2, type: "LoraStackMixerStrength", widgets: [],
    inputs: [{ name: "lora_stack", link: 1 }],
    graph: { links: { 1: { origin_id: 1 } }, getNodeById: () => source },
  };
  runInContext('globalThis.entries=stackEntryDataFromSource(node); globalThis.count=styleStackCountFromSource(node); globalThis.counts=stackArtistCountsFromNode(node)', fixture.context);
  assert.equal(fixture.context.entries[0].artists.length, 2);
  assert.equal(fixture.context.count, 1);
  assert.equal(fixture.context.counts[0], 2);
});

function connectedStackNode(type, sources = {}, values = {}) {
  const names = Object.keys(sources);
  return {
    type, graph: {}, inputs: names.map((name, index) => ({ name, link: index + 1 })),
    widgets: Object.entries(values).map(([name, value]) => ({ name, value, options: {} })),
    getInputNode: (index) => sources[names[index]],
  };
}

function artistTextNode(text) {
  return connectedStackNode("ArtistTagTextParser", {}, { artist_text: text });
}

test("prompt naming keeps axis metadata and a localized readable multiline layout", () => {
  const fixture = runtime();
  const prompts = connectedStackNode("LoraTesterMultiPromptInput", {}, { prompt_count: 3 });
  const named = connectedStackNode("LoraTesterPromptListName", { prompt_list: prompts }, { names: "first\n\nthird" });
  fixture.context.node = named;
  runInContext('installNodeLabels(node,"LoraTesterPromptListName")', fixture.context);
  assert.equal(named.widgets[0].label, "名称（每行一项）");
  assert.equal(named.widgets[0].options.placeholder, "名称（每行一项）");
  named.size = [240, 200];
  named.setSize = (size) => { named.size = size; };
  runInContext('installMultiPromptLayout(node)', fixture.context);
  assert.equal(named.size[0], 480);
  assert.equal(named.widgets[0].value, "first\n\nthird");
  fixture.context.axis = connectedStackNode("LoraTesterAxisComposer", { source: named });
  const metadata = runInContext('axisMetadataFromSource(axis)', fixture.context);
  assert.equal(metadata.count, 3);
  assert.deepEqual(Array.from(metadata.parameters), ["prompt"]);
  fixture.state.locale = "en";
  runInContext('installNodeLabels(node,"LoraTesterPromptListName")', fixture.context);
  assert.equal(named.widgets[0].label, "Names (One per Line)");
});

test("StyleStack extraction selects one entry and keeps duplicate positions", () => {
  const fixture = runtime();
  const first = artistTextNode("(@first:0.8)");
  const second = artistTextNode("@second, @third");
  const list = connectedStackNode("LoraStackLister", { stack_1: first, stack_2: second, stack_3: first });
  fixture.context.node = connectedStackNode("StyleStackExtract", { lora_stack_list: list }, { index: 0 });
  for (const [index, artists] of [[0, ["first"]], [1, ["second", "third"]], [2, ["first"]], [-1, ["first"]], [-2, ["first"]]]) {
    fixture.context.node.widgets[0].value = index;
    const actual = runInContext('stackEntryDataFromSource(node).flatMap(entry=>entry.artists)', fixture.context);
    assert.deepEqual(Array.from(actual), artists);
  }
  assert.equal(runInContext('stackEntryDataFromSource(node)[0].strength', fixture.context), 0.8);
  fixture.context.list = list;
  assert.deepEqual(Array.from(runInContext('stackArtistCountsFromNode(list)', fixture.context)), [1, 2, 1]);
  for (const index of [3, 99, -3]) {
    fixture.context.node.widgets[0].value = index;
    assert.equal(runInContext('stackEntryDataFromSource(node)', fixture.context), null);
  }
});

test("StyleStack insertion metadata matches all modes and boundary positions", () => {
  const fixture = runtime();
  const original = [artistTextNode("@first"), artistTextNode("@second, @third"), artistTextNode("@last")];
  const list = connectedStackNode("LoraStackLister", { stack_1: original[0], stack_2: original[1], stack_3: original[2] });
  const added = artistTextNode("@added, @extra, @third_added");
  for (const mode of ["before", "after", "replace"]) {
    for (const index of [0, 1, 2, 3, 99, -1, -2]) {
      fixture.context.node = connectedStackNode("StyleStackSet", { lora_stack_list: list, lora_stack: added }, { index, mode });
      const position = index === -1 ? 2 : index === -2 ? 0 : index;
      const expected = [["first"], ["second", "third"], ["last"]];
      const insertion = ["added", "extra", "third_added"];
      if (position >= expected.length) expected.push(insertion);
      else if (mode === "replace") expected[position] = insertion;
      else expected.splice(position + (mode === "after" ? 1 : 0), 0, insertion);
      assert.equal(runInContext('styleStackCountFromSource(node)', fixture.context), expected.length);
      const actual = runInContext('Array.from({length:styleStackCountFromSource(node)},(_,index)=>styleStackDataAtIndex(node,index).flatMap(entry=>entry.artists))', fixture.context);
      assert.deepEqual(JSON.parse(JSON.stringify(actual)), expected);
      assert.deepEqual(Array.from(runInContext('stackArtistCountsFromNode(node)', fixture.context)), expected.map((artists) => artists.length));
      const composer = connectedStackNode("LoraTesterAxisComposer", { source: fixture.context.node }, { include_base: false });
      fixture.context.composer = composer;
      assert.equal(runInContext('axisMetadataFromSource(composer).count', fixture.context), expected.length);
      assert.equal(runInContext('axisMetadataFromSource(composer).parameters.has("lora_stack")', fixture.context), true);
    }
  }
  for (const mode of ["before", "after", "replace"]) {
    fixture.context.node = connectedStackNode("StyleStackSet", {
      lora_stack_list: connectedStackNode("LoraStackLister"), lora_stack: added,
    }, { index: -1, mode });
    assert.equal(runInContext('styleStackCountFromSource(node)', fixture.context), 1);
    assert.equal(runInContext('styleStackDataAtIndex(node,0).length', fixture.context), 3);
    assert.deepEqual(Array.from(runInContext('stackArtistCountsFromNode(node)', fixture.context)), [3]);
  }
});

test("StyleStack extraction follows combination order without eager expansion", () => {
  const fixture = runtime();
  const splitter = connectedStackNode("LoraStackSplitter", { lora_stack: artistTextNode("@first, @second, @third") });
  fixture.context.node = connectedStackNode("StyleStackExtract", { lora_stack_list: splitter }, { index: 0 });
  const expected = [["first"], ["second"], ["third"], ["first", "second"], ["first", "third"], ["second", "third"], ["first", "second", "third"]];
  for (const [index, artists] of expected.entries()) {
    fixture.context.node.widgets[0].value = index;
    assert.deepEqual(Array.from(runInContext('stackEntryDataFromSource(node).flatMap(entry=>entry.artists)', fixture.context)), artists);
  }
  fixture.context.splitter = splitter;
  assert.deepEqual(Array.from(runInContext('stackArtistCountsFromNode(splitter)', fixture.context)), [1, 1, 1, 2, 2, 2, 3]);
  fixture.context.entries = Array.from({ length: 16 }, (_, index) => index);
  assert.deepEqual(Array.from(runInContext('stackCombinationAtIndex(entries,65534)', fixture.context)), Array.from({ length: 16 }, (_, index) => index));
  assert.deepEqual(Array.from(runInContext('stackCombinationAtIndex(entries,16)', fixture.context)), [0, 1]);
  assert.equal(runInContext('stackCombinationAtIndex(entries,65535)', fixture.context), null);
});

test("StyleStack extraction preserves flatten weight modes and original positions", () => {
  const fixture = runtime();
  const source = artistTextNode("(@first:0.8), @second");
  for (const [mode, original, expected] of [
    ["inherit", false, [[0.8], [1]]], ["normalize", true, [[0.8, 1], [1], [1]]],
    ["dual", true, [[0.8, 1], [1], [0.8], [1]]],
  ]) {
    const flat = connectedStackNode("LoraStackFlattener", { lora_stack: source }, { weight_mode: mode, include_original: original });
    const named = connectedStackNode("LoraStackListName", { lora_stack_list: flat });
    fixture.context.node = connectedStackNode("StyleStackExtract", { lora_stack_list: named }, { index: 0 });
    for (const [index, strengths] of expected.entries()) {
      fixture.context.node.widgets[0].value = index;
      assert.deepEqual(Array.from(runInContext('stackEntryDataFromSource(node).map(entry=>entry.strength)', fixture.context)), strengths);
    }
  }
  const changed = connectedStackNode("StyleStackSet", {
    lora_stack_list: connectedStackNode("LoraStackLister", { stack_1: source }),
    lora_stack: artistTextNode("@replacement"),
  }, { index: -1, mode: "replace" });
  const extracted = connectedStackNode("StyleStackExtract", { lora_stack_list: changed }, { index: -1 });
  fixture.context.node = connectedStackNode("LoraStackFlattener", { lora_stack: extracted }, { weight_mode: "dual", include_original: true });
  assert.equal(runInContext('styleStackCountFromSource(node)', fixture.context), 1);
  assert.deepEqual(Array.from(runInContext('stackArtistCountsFromNode(node)', fixture.context)), [1]);
});

test("StyleStack controls localize stable enum values and observe both widget paths", () => {
  const fixture = runtime();
  fixture.context.node = connectedStackNode("StyleStackSet", {}, { index: -1, mode: "after" });
  fixture.context.node.widgets.push({ name: "lora_stack", value: null });
  runInContext('installNodeLabels(node,"StyleStackSet"); installWidgetTranslations(node,"StyleStackSet")', fixture.context);
  const mode = fixture.context.node.widgets[1];
  assert.equal(fixture.context.node.widgets[2].label, "风格组合");
  assert.equal(mode.label, "插入方式");
  assert.equal(mode.options.getOptionLabel("after"), "后");
  assert.equal(mode.value, "after");
  fixture.state.locale = "en";
  runInContext('installNodeLabels(node,"StyleStackSet"); installWidgetTranslations(node,"StyleStackSet")', fixture.context);
  assert.equal(fixture.context.node.widgets[2].label, "StyleStack");
  assert.equal(mode.label, "Insertion Mode");
  assert.equal(mode.options.getOptionLabel("after"), "After");
  runInContext('globalThis.observed=0; scheduleGraphNodeUi=()=>{observed+=1}; installXySourceObservers(node,"StyleStackSet")', fixture.context);
  fixture.context.node.widgets[0].callback(-2);
  fixture.context.node.onWidgetChanged("mode", "replace");
  fixture.context.node.onWidgetChanged("unrelated", 1);
  assert.equal(fixture.context.observed, 2);
  const extracted = connectedStackNode("StyleStackExtract", {}, { index: 0 });
  fixture.context.extracted = extracted;
  runInContext('installXySourceObservers(extracted,"StyleStackExtract")', fixture.context);
  extracted.onWidgetChanged("index", 1);
  assert.equal(fixture.context.observed, 3);
});

test("StyleStack metadata handles connected indices and cyclic graphs safely", () => {
  const fixture = runtime();
  const source = artistTextNode("@first");
  const list = connectedStackNode("LoraStackLister", { stack_1: source });
  fixture.context.node = connectedStackNode("StyleStackExtract", { lora_stack_list: list, index: source }, { index: 0 });
  assert.equal(runInContext('stackEntryDataFromSource(node)', fixture.context), null);
  fixture.context.node = connectedStackNode("StyleStackSet", { lora_stack_list: list, lora_stack: source, index: source }, { index: 0, mode: "replace" });
  assert.equal(runInContext('styleStackCountFromSource(node)', fixture.context), null);
  const cycle = connectedStackNode("LoraStackListName");
  cycle.inputs = [{ name: "lora_stack_list", link: 1 }];
  cycle.getInputNode = () => cycle;
  fixture.context.node = connectedStackNode("StyleStackExtract", { lora_stack_list: cycle }, { index: -1 });
  assert.equal(runInContext('stackEntryDataFromSource(node)', fixture.context), null);
  assert.deepEqual(Array.from(runInContext('stackArtistCountsFromNode(node)', fixture.context)), []);
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
