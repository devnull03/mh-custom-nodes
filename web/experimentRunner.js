import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// MH Batch Input Specifier — client-side parameter sweep for ComfyUI.
//
// ComfyUI runs a graph exactly once per /prompt call (no native looping), so we
// build the sweep here: scan the graph for MH_BatchInputSpecifier nodes, take the
// cartesian product of their value lists, and queue one fully-resolved prompt per
// combination.
//
// The specifier node clones the widget of whatever input it is wired into, so the
// value slots are the *same* widget the real input uses (a dropdown of valid combo
// options, a bounded INT/FLOAT field, ...). That makes every swept value valid by
// construction — the runner never has to guess or re-validate types.
//
// All persistent node state lives in the single serialized `payload` widget as
// {"type": "...", "values": [...]}. The dynamic value/button widgets are marked
// serialize:false and rebuilt from `payload` on load, which sidesteps ComfyUI's
// fragile positional widget-value restore for dynamically-added widgets.
// ---------------------------------------------------------------------------

const NODE_TYPE = "MH_BatchInputSpecifier";
const LARGE_RUN_THRESHOLD = 50;

// ------------------------------- small utils -------------------------------

function notify(message, isError = false) {
	// Prefer the ComfyUI toast system when present, fall back to alert.
	const toast = app?.extensionManager?.toast;
	if (toast?.add) {
		toast.add({
			severity: isError ? "error" : "info",
			summary: isError ? "Batch Experiment" : "Batch Experiment",
			detail: message,
			life: 4000,
		});
	} else {
		alert(message);
	}
}

// Generic cartesian product of N arrays.
//   cartesianProduct([[1,2],[3,4]]) -> [[1,3],[1,4],[2,3],[2,4]]
// Reduce builds combinations left-to-right; each step appends one value from the
// next array to every combination accumulated so far. [[]] is the identity so an
// empty input yields a single empty combo.
function cartesianProduct(arrays) {
	return arrays.reduce(
		(acc, arr) => acc.flatMap((combo) => arr.map((v) => [...combo, v])),
		[[]],
	);
}

// Visually collapse a canvas-drawn widget without removing it (so it still
// serializes). Used to hide the `payload` widget.
function hideWidget(widget) {
	widget.type = "hidden";
	widget.computeSize = () => [0, -4];
}

// ------------------------ downstream link resolution ------------------------

// Follow a specifier's output link to the node + input it feeds.
// Returns { targetNode, inputName, targetInput, downstreamId } or null.
//
// Graph link data: node.outputs[0].links is an array of link IDs; each ID indexes
// app.graph.links, whose entries carry { origin_id, origin_slot, target_id,
// target_slot, type }. We take the first link — a specifier is meant to drive one
// input.
function resolveDownstream(node) {
	const out = node.outputs && node.outputs[0];
	if (!out || !out.links || out.links.length === 0) return null;

	const link = app.graph.links[out.links[0]];
	if (!link) return null;

	const targetNode = app.graph.getNodeById(link.target_id);
	if (!targetNode || !targetNode.inputs) return null;

	const targetInput = targetNode.inputs[link.target_slot];
	if (!targetInput) return null;

	return {
		targetNode,
		targetInput,
		inputName: targetInput.name,
		// Prompt (graphToPrompt output) keys node IDs as strings.
		downstreamId: String(link.target_id),
	};
}

// ------------------------- widget-spec extraction --------------------------

// Turn a ComfyUI INPUT_TYPES config array into a normalized spec.
//   ["INT",   {default,min,max,step}]        -> { type:"INT",   options:{...} }
//   ["FLOAT", {default,min,max,step}]        -> { type:"FLOAT", options:{...} }
//   ["STRING",{default}]                     -> { type:"STRING",options:{...} }
//   ["BOOLEAN",{default}]                    -> COMBO over [true,false]
//   [["euler","dpmpp_2m",...], {default}]    -> { type:"COMBO", options:{values} }
function specFromConfig(config) {
	if (!Array.isArray(config) || config.length === 0) return null;
	const head = config[0];
	const opts = config[1] || {};

	// Combo inputs declare their options as an array in the first slot.
	if (Array.isArray(head)) {
		return { type: "COMBO", options: { values: head, default: opts.default } };
	}
	if (head === "INT" || head === "FLOAT") {
		return {
			type: head,
			options: {
				min: opts.min,
				max: opts.max,
				step: opts.step,
				default: opts.default,
			},
		};
	}
	if (head === "BOOLEAN") {
		return {
			type: "COMBO",
			options: { values: [true, false], default: opts.default },
		};
	}
	// STRING and anything else -> free text.
	return { type: "STRING", options: { default: opts.default } };
}

// Resolve the widget spec of the connected downstream input. ComfyUI's frontend
// has shifted where this info lives across versions, so we try several sources in
// order and fall back to plain text if none work. This is the fiddliest part of
// the extension — keep the fallbacks.
function resolveTargetSpec(targetNode, inputName, targetInput) {
	// 1) A widget that was converted into an input keeps its config on the slot.
	const converted = targetInput?.widget?.config;
	if (converted) {
		const spec = specFromConfig(converted);
		if (spec) return spec;
	}

	// 2) The input is still a live widget on the downstream node (not converted).
	//    Read the widget's own type/options directly.
	const live = targetNode.widgets?.find((w) => w.name === inputName);
	if (live) {
		if (live.type === "combo" && live.options?.values) {
			return { type: "COMBO", options: { values: live.options.values, default: live.value } };
		}
		if (live.type === "number") {
			// precision 0 (or an integer step) => treat as INT.
			const isInt = live.options?.precision === 0 ||
				(Number.isInteger(live.options?.step) && Number.isInteger(live.value));
			return {
				type: isInt ? "INT" : "FLOAT",
				options: {
					min: live.options?.min,
					max: live.options?.max,
					step: live.options?.step,
					default: live.value,
				},
			};
		}
		if (live.type === "toggle") {
			return { type: "COMBO", options: { values: [true, false], default: live.value } };
		}
		return { type: "STRING", options: { default: live.value } };
	}

	// 3) The registered node definition's INPUT_TYPES.
	const def =
		targetNode.constructor?.nodeData ||
		window.LiteGraph?.registered_node_types?.[targetNode.type]?.nodeData;
	const inputs = def?.input;
	if (inputs) {
		const config =
			inputs.required?.[inputName] || inputs.optional?.[inputName];
		const spec = specFromConfig(config);
		if (spec) return spec;
	}

	return null;
}

// ---------------------- dynamic value-widget management ----------------------

function findPayloadWidget(node) {
	return node.widgets?.find((w) => w.name === "payload");
}

function readPayload(node) {
	const w = findPayloadWidget(node);
	try {
		const data = JSON.parse(w?.value || "{}");
		return {
			type: data.type || "STRING",
			options: data.options || {},
			values: data.values || [],
		};
	} catch (e) {
		return { type: "STRING", options: {}, values: [] };
	}
}

// Serialize the full spec (type + options, so combo option lists and numeric
// bounds survive reload) plus the current value-widget values into the hidden
// `payload` widget. Called on every value/count change so the saved workflow
// stays in sync. Python only reads type+values; extra keys are ignored there.
function writePayload(node) {
	const w = findPayloadWidget(node);
	if (!w) return;
	const type = node._batchSpec?.type || "STRING";
	const options = node._batchSpec?.options || {};
	const values = (node._valueWidgets || []).map((vw) => vw.value);
	w.value = JSON.stringify({ type, options, values });
}

function defaultValueFor(spec) {
	if (!spec) return "";
	if (spec.type === "COMBO") {
		return spec.options?.default ?? spec.options?.values?.[0] ?? "";
	}
	if (spec.type === "INT" || spec.type === "FLOAT") {
		return spec.options?.default ?? spec.options?.min ?? 0;
	}
	return spec.options?.default ?? "";
}

// Create one value-slot widget cloned from the resolved spec. Marked
// serialize:false — its state is persisted via `payload`, not positionally.
function createValueWidget(node, spec, value, index) {
	const name = `value_${index + 1}`;
	const onChange = () => writePayload(node);
	let widget;

	if (spec.type === "COMBO") {
		widget = node.addWidget("combo", name, value, onChange, {
			values: spec.options.values,
		});
	} else if (spec.type === "INT" || spec.type === "FLOAT") {
		widget = node.addWidget("number", name, value, onChange, {
			min: spec.options.min,
			max: spec.options.max,
			step: spec.options.step,
			precision: spec.type === "INT" ? 0 : undefined,
		});
	} else {
		widget = node.addWidget("text", name, value, onChange, {});
	}

	widget.serialize = false;
	if (widget.options) widget.options.serialize = false;
	return widget;
}

// Remove all existing value-slot widgets, then create fresh ones for `values`.
function rebuildValueWidgets(node, spec, values) {
	// Drop previously-created value widgets from the node's widget list.
	if (node._valueWidgets?.length) {
		node.widgets = node.widgets.filter(
			(w) => !node._valueWidgets.includes(w),
		);
	}
	node._valueWidgets = [];
	node._batchSpec = spec;

	const list = values.length ? values : [defaultValueFor(spec)];
	list.forEach((val, i) => {
		node._valueWidgets.push(createValueWidget(node, spec, val, i));
	});

	writePayload(node);
	node.setSize(node.computeSize());
	app.graph.setDirtyCanvas(true, true);
}

// "Update node": (re)clone the connected input's widget.
function updateFromConnection(node) {
	const down = resolveDownstream(node);
	if (!down) {
		notify(
			"Connect this node's output to an input first, then click Update node.",
			true,
		);
		return;
	}

	const spec = resolveTargetSpec(down.targetNode, down.inputName, down.targetInput);
	const existing = readPayload(node).values;

	if (!spec) {
		notify(
			`Could not read the widget type of '${down.inputName}'. Falling back to text input.`,
			true,
		);
		rebuildValueWidgets(node, { type: "STRING", options: {} }, existing);
		return;
	}

	// Preserve existing values only if the type is unchanged; otherwise start
	// fresh with one default slot (old values may be invalid for the new widget).
	const prevType = node._batchSpec?.type;
	const keep = prevType === spec.type ? existing : [];
	rebuildValueWidgets(node, spec, keep);
	node.title = `Batch: ${down.inputName}`;
	notify(`Cloned '${down.inputName}' (${spec.type}).`);
}

function addValueSlot(node) {
	const spec = node._batchSpec || { type: "STRING", options: {} };
	const values = node._valueWidgets.map((w) => w.value);
	values.push(defaultValueFor(spec));
	rebuildValueWidgets(node, spec, values);
}

function removeValueSlot(node) {
	if (!node._valueWidgets || node._valueWidgets.length <= 1) return; // keep >=1
	const spec = node._batchSpec || { type: "STRING", options: {} };
	const values = node._valueWidgets.map((w) => w.value);
	values.pop();
	rebuildValueWidgets(node, spec, values);
}

// --------------------------------- runner ----------------------------------

async function queueSingle() {
	const p = await app.graphToPrompt();
	await api.queuePrompt(0, { output: p.output, workflow: p.workflow });
}

async function runExperimentGrid() {
	// 1. Find all specifier nodes currently in the graph.
	const specifiers = app.graph._nodes.filter((n) => n.type === NODE_TYPE);

	if (specifiers.length === 0) {
		notify("No batch inputs found, queuing single run.");
		try {
			await queueSingle();
		} catch (e) {
			notify(`Queue failed: ${e.message}`, true);
		}
		return;
	}

	// 2/3. Read each specifier's values and resolve its downstream target.
	const specs = [];
	for (const node of specifiers) {
		const { values } = readPayload(node);
		if (!values || values.length === 0) {
			notify(`'${node.title}' has no values. Set some, then re-run.`, true);
			return; // abort — queue nothing on a half-configured sweep
		}
		const down = resolveDownstream(node);
		if (!down) {
			notify(
				`'${node.title}' is not connected to any input. Wire its output, then re-run.`,
				true,
			);
			return;
		}
		// Values came from cloned widgets, so they are already valid & correctly
		// typed (numbers stay numbers, combos stay allowed strings) — no
		// re-validation needed here.
		specs.push({
			downstreamId: down.downstreamId,
			inputName: down.inputName,
			values,
		});
	}

	// 4. Cartesian product across every specifier.
	const combos = cartesianProduct(specs.map((s) => s.values));
	if (combos.length > LARGE_RUN_THRESHOLD) {
		const ok = confirm(
			`This will queue ${combos.length} runs (> ${LARGE_RUN_THRESHOLD}). Proceed?`,
		);
		if (!ok) return;
	}

	// 5. Serialize the base workflow once; clone + patch it per combination.
	const base = await app.graphToPrompt();
	const results = [];

	for (let i = 0; i < combos.length; i++) {
		const combo = combos[i];
		const clone = structuredClone(base.output);

		// Overwrite each targeted downstream input with this combo's value. This
		// replaces the link reference with a literal; the specifier node then has
		// no consumers and ComfyUI prunes it from execution.
		combo.forEach((value, idx) => {
			const s = specs[idx];
			const target = clone[s.downstreamId];
			if (target && target.inputs) {
				target.inputs[s.inputName] = value;
			}
		});

		// 6. Optional: disambiguate output filenames so combos don't overwrite each
		//    other on disk. Best-effort — appends _cN to any filename_prefix input.
		for (const nodeId of Object.keys(clone)) {
			const inputs = clone[nodeId].inputs;
			if (inputs && typeof inputs.filename_prefix === "string") {
				inputs.filename_prefix = `${inputs.filename_prefix}_c${i}`;
			}
		}

		// NOTE: seeds are independent of this sweep. Any node with
		// control_after_generate = randomize/increment will still advance its seed
		// on every queue call regardless of these combinations — not fought here.
		try {
			const res = await api.queuePrompt(0, {
				output: clone,
				workflow: base.workflow,
			});
			const label = combo.map((v, idx) => `${specs[idx].inputName}=${v}`).join(", ");
			results.push({ combo: label, prompt_id: res?.prompt_id });
			console.log(`[MH] Queued ${i + 1}/${combos.length}: ${label}`, res?.prompt_id);
		} catch (e) {
			console.error(`[MH] Failed combo ${i + 1}:`, combo, e);
			notify(`Combo ${i + 1} failed: ${e.message}`, true);
		}
	}

	console.table(results);
	notify(`Queued ${results.length}/${combos.length} runs. See console for prompt_ids.`);
}

// ------------------------------- extension ---------------------------------

app.registerExtension({
	name: "MH.BatchExperimentRunner",

	setup() {
		// Add a "Run Experiment Grid" button to the classic sidebar menu.
		const menu = document.querySelector(".comfy-menu");
		if (!menu) return;

		const separator = document.createElement("hr");
		separator.style.margin = "20px 0";
		separator.style.width = "100%";
		menu.append(separator);

		const button = document.createElement("button");
		button.textContent = "Run Experiment Grid";
		button.style.width = "100%";
		button.onclick = () => runExperimentGrid();
		menu.append(button);
	},

	async beforeRegisterNodeDef(nodeType, nodeData) {
		if (nodeData.name !== NODE_TYPE) return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			onNodeCreated?.apply(this, arguments);

			// Hide the JSON state widget.
			const payload = findPayloadWidget(this);
			if (payload) hideWidget(payload);

			// Control buttons.
			this.addWidget("button", "Update node", null, () => updateFromConnection(this));
			this.addWidget("button", "+ value", null, () => addValueSlot(this));
			this.addWidget("button", "− value", null, () => removeValueSlot(this));

			// Initial value slots from whatever payload holds (empty => one text slot).
			const { type, options, values } = readPayload(this);
			this._batchSpec = { type, options };
			rebuildValueWidgets(this, this._batchSpec, values);
		};

		// Restore value slots from `payload` after a saved workflow loads. onConfigure
		// runs after ComfyUI has re-applied widget values, so payload is authoritative.
		const onConfigure = nodeType.prototype.onConfigure;
		nodeType.prototype.onConfigure = function (info) {
			onConfigure?.apply(this, arguments);
			const { type, options, values } = readPayload(this);
			this._batchSpec = { type, options };
			rebuildValueWidgets(this, this._batchSpec, values);
		};
	},
});
