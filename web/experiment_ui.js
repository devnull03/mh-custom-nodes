import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
	name: "MH.ExperimentHub",

	async setup() {
		// Add button to the menu bar
		const menu = document.querySelector(".comfy-menu");
		if (menu) {
			const separator = document.createElement("hr");
			separator.style.margin = "20px 0";
			separator.style.width = "100%";
			menu.append(separator);

			const runButton = document.createElement("button");
			runButton.textContent = "Run Experiments";
			runButton.style.width = "100%";
			runButton.onclick = () => runExperiments();
			menu.append(runButton);
		}
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "MH_ExperimentHub") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;

			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}

				// Add "Update Values" button widget
				this.addWidget("button", "Update Values", null, () => {
					this.updateHookValues();
				});
			};

			nodeType.prototype.updateHookValues = function () {
				const graph = app.graph;
				const numRuns =
					this.widgets.find((w) => w.name === "num_runs")?.value || 1;

				if (!this.inputs || this.inputs.length === 0) {
					alert("No hooks connected!");
					return;
				}

				for (const input of this.inputs) {
					if (input.link) {
						const link = graph.links[input.link];
						if (!link) continue;

						const sourceNode = graph.getNodeById(link.origin_id);
						if (!sourceNode || sourceNode.type !== "MH_ValueHook")
							continue;

						const nameWidget = sourceNode.widgets.find(
							(w) => w.name === "name",
						);
						const label = nameWidget?.value || "param";

						const rangeStr = prompt(
							`Values for '${label}'? (${numRuns} runs)\n\n` +
								`Formats:\n` +
								`  - CSV: 7, 8, 9\n` +
								`  - Range: start:end:step (e.g., 0.5:1.5:0.1)\n` +
								`  - Single: 7.5`,
						);

						if (rangeStr === null) return;

						const values = parseRange(rangeStr);
						if (values.length === 0) {
							alert(`Invalid format for '${label}'`);
							return;
						}

						// Store values on the source node for later use
						sourceNode._experimentValues = values;
						console.log(
							`[MH] ${label}: ${values.length} values stored`,
						);
					}
				}

				alert(
					"Values updated! Click 'Run Experiments' in the menu to start.",
				);
			};
		}
		if (nodeData.name === "mh_MaskMinimalCrop") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			const onWidgetChanged = nodeType.prototype.onWidgetChanged;

			function updateResolutionVisibility(node) {
				const scaleWidget = node.widgets?.find(
					(w) => w.name === "scale_mode",
				);
				const resolutionWidget = node.widgets?.find(
					(w) => w.name === "resolution",
				);
				if (!scaleWidget || !resolutionWidget) return;
				const show = scaleWidget.value === "custom";
				resolutionWidget.hidden = !show;
				if (resolutionWidget.inputEl) {
					resolutionWidget.inputEl.style.display = show ? "" : "none";
				}
				if (node.setSize && node.computeSize) {
					node.setSize(node.computeSize());
				}
			}

			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}
				updateResolutionVisibility(this);
			};

			nodeType.prototype.onWidgetChanged = function (
				name,
				value,
				widget,
			) {
				if (onWidgetChanged) {
					onWidgetChanged.apply(this, arguments);
				}
				if (name === "scale_mode") {
					updateResolutionVisibility(this);
				}
			};
		}
	},
});

async function runExperiments() {
	const graph = app.graph;

	// Find the ExperimentHub node
	const hubNode = graph.findNodesByType("MH_ExperimentHub")[0];
	if (!hubNode) {
		alert("No MH Experiment Hub node found in the workflow!");
		return;
	}

	const numRuns =
		hubNode.widgets.find((w) => w.name === "num_runs")?.value || 1;

	// Collect all connected hooks with their values
	const hooks = [];
	if (hubNode.inputs) {
		for (const input of hubNode.inputs) {
			if (input.link) {
				const link = graph.links[input.link];
				if (!link) continue;

				const sourceNode = graph.getNodeById(link.origin_id);
				if (!sourceNode || sourceNode.type !== "MH_ValueHook") continue;

				const nameWidget = sourceNode.widgets.find(
					(w) => w.name === "name",
				);
				const values = sourceNode._experimentValues;

				if (!values || values.length === 0) {
					alert(
						`No values set for '${nameWidget?.value || "param"}'. Click 'Update Values' first.`,
					);
					return;
				}

				hooks.push({
					nodeId: sourceNode.id,
					label: nameWidget?.value || "param",
					values: values,
				});
			}
		}
	}

	if (hooks.length === 0) {
		alert(
			"No hooks with values found! Connect ValueHook nodes and click 'Update Values'.",
		);
		return;
	}

	// Generate combinations
	const combinations = cartesian(hooks.map((h) => h.values));
	const totalRuns = Math.min(combinations.length, numRuns);

	const confirmRun = confirm(
		`Experiment Setup:\n\n` +
			`Parameters:\n${hooks.map((h) => `  - ${h.label}: ${h.values.length} values`).join("\n")}\n\n` +
			`Total combinations: ${combinations.length}\n` +
			`Runs to queue: ${totalRuns}\n\n` +
			`Proceed?`,
	);

	if (!confirmRun) return;

	try {
		const apiGraph = await app.graphToPrompt();

		for (let i = 0; i < totalRuns; i++) {
			const combo = combinations[i];
			const promptPayload = JSON.parse(JSON.stringify(apiGraph.output));

			combo.forEach((val, idx) => {
				const hookInfo = hooks[idx];
				const hookNodeData = promptPayload[hookInfo.nodeId];

				if (hookNodeData && hookNodeData.inputs) {
					hookNodeData.inputs.value = val;
				}
			});

			await api.queuePrompt(0, { output: promptPayload });
			console.log(`[MH] Queued run ${i + 1}/${totalRuns}`);
		}

		alert(`Queued ${totalRuns} experiment runs!`);
	} catch (error) {
		console.error("[MH] Error:", error);
		alert(`Error: ${error.message}`);
	}
}

function parseRange(str) {
	if (!str || str.trim() === "") return [];

	str = str.trim();

	if (str.includes(":")) {
		const parts = str.split(":").map((s) => parseFloat(s.trim()));
		if (parts.length < 2 || parts.some(isNaN)) return [];

		const start = parts[0];
		const end = parts[1];
		const step = parts.length >= 3 ? parts[2] : 1;

		if (
			step === 0 ||
			(step > 0 && start > end) ||
			(step < 0 && start < end)
		) {
			return [start];
		}

		const result = [];
		const precision = Math.max(
			getDecimalPlaces(start),
			getDecimalPlaces(end),
			getDecimalPlaces(step),
		);

		if (step > 0) {
			for (let i = start; i <= end + 0.0001; i += step) {
				result.push(Number(i.toFixed(precision)));
			}
		} else {
			for (let i = start; i >= end - 0.0001; i += step) {
				result.push(Number(i.toFixed(precision)));
			}
		}

		return result;
	}

	return str
		.split(",")
		.map((s) => {
			const trimmed = s.trim();
			const num = parseFloat(trimmed);
			return isNaN(num) ? trimmed : num;
		})
		.filter((v) => v !== "" && v !== null);
}

function getDecimalPlaces(num) {
	const str = String(num);
	const idx = str.indexOf(".");
	return idx === -1 ? 0 : str.length - idx - 1;
}

function cartesian(arrays) {
	if (arrays.length === 0) return [[]];
	return arrays.reduce(
		(acc, arr) =>
			acc.flatMap((combo) => arr.map((item) => [...combo, item])),
		[[]],
	);
}
