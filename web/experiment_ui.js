import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Experiment Controller UI Extension
 * Adds custom UI for managing experiment parameter sweeps in ComfyUI.
 */

app.registerExtension({
	name: "Comfy.ExperimentController",

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "MH_ExperimentController") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;

			nodeType.prototype.onNodeCreated = function () {
				if (onNodeCreated) {
					onNodeCreated.apply(this, arguments);
				}

				this.addWidget("button", "RUN EXPERIMENTS", null, () => {
					this.runExperiments();
				});

				this.statusWidget = this.addWidget(
					"text",
					"Status",
					"Ready",
					() => {},
					{
						multiline: false,
						disabled: true,
					},
				);

				this.setSize([280, 150]);
			};

			nodeType.prototype.runExperiments = async function () {
				const graph = app.graph;
				const connectedHooks = [];

				if (!this.inputs || this.inputs.length === 0) {
					alert(
						"No hooks connected! Connect ValueHook nodes to the inputs.",
					);
					return;
				}

				for (const input of this.inputs) {
					if (input.link) {
						const link = graph.links[input.link];
						if (!link) continue;

						const sourceNode = graph.getNodeById(link.origin_id);
						if (!sourceNode) continue;

						if (sourceNode.type === "MH_ValueHook") {
							let label = "param";
							let currentValue = 1.0;

							if (sourceNode.widgets) {
								for (const widget of sourceNode.widgets) {
									if (widget.name === "name") {
										label = widget.value;
									}
									if (widget.name === "value") {
										currentValue = widget.value;
									}
								}
							}

							const rangeStr = prompt(
								`Values for '${label}'?\n\n` +
									`Current value: ${currentValue}\n\n` +
									`Formats:\n` +
									`  - CSV: 7, 8, 9\n` +
									`  - Range: start:end:step (e.g., 0.5:1.5:0.1)\n` +
									`  - Single: 7.5`,
								String(currentValue),
							);

							if (rangeStr === null) {
								return;
							}

							const values = parseRange(rangeStr);
							if (values.length === 0) {
								alert(
									`Invalid range format for '${label}': ${rangeStr}`,
								);
								return;
							}

							connectedHooks.push({
								nodeId: sourceNode.id,
								label: label,
								values: values,
								inputName: input.name,
							});
						}
					}
				}

				if (connectedHooks.length === 0) {
					alert(
						"No MH Value Hook nodes found among connected inputs!",
					);
					return;
				}

				const combinations = cartesian(
					connectedHooks.map((h) => h.values),
				);
				const expId =
					"Exp_" +
					new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);

				const confirmRun = confirm(
					`Experiment Setup:\n\n` +
						`Parameters:\n${connectedHooks.map((h) => `  - ${h.label}: ${h.values.length} values`).join("\n")}\n\n` +
						`Total combinations: ${combinations.length}\n` +
						`Experiment ID: ${expId}\n\n` +
						`Queue ${combinations.length} runs?`,
				);

				if (!confirmRun) return;

				this.updateStatus(`Queueing 0/${combinations.length}...`);

				try {
					const apiGraph = await app.graphToPrompt();

					for (let i = 0; i < combinations.length; i++) {
						const combo = combinations[i];

						const promptPayload = JSON.parse(
							JSON.stringify(apiGraph.output),
						);
						const paramsRecord = { experiment_id: expId };

						combo.forEach((val, idx) => {
							const hookInfo = connectedHooks[idx];
							const hookNodeData = promptPayload[hookInfo.nodeId];

							if (hookNodeData && hookNodeData.inputs) {
								hookNodeData.inputs.value = val;
							}

							paramsRecord[hookInfo.label] = val;
						});

						await api.queuePrompt(0, {
							output: promptPayload,
							extra_data: {
								extra_pnginfo: {
									experiment_params: paramsRecord,
								},
							},
						});

						this.updateStatus(
							`Queued ${i + 1}/${combinations.length}`,
						);
					}

					this.updateStatus(
						`Done: Queued ${combinations.length} runs`,
					);

					console.log(
						`[ExperimentController] Queued ${combinations.length} experiment runs with ID: ${expId}`,
					);
				} catch (error) {
					console.error(
						"[ExperimentController] Error queueing experiments:",
						error,
					);
					this.updateStatus(`Error: ${error.message}`);
					alert(`Error queueing experiments: ${error.message}`);
				}
			};

			nodeType.prototype.updateStatus = function (text) {
				if (this.statusWidget) {
					this.statusWidget.value = text;
				}
				this.setDirtyCanvas(true, true);
			};
		}
	},
});

/**
 * Parse a range string into an array of values.
 * Supports:
 *   - CSV: "1, 2, 3" or "1,2,3"
 *   - Range: "start:end:step" e.g., "0.1:1.0:0.1"
 *   - Single value: "7.5"
 */
function parseRange(str) {
	if (!str || str.trim() === "") {
		return [];
	}

	str = str.trim();

	if (str.includes(":")) {
		const parts = str.split(":").map((s) => parseFloat(s.trim()));

		if (parts.length < 2 || parts.some(isNaN)) {
			return [];
		}

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

/**
 * Get the number of decimal places in a number.
 */
function getDecimalPlaces(num) {
	const str = String(num);
	const decimalIndex = str.indexOf(".");
	if (decimalIndex === -1) return 0;
	return str.length - decimalIndex - 1;
}

/**
 * Generate the Cartesian product of multiple arrays.
 * Example: cartesian([[1,2], [a,b]]) => [[1,a], [1,b], [2,a], [2,b]]
 */
function cartesian(arrays) {
	if (arrays.length === 0) return [[]];

	return arrays.reduce(
		(acc, arr) =>
			acc.flatMap((combo) => arr.map((item) => [...combo, item])),
		[[]],
	);
}
