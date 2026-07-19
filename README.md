# MH Custom Nodes

A personal collection of ComfyUI custom nodes (masking, pose, cropping, and a
client-side batch-experiment runner).

## Install

Clone (or copy) this repository into your ComfyUI `custom_nodes` directory and
restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> mh-custom-nodes
pip install -r mh-custom-nodes/requirements.txt   # torch / numpy / Pillow
```

Then restart ComfyUI. The nodes appear under their respective categories (the
batch runner is under **MH/Experiment**).

## Batch Experiment Runner

Sweep parameter combinations across a workflow without manually re-queuing for each
value. ComfyUI runs a graph once per queue call, so the sweep is generated in the
browser: one fully-resolved prompt is queued per combination.

### Usage

1. Add an **MH Batch Input Specifier** node.
2. Drag its **value** output onto the input you want to sweep (e.g. drag onto
   KSampler's `steps` — this converts the widget into an input slot).
3. Click **Update node** on the specifier. Its value slot becomes a clone of that
   input's widget (a bounded `INT` field for `steps`, a dropdown for `sampler_name`,
   etc.), so only valid values can be entered.
4. Use **+ value** / **− value** to set how many values to sweep, and fill them in.
5. Repeat for as many parameters as you like — each specifier is one swept axis.
6. Click **Run Experiment Grid** in the sidebar menu. It queues the cartesian product
   of all specifiers' values (e.g. 3 steps × 2 samplers = 6 runs) and logs each
   `prompt_id` to the browser console.

### Notes

- **No specifiers present** → **Run Experiment Grid** just queues the workflow once.
- **Filenames**: if a node has a `filename_prefix` input, `_c{index}` is appended per
  combination so runs don't overwrite each other on disk.
- **Seeds**: nodes with `control_after_generate` = randomize/increment vary their seed
  on every queue call, independently of the sweep.
- **Large grids**: more than 50 combinations prompts a confirmation dialog first.
- Sweep state is stored on the node and restored when a saved workflow is reloaded.

### Files

- `experiment_nodes.py` — the `MH_BatchInputSpecifier` backend node.
- `web/experimentRunner.js` — the frontend extension (dynamic widgets + grid runner).
