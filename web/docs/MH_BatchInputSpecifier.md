# Batch Input Specifier

Marks a single workflow parameter for a client-side batch sweep. Wire this node's
output into any widget-input (e.g. KSampler `steps`, `sampler_name`, `cfg`), click
**Update node**, and it clones that input's widget so each swept value uses the exact
same control — a dropdown of valid combo options, a bounded number field, or a text
box. The **Run Experiment Grid** button (sidebar menu) then queues one run per
combination across every specifier in the graph.

## Node controls

- **Update node**: reads the connected downstream input's widget definition and
  (re)builds the value slots to match. Click again after re-wiring to a different input.
- **+ value** / **− value**: add or remove a value slot (minimum one).
- **value_1, value_2, …**: the swept values — each is a clone of the target widget, so
  invalid entries are impossible.

## How it runs

- **Run Experiment Grid** scans all `MH_BatchInputSpecifier` nodes, takes the cartesian
  product of their value lists, and for each combination clones the serialized prompt and
  overwrites the targeted downstream input with that combination's value before queuing.
- If a `filename_prefix` input exists anywhere in the graph, `_c{index}` is appended per
  combination so outputs don't overwrite on disk.
- The returned `prompt_id` for each run is logged to the browser console.

## Outputs

- **value**: wildcard (`*`) output. For a plain single **Queue Prompt** (outside the grid
  runner) it emits the first value, coerced to the cloned type.

## Notes

- With no specifier nodes present, **Run Experiment Grid** just queues the workflow once.
- Seeds are independent of the sweep: any node with `control_after_generate` set to
  randomize/increment still advances its seed on every queue call.
- All state lives in a hidden `payload` widget, so saved workflows restore their value
  slots on reload without needing to re-connect and re-click **Update node**.
- If more than 50 combinations would be queued, a confirmation dialog appears first.
