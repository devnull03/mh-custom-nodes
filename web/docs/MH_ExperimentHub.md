# Experiment Hub

Central hub for connecting multiple Value Hooks in experimental workflows. Allows orchestrating parameter sweeps across multiple runs.

## Parameters

- **num_runs**: Number of experiment iterations to run (1-1000)
- **hook_1** to **hook_8**: Optional hook reference inputs from `MH_ValueHook` nodes

## Outputs

- **run_index**: Current run iteration index (starting from 0)

## Usage

1. Create `MH_ValueHook` nodes for each parameter you want to experiment with
2. Connect their `hook_ref` outputs to this node's hook inputs
3. Set `num_runs` to the desired number of iterations
4. Use the `run_index` output to vary behavior across runs