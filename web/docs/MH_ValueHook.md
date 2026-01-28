# Value Hook

Creates a named parameter hook for experimental workflows. Allows exposing a value with a custom name for use with the Experiment Hub.

## Parameters

- **value**: Numeric value to expose (-10000.0 to 10000.0)
- **name**: Identifier name for this parameter hook

## Outputs

- **float**: The value as a float
- **int**: The value as an integer (truncated)
- **hook_ref**: Reference object containing the name and value for connecting to `MH_ExperimentHub`

## Usage

Connect the `hook_ref` output to one of the `MH_ExperimentHub` hook inputs to register this parameter for experimentation.