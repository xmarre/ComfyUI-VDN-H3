export const LEGACY_ADVANCED_WIDGET_NAMES = Object.freeze([
  "vdn_checkpoint",
  "apply_turbo_adapter",
  "stage_b_strength",
  "turbo_strength",
  "lora_mode",
  "branch_weights",
  "verbose",
  "attention_backend",
  "window_radius",
  "window_chunk",
  "anchor_frames",
  "text_state",
  "linear_branch",
  "fast_kernels",
]);

export const CURRENT_ADVANCED_WIDGET_NAMES = Object.freeze([
  "vdn_checkpoint",
  "apply_turbo_adapter",
  "stage_b_strength",
  "turbo_strength",
  "lora_mode",
  "branch_weights",
  "retain_buffers",
  "verbose",
  "attention_backend",
  "architecture_mode",
  "window_radius",
  "window_chunk",
  "anchor_frames",
  "text_state",
  "linear_branch",
  "fast_kernels",
]);

function namedValues(names, values) {
  const result = {};
  for (let i = 0; i < names.length; i += 1) result[names[i]] = values[i];
  return result;
}

/**
 * Repair positional ApplyVDNH3Advanced workflows saved before
 * widgets_values_named existed.
 *
 * The 14-value legacy layout predates retain_buffers and architecture_mode.
 * Those workflows used the architecture fields unconditionally, so behavioral
 * compatibility requires architecture_mode=override rather than the new-node
 * default checkpoint mode.
 *
 * A short-lived 16-value positional layout is also accepted. It already has the
 * current semantics, so values are simply named without changing defaults.
 */
export function migrateLegacyAdvancedWidgetValues(info) {
  if (!info || info.widgets_values_named || !Array.isArray(info.widgets_values)) return null;

  if (info.widgets_values.length === LEGACY_ADVANCED_WIDGET_NAMES.length) {
    info.widgets_values_named = namedValues(
      LEGACY_ADVANCED_WIDGET_NAMES,
      info.widgets_values,
    );
    info.widgets_values_named.retain_buffers = "auto";
    info.widgets_values_named.architecture_mode = "override";
    return "legacy-14";
  }

  if (info.widgets_values.length === CURRENT_ADVANCED_WIDGET_NAMES.length) {
    info.widgets_values_named = namedValues(
      CURRENT_ADVANCED_WIDGET_NAMES,
      info.widgets_values,
    );
    return "current-16";
  }

  return null;
}
