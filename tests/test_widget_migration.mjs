import assert from "node:assert/strict";
import test from "node:test";

import {
  CURRENT_ADVANCED_WIDGET_NAMES,
  LEGACY_ADVANCED_WIDGET_NAMES,
  migrateLegacyAdvancedWidgetValues,
} from "../web/widget_migration_core.mjs";

const legacyValues = [
  "stage-dmd-step-250",
  true,
  0.8,
  0.9,
  "bypass",
  "stream",
  false,
  "flex",
  2,
  7,
  "columns",
  false,
  true,
  false,
];

test("legacy 14-value Advanced workflow restores by original widget name", () => {
  const info = { widgets_values: [...legacyValues] };
  assert.equal(migrateLegacyAdvancedWidgetValues(info), "legacy-14");
  assert.deepEqual(LEGACY_ADVANCED_WIDGET_NAMES, [
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

  const named = info.widgets_values_named;
  assert.equal(named.vdn_checkpoint, "stage-dmd-step-250");
  assert.equal(named.apply_turbo_adapter, true);
  assert.equal(named.stage_b_strength, 0.8);
  assert.equal(named.turbo_strength, 0.9);
  assert.equal(named.lora_mode, "bypass");
  assert.equal(named.branch_weights, "stream");
  assert.equal(named.verbose, false);
  assert.equal(named.attention_backend, "flex");
  assert.equal(named.window_radius, 2);
  assert.equal(named.window_chunk, 7);
  assert.equal(named.anchor_frames, "columns");
  assert.equal(named.text_state, false);
  assert.equal(named.linear_branch, true);
  assert.equal(named.fast_kernels, false);

  // New fields get compatibility semantics, not new-node semantics.
  assert.equal(named.retain_buffers, "auto");
  assert.equal(named.architecture_mode, "override");

  // The next save has a complete named representation, protecting it from
  // subsequent widget order changes.
  assert.equal(Object.keys(named).length, 16);
  assert.doesNotThrow(() => JSON.stringify({ ...info, widgets_values_named: named }));
});

test("current 16-value positional workflow is named without changing semantics", () => {
  const values = [
    "stage-b-step-2000",
    false,
    1,
    0,
    "merge",
    "resident",
    "on",
    true,
    "grouped",
    "checkpoint",
    1,
    5,
    "both",
    true,
    true,
    true,
  ];
  const info = { widgets_values: values };
  assert.equal(migrateLegacyAdvancedWidgetValues(info), "current-16");
  assert.deepEqual(
    info.widgets_values_named,
    Object.fromEntries(CURRENT_ADVANCED_WIDGET_NAMES.map((name, i) => [name, values[i]])),
  );
  assert.equal(info.widgets_values_named.architecture_mode, "checkpoint");
  assert.equal(info.widgets_values_named.retain_buffers, "on");
});

test("named workflows and unknown positional layouts are left untouched", () => {
  const named = { widgets_values: legacyValues, widgets_values_named: { window_radius: 3 } };
  assert.equal(migrateLegacyAdvancedWidgetValues(named), null);
  assert.deepEqual(named.widgets_values_named, { window_radius: 3 });

  const unknown = { widgets_values: [1, 2, 3] };
  assert.equal(migrateLegacyAdvancedWidgetValues(unknown), null);
  assert.equal(unknown.widgets_values_named, undefined);
});
