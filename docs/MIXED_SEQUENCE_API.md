# External mixed-sequence API 2

Capability 2 supports Flow's `mixed_grid_low_suffix` topology. The existing
`vdn_h3_external_sequence_v1` key remains in use; consumers set `api=2` and
`mode=dense_gate_no_linear`. Existing reduced-sequence API 1 remains accepted.

API 2 explicitly permits a stream larger than the published low-carrier
layout. Required integer fields are `native_sequence_rows`, `sequence_rows`,
`video_start`, `temporal`, `prefix_t`, `source_rows_per_frame`, and
`prefix_rows_per_frame`. They must satisfy the native layout and:

`sequence_rows = video_start + prefix_t * prefix_rows_per_frame + (temporal - prefix_t) * source_rows_per_frame`.

The protected prefix must be nonempty and shorter than the chunk; its frame
density must exceed the source frame density. Matching explicit RoPE is
mandatory. Dense attention and the learned softmax gate operate on the mixed
rows; local windows, short convolution, and the linear complement remain off.
No external contract is permitted on a normal full-native call.

This is an explicit compatibility mode, not a trained mixed-grid VDN model.
GPU quality, compiler behavior, peak VRAM, and end-to-end performance remain
unvalidated. Preserve all existing PR #2 runtime acceptance requirements.
