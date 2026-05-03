# Visual-Control Interface

This project now exports a control-oriented per-frame interface during `predict_video.py` runs.

## Task Assumption

- Vehicle: fixed-wing model aircraft
- Motion regime: low-speed, continuous ground roll without stopping
- Control goal:
  1. align with the pre-specified takeoff direction
  2. stay geometrically near the true runway centerline
- Default assumption:
  - the desired takeoff direction is aligned with the true runway centerline

## Output File

For each video run, `predict_video.py` writes:

- `*_control.csv`

alongside the overlay video and the existing geometry/frame CSV.

## Core Fields

- `interface_valid`
  - Whether the visual interface is trustworthy enough for control consumption on this frame.
- `heading_error_deg`
  - Signed heading-like error in image space.
  - Computed from the line joining a near runway-center sample and a farther runway-center sample.
  - Interpretation:
    - positive: runway centerline trends to the right in the forward direction
    - negative: runway centerline trends to the left in the forward direction
- `lateral_error_px`
  - Signed lateral offset between the image center and the runway center sampled near the control row.
  - Interpretation:
    - positive: runway center lies to the right of the image center
    - negative: runway center lies to the left of the image center
- `lateral_error_norm`
  - `lateral_error_px` normalized by half image width.
- `lateral_error_runway_half_width`
  - `lateral_error_px` normalized by half of the locally observed runway width.
- `confidence`
  - Aggregate confidence score in `[0, 1]`, based on support points, visible runway width, runway ratio, and far-row observability.

## Supporting Geometry Fields

- `centerline_x_at_control_row`
  - Local runway center sampled near the control row, not a far extrapolated global-line value.
- `near_center_x`
  - Same as the local near-row runway center used for lateral error.
- `far_center_x`
  - Farther runway center used jointly with the near center to estimate heading error.
- `control_row_y`
  - The near row used for lateral control extraction.
- `far_row_y`
  - The farther row used for heading extraction.
- `runway_width_px`
  - Observed runway width at the control row.
- `support_points`
- `reliable_support_points`
- `auxiliary_support_points`
- `runway_ratio`

## Intended Control Use

- `heading_error_deg`:
  - feed direction-alignment logic
- `lateral_error_*`:
  - feed centerline-centering logic
- `confidence` and `interface_valid`:
  - gate controller engagement, gain scheduling, fallback logic, or hold-last-value logic

## Notes

- This interface is image-space and vision-derived.
- It is intended as the bridge between segmentation/geometry and control design.
- Final sign conventions should be cross-checked against the aircraft body-axis definition and the actual steering/rudder command convention before closing the loop.
