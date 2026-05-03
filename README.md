# PX4 Runway Align Baseline

This repository is a project-focused snapshot for the fixed-wing runway visual alignment work.

It contains the ROS/PX4/Gazebo integration files needed for the runway alignment baseline:

- `scripts/runway_*.py`: vision, controller, ground motion assist, RC bridge, and monitor nodes
- `launch/runway_*.launch`: visual alignment launch entry points
- `fixed_wing_vm_bundle_20260416/`: runway segmentation inference and control-interface code
- `runway_gazebo_assets/`: copied Gazebo plane and grass runway assets used by this project
- `summary/`: project progress and tuning notes

The full upstream PX4 history and unrelated PX4 source tree are intentionally not included in this snapshot.
