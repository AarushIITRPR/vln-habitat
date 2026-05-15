# Task 2: Habitat and VLN-CE Setup

This folder documents the environment and simulator setup work. Task 2 established that Habitat-Lab and VLN-CE can run on this machine using the Intel/Mesa OpenGL path.

## Purpose

Task 2 covered:

- setting up Habitat-Lab and Habitat-Baselines;
- cloning and preparing VLN-CE;
- verifying that Matterport3D-style scene paths are available to Habitat;
- confirming that `Sim-v0` and `VLN-v0` can initialize;
- running Intel-safe rollout/viewer paths without CUDA or NVIDIA assumptions.

## Folder Contents

```text
code_and_scripts/
├── intel_safe_training_launcher.sh
└── intel_safe_video_eval_launcher.sh

outputs/
└── simulator_videos
```

`code_and_scripts/` links to the launch scripts used for Intel-safe CMA training/evaluation.

`outputs/simulator_videos` links to the simulator video output directory in the live VLN-CE data folder.

## Machine-Specific Rules

This setup is for Ubuntu Linux with X11 and Intel integrated graphics through Mesa iris. OpenGL works, but CUDA/NVIDIA is not available.

For Habitat-Lab config overrides on this machine:

```text
habitat.simulator.habitat_sim_v0.gpu_device_id = -1
habitat.simulator.concur_render = False
```

Use the existing Intel-safe scripts and OpenGL path. Do not switch to CUDA, `gpu_device_id=0`, or a windowless EGL-only path.

## Related Runtime Folders

The actual runtime folders are linked from:

```text
../90_Runtime_Dependencies/
```

That folder points to the live `VLN-CE` repository and to the bundled utility dependencies under `utility_dependencies/`.
