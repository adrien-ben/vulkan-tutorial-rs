# Vulkan tutorial

Vulkan [tutorials][0] written in Rust using [Ash][1].

## Introduction

This repository will follow the structure of the original tutorial. Each 
commit will correspond to one page or on section of the page for 
long chapters.

Sometimes an 'extra' commit will be added with some refactoring or commenting.

## Commits

### 1.1.1: Base code

Application setup. We don't setup the window system now as it's done in 
the original tutorial.

### 1.1.2: Instance

Create and destroy the Vulkan instance with required surface extensions.

### 1.1.3: Validation layers

Add `VK_LAYER_LUNARG_standard_validation` at instance creation and creates
a debug report callback function after checking that it is available. 
Since we are using the `log` crate, we log the message with the proper log level.
The callback is detroyed at application termination.

### 1.1.4: Physical devices and queue families

Find a physical device with at least a queue family supporting graphics.

### 1.1.5: Logical device and queues

Create the logical device interfacing with the physical device. Then create
the graphics queue from the device.

### 1.1.extra: Refactoring and comments

- Update the readme with explanations on the structure of the repository. 
- Move validation layers related code to its own module.
- Disabled validation layers on release build.

### 1.2.1: Window surface

Create the window, the window surface and the presentation queue.
Update the physical device creation to get a device with presentation support.
At that point, the code will only work on Windows.

## Run it

With validation layers:

```sh
RUST_LOG=vulkan_tutorial_ash=debug cargo run
```

or without:

```sh
RUST_LOG=vulkan_tutorial_ash=debug cargo run --release
```

[0]: https://vulkan-tutorial.com/Introduction
[1]: https://github.com/MaikKlein/ash
