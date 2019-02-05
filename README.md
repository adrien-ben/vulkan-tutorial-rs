# Vulkan tutorial

Vulkan [tutorials][0] written in Rust using [Ash][1].

## 1.1.1: Base code

Application setup. We don't setup the window system now as it's done in 
the original tutorial.

## 1.1.2: Instance

Create and destroy the Vulkan instance with required surface extensions.

## 1.1.3: Validation layers

Add `VK_LAYER_LUNARG_standard_validation` at instance creation and creates
a debug report callback function after checking that it is available. 
Since we are using the `log` crate, we log the message with the proper log level.
The callback is detroyed at application termination.

## 1.1.4: Physical devices and queue families

Find a physical device with at least a queue family supporting graphics.

## Run it

```sh
RUST_LOG=vulkan_tutorial_ash=debug cargo run
```

[0]: https://vulkan-tutorial.com/Introduction
[1]: https://github.com/MaikKlein/ash
