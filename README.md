# Vulkan tutorial

Vulkan [tutorials][0] written in Rust using [Ash][1].

## 1.1.1: Base code

Application setup. We don't setup the window system now as it's done in 
the original tutorial.

## 1.1.2: Instance

Create and destroy the Vulkan instance with required surface extensions.

## Run it

```sh
RUST_LOG=vulkan_tutorial_ash=debug cargo run
```

[0]: https://vulkan-tutorial.com/Introduction
[1]: https://github.com/MaikKlein/ash
