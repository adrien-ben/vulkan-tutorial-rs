use ash::vk;
use ash::{
    extensions::khr::Surface,
    version::{EntryV1_0, InstanceV1_0},
};
use winit::Window;

/// Get required instance extensions.
/// This is windows specific.
#[cfg(all(windows))]
pub fn required_extension_names() -> Vec<*const i8> {
    use ash::extensions::khr::Win32Surface;
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

/// Get required instance extensions.
/// This is linux specific.
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const i8> {
    use ash::extensions::khr::XlibSurface;
    vec![Surface::name().as_ptr(), XlibSurface::name().as_ptr()]
}

/// Get required instance extensions.
/// This is macos specific.
#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const i8> {
    use ash::extensions::mvk::MacOSSurface;
    vec![Surface::name().as_ptr(), MacOSSurface::name().as_ptr()]
}

/// Create the surface.
/// This is windows specific.
#[cfg(all(windows))]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::khr::Win32Surface;
    use std::{os::raw::c_void, ptr};
    use winapi::{shared::windef::HWND, um::libloaderapi::GetModuleHandleW};
    use winit::os::windows::WindowExt;

    let hwnd = window.get_hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}

/// Create the surface.
/// This is linux specific.
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::khr::XlibSurface;
    use winit::os::unix::WindowExt;

    let x11_display = window.get_xlib_display().unwrap();
    let x11_window = window.get_xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::builder()
        .window(x11_window)
        .dpy(x11_display as *mut vk::Display);

    let xlib_surface_loader = XlibSurface::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

/// Create the surface.
/// This is macos specific.
#[cfg(target_os = "macos")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use ash::extensions::mvk::MacOSSurface;
    use cocoa::{
        appkit::{NSView, NSWindow},
        base::id as cocoa_id,
    };
    use metal_rs::CoreAnimationLayer;
    use objc::runtime::YES;
    use std::{mem, os::raw::c_void, ptr};
    use winit::os::macos::WindowExt;

    let wnd: cocoa_id = mem::transmute(window.get_nswindow());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_M,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.get_nsview() as *const c_void,
    };

    let macos_surface_loader = MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface_mvk(&create_info, None)
}
