use ash::vk;
use ash::{
    extensions::khr::{Surface, Win32Surface},
    version::{EntryV1_0, InstanceV1_0},
};
use std::{os::raw::c_void, ptr};
use winapi::{shared::windef::HWND, um::libloaderapi::GetModuleHandleW};
use winit::{os::windows::WindowExt, Window};

/// Get required instance extensions.
/// This is windows specific.
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

/// Create the surface.
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
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
