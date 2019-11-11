use ash::extensions::khr::Surface;
use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::os::raw::c_char;
use winit::Window;

#[derive(Copy, Clone, Debug)]
pub enum SurfaceError {
    SurfaceCreationError(vk::Result),
    WindowNotSupportedError,
}

impl std::error::Error for SurfaceError {}

impl std::fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurfaceError::SurfaceCreationError(result) => {
                write!(f, "SurfaceCreationError: {}", result)
            }
            SurfaceError::WindowNotSupportedError => write!(f, "WindowNotSupportedError"),
        }
    }
}

/// Get required instance extensions.
/// This is windows specific.
#[cfg(target_os = "windows")]
pub fn required_extension_names() -> Vec<*const c_char> {
    use ash::extensions::khr::Win32Surface;
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}

/// Get required instance extensions.
/// This is linux specific.
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const c_char> {
    use ash::extensions::khr::XlibSurface;
    vec![Surface::name().as_ptr(), XlibSurface::name().as_ptr()]
}

/// Get required instance extensions.
/// This is macos specific.
#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const c_char> {
    use ash::extensions::mvk::MacOSSurface;
    vec![Surface::name().as_ptr(), MacOSSurface::name().as_ptr()]
}

/// Get required instance extensions.
/// This is android specific.
#[cfg(target_os = "android")]
pub fn required_extension_names() -> Vec<*const c_char> {
    use ash::extensions::khr::AndroidSurface;
    vec![Surface::name().as_ptr(), AndroidSurface::name().as_ptr()]
}

/// Create the surface.
/// This is windows specific.
#[cfg(target_os = "windows")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, SurfaceError> {
    use ash::extensions::khr::Win32Surface;

    log::debug!("Creating windows surface");
    match window.raw_window_handle() {
        RawWindowHandle::Windows(handle) => {
            let create_info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(handle.hinstance)
                .hwnd(handle.hwnd);
            let surface_loader = Win32Surface::new(entry, instance);
            surface_loader
                .create_win32_surface(&create_info, None)
                .map_err(|e| SurfaceError::SurfaceCreationError(e))
        }
        _ => Err(SurfaceError::WindowNotSupportedError),
    }
}

/// Create the surface.
/// This is linux specific.
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, SurfaceError> {
    use ash::extensions::khr::XlibSurface;
    use std::ffi::c_void;

    log::debug!("Creating linux surface");
    match window.raw_window_handle() {
        RawWindowHandle::Xlib(handle) => {
            let create_info = vk::XlibSurfaceCreateInfoKHR::builder()
                .window(handle.window)
                .dpy(handle.display as *mut *const c_void);
            let surface_loader = XlibSurface::new(entry, instance);
            surface_loader
                .create_xlib_surface(&create_info, None)
                .map_err(|e| SurfaceError::SurfaceCreationError(e))
        }
        _ => Err(SurfaceError::WindowNotSupportedError),
    }
}

/// Create the surface.
/// This is macos specific.
#[cfg(target_os = "macos")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, SurfaceError> {
    use ash::extensions::mvk::MacOSSurface;

    log::debug!("Creating macos surface");
    match window.raw_window_handle() {
        RawWindowHandle::MacOS(handle) => {
            let create_info = vk::MacOSSurfaceCreateInfoMVK::builder().view(&*(handle.ns_view));
            let surface_loader = MacOSSurface::new(entry, instance);
            surface_loader
                .create_mac_os_surface_mvk(&create_info, None)
                .map_err(|e| SurfaceError::SurfaceCreationError(e))
        }
        _ => Err(SurfaceError::WindowNotSupportedError),
    }
}

/// Create the surface.
/// This is android specific.
#[cfg(target_os = "android")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &Window,
) -> Result<vk::SurfaceKHR, SurfaceError> {
    use ash::extensions::khr::AndroidSurface;

    log::debug!("Creating android surface");
    match window.raw_window_handle() {
        RawWindowHandle::Android(handle) => {
            let create_info =
                vk::AndroidSurfaceCreateInfoKHR::builder().window(handle.a_native_window);

            let surface_loader = AndroidSurface::new(entry, instance);
            surface_loader
                .create_android_surface(&create_info, None)
                .map_err(|e| SurfaceError::SurfaceCreationError(e))
        }
        _ => Err(SurfaceError::WindowNotSupportedError),
    }
}
