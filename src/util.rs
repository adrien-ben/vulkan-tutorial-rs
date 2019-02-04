use ash::extensions::khr::{Surface, Win32Surface};

/// Get required instance extensions.
/// This is windows specific.
pub fn required_extension_names() -> Vec<*const i8> {
    vec![Surface::name().as_ptr(), Win32Surface::name().as_ptr()]
}
