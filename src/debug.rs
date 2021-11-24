use ash::extensions::ext::DebugUtils;
use ash::{vk, Entry, Instance};
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
};

#[cfg(debug_assertions)]
pub const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
pub const ENABLE_VALIDATION_LAYERS: bool = false;

const REQUIRED_LAYERS: [&str; 1] = ["VK_LAYER_KHRONOS_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugUtilsMessageSeverityFlagsEXT,
    typ: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    use vk::DebugUtilsMessageSeverityFlagsEXT as Flag;

    let message = CStr::from_ptr((*p_callback_data).p_message);
    match flag {
        Flag::VERBOSE => log::debug!("{:?} - {:?}", typ, message),
        Flag::INFO => log::info!("{:?} - {:?}", typ, message),
        Flag::WARNING => log::warn!("{:?} - {:?}", typ, message),
        _ => log::error!("{:?} - {:?}", typ, message),
    }
    vk::FALSE
}

/// Get the pointers to the validation layers names.
/// Also return the corresponding `CString` to avoid dangling pointers.
pub fn get_layer_names_and_pointers() -> (Vec<CString>, Vec<*const c_char>) {
    let layer_names = REQUIRED_LAYERS
        .iter()
        .map(|name| CString::new(*name).unwrap())
        .collect::<Vec<_>>();
    let layer_names_ptrs = layer_names
        .iter()
        .map(|name| name.as_ptr())
        .collect::<Vec<_>>();
    (layer_names, layer_names_ptrs)
}

/// Check if the required validation set in `REQUIRED_LAYERS`
/// are supported by the Vulkan instance.
///
/// # Panics
///
/// Panic if at least one on the layer is not supported.
pub fn check_validation_layer_support(entry: &Entry) {
    for required in REQUIRED_LAYERS.iter() {
        let found = entry
            .enumerate_instance_layer_properties()
            .unwrap()
            .iter()
            .any(|layer| {
                let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
                let name = name.to_str().expect("Failed to get layer name pointer");
                required == &name
            });

        if !found {
            panic!("Validation layer not supported: {}", required);
        }
    }
}

/// Setup the debug message if validation layers are enabled.
pub fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Option<(DebugUtils, vk::DebugUtilsMessengerEXT)> {
    if !ENABLE_VALIDATION_LAYERS {
        return None;
    }

    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .flags(vk::DebugUtilsMessengerCreateFlagsEXT::all())
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_callback));
    let debug_utils = DebugUtils::new(entry, instance);
    let debug_utils_messenger = unsafe {
        debug_utils
            .create_debug_utils_messenger(&create_info, None)
            .unwrap()
    };

    Some((debug_utils, debug_utils_messenger))
}
