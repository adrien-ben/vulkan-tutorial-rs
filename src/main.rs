mod util;

use ash::{
    extensions::ext::DebugReport,
    version::{EntryV1_0, InstanceV1_0},
};
use ash::{vk, Entry, Instance};
use std::{
    ffi::{CStr, CString},
    os::raw::{c_char, c_void},
};

const ENABLE_VALIDATION_LAYERS: bool = true;
const REQUIRED_LAYERS: [&'static str; 1] = ["VK_LAYER_LUNARG_standard_validation"];

unsafe extern "system" fn vulkan_debug_callback(
    flag: vk::DebugReportFlagsEXT,
    typ: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void,
) -> u32 {
    if flag == vk::DebugReportFlagsEXT::DEBUG {
        log::debug!("{} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::INFORMATION {
        log::info!("{} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::WARNING {
        log::warn!("{} - {:?}", typ, CStr::from_ptr(p_message));
    } else if flag == vk::DebugReportFlagsEXT::PERFORMANCE_WARNING {
        log::warn!("{} - {:?}", typ, CStr::from_ptr(p_message));
    } else {
        log::error!("{} - {:?}", typ, CStr::from_ptr(p_message));
    }
    vk::FALSE
}

struct VulkanApp {
    _entry: Entry,
    instance: Instance,
    debug_report_callback: Option<(DebugReport, vk::DebugReportCallbackEXT)>,
}

impl VulkanApp {
    fn new() -> Self {
        log::debug!("Creating application.");

        let entry = ash::Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry);
        let debug_report_callback = Self::setup_debug_messenger(&entry, &instance);

        Self {
            _entry: entry,
            instance,
            debug_report_callback,
        }
    }

    fn create_instance(entry: &Entry) -> Instance {
        let app_name = CString::new("Vulkan Application").unwrap();
        let engine_name = CString::new("No Engine").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(app_name.as_c_str())
            .application_version(ash::vk_make_version!(0, 1, 0))
            .engine_name(engine_name.as_c_str())
            .engine_version(ash::vk_make_version!(0, 1, 0))
            .api_version(ash::vk_make_version!(1, 0, 0))
            .build();

        let mut extension_names = util::required_extension_names();
        if ENABLE_VALIDATION_LAYERS {
            extension_names.push(DebugReport::name().as_ptr());
        }

        let layer_names = REQUIRED_LAYERS
            .iter()
            .map(|name| CString::new(*name).expect("Failed to build CString"))
            .collect::<Vec<_>>();
        let layer_names_ptrs = layer_names
            .iter()
            .map(|name| name.as_ptr())
            .collect::<Vec<_>>();

        let mut instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        if ENABLE_VALIDATION_LAYERS {
            Self::check_validation_layer_support(&entry);
            instance_create_info = instance_create_info.enabled_layer_names(&layer_names_ptrs);
        }

        unsafe { entry.create_instance(&instance_create_info, None).unwrap() }
    }

    fn check_validation_layer_support(entry: &Entry) {
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

    fn setup_debug_messenger(
        entry: &Entry,
        instance: &Instance,
    ) -> Option<(DebugReport, vk::DebugReportCallbackEXT)> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }
        let create_info = vk::DebugReportCallbackCreateInfoEXT::builder()
            .flags(vk::DebugReportFlagsEXT::all())
            .pfn_callback(Some(vulkan_debug_callback))
            .build();
        let debug_report = DebugReport::new(entry, instance);
        let debug_report_callback = unsafe {
            debug_report
                .create_debug_report_callback(&create_info, None)
                .unwrap()
        };
        Some((debug_report, debug_report_callback))
    }

    fn run(&mut self) {
        log::debug!("Running application.");
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        unsafe {
            if let Some((report, callback)) = self.debug_report_callback.take() {
                report.destroy_debug_report_callback(callback, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    env_logger::init();
    VulkanApp::new().run()
}
