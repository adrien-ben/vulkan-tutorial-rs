mod util;

use ash::version::{EntryV1_0, InstanceV1_0};
use ash::{vk, Entry, Instance};
use std::{error::Error, ffi::CString, result::Result};

struct VulkanApp {
    _entry: Entry,
    instance: Instance,
}

impl VulkanApp {
    fn new() -> Result<Self, Box<dyn Error>> {
        log::debug!("Creating application.");

        let entry = ash::Entry::new().expect("Failed to create entry.");
        let instance = Self::create_instance(&entry)?;

        Ok(Self {
            _entry: entry,
            instance,
        })
    }

    fn create_instance(entry: &Entry) -> Result<Instance, Box<dyn Error>> {
        let app_info = vk::ApplicationInfo::builder()
            .application_name(CString::new("Vulkan Application")?.as_c_str())
            .application_version(ash::vk_make_version!(0, 1, 0))
            .engine_name(CString::new("No Engine")?.as_c_str())
            .engine_version(ash::vk_make_version!(0, 1, 0))
            .api_version(ash::vk_make_version!(1, 0, 0))
            .build();

        let extension_names = util::required_extension_names();

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);

        unsafe { Ok(entry.create_instance(&instance_create_info, None)?) }
    }

    fn run(&mut self) {
        log::debug!("Running application.");
    }
}

impl Drop for VulkanApp {
    fn drop(&mut self) {
        log::debug!("Dropping application.");
        unsafe {
            self.instance.destroy_instance(None);
        }
    }
}

fn main() {
    env_logger::init();
    match VulkanApp::new() {
        Ok(mut app) => app.run(),
        Err(error) => log::error!("Failed to create application. Cause: {}", error),
    }
}
