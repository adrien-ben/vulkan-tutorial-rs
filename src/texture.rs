use ash::{Device, vk};

#[derive(Clone, Copy)]
pub struct Texture {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: Option<vk::Sampler>,
}

impl Texture {
    pub fn new(
        image: vk::Image,
        memory: vk::DeviceMemory,
        view: vk::ImageView,
        sampler: Option<vk::Sampler>,
    ) -> Self {
        Texture {
            image,
            memory,
            view,
            sampler,
        }
    }

    pub fn destroy(&mut self, device: &Device) {
        unsafe {
            if let Some(sampler) = self.sampler.take() {
                device.destroy_sampler(sampler, None);
            }
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
