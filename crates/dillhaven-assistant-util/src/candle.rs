use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::Device;

pub fn get_device() -> Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        Ok(Device::Cpu)
    }
}
