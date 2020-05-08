mod detections;
mod error;
mod image;
mod network;

pub use crate::image::Image;
pub use detections::Detections;
pub use network::Network;

pub type BBox = darknet_sys::box_;
