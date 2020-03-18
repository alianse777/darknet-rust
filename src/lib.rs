mod detections;
mod image;
mod network;
pub use crate::detections::Detections;
pub use crate::image::Image; //, IMTYPE_BMP, IMTYPE_JPG, IMTYPE_PNG, IMTYPE_TGA};
pub use crate::network::{load_labels, Network};
