mod detections;
mod error;
mod image;
mod kinds;
mod layers;
mod network;

pub use crate::image::Image;
pub use detections::Detections;
pub use error::Error;
pub use kinds::{
    Activation, BinaryActivation, CostType, IoULoss, LayerType, NmsKind, WeightsNormalizion,
    WeightsType, YoloPoint,
};
pub use layers::{Layer, Layers, LayersIter};
pub use network::Network;

/// Bounding box in cxcywh format.
pub type BBox = darknet_sys::box_;
