mod detections;
mod error;
mod image;
mod kinds;
mod layers;
mod network;
mod train;
mod utils;

pub use crate::image::{Image, IntoCowImage};
pub use detections::{Detection, Detections, DetectionsIter};
pub use error::Error;
pub use kinds::{
    Activation, BinaryActivation, CostType, IoULoss, LayerType, NmsKind, WeightsNormalizion,
    WeightsType, YoloPoint,
};
pub use layers::{Layer, Layers, LayersIter};
pub use network::Network;
pub use train::train_detector;

/// Bounding box in cxcywh format.
pub type BBox = darknet_sys::box_;
