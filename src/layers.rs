use crate::kinds::{
    Activation, CostType, IoULoss, LayerType, NmsKind, WeightsNormalizion, WeightsType, YoloPoint,
};
use darknet_sys as sys;
use num_traits::FromPrimitive;
use std::iter::{ExactSizeIterator, FusedIterator};

/// A collection layers.
#[derive(Debug)]
pub struct Layers<'a> {
    pub(crate) layers: &'a [sys::layer],
}

impl<'a> Layers<'a> {
    /// Get the layer by index.
    pub fn get(&self, index: usize) -> Option<Layer<'a>> {
        self.layers.get(index).map(|layer| Layer { layer })
    }

    /// Get the iterator of the collection of layers.
    pub fn iter(&'a self) -> LayersIter<'a> {
        LayersIter {
            layers: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Layers<'a> {
    type Item = Layer<'a>;
    type IntoIter = LayersIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// A iterator of a collection of layers.
#[derive(Debug, Clone)]
pub struct LayersIter<'a> {
    layers: &'a Layers<'a>,
    index: usize,
}

impl<'a> Iterator for LayersIter<'a> {
    type Item = Layer<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let opt = self.layers.get(self.index);
        if opt.is_some() {
            self.index += 1;
        }
        opt
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.layers.layers.len();
        (len, Some(len))
    }
}

impl<'a> FusedIterator for LayersIter<'a> {}

impl<'a> ExactSizeIterator for LayersIter<'a> {}

/// A layer of the network.
#[derive(Debug)]
pub struct Layer<'a> {
    pub(crate) layer: &'a sys::layer,
}

impl<'a> Layer<'a> {
    /// Get the type of layer.
    pub fn type_(&self) -> Option<LayerType> {
        FromPrimitive::from_usize(self.layer.type_ as usize)
    }

    /// Get the activation type.
    pub fn activation(&self) -> Option<Activation> {
        FromPrimitive::from_usize(self.layer.activation as usize)
    }

    /// Get the cost (or namely the loss) type.
    pub fn cost_type(&self) -> Option<CostType> {
        FromPrimitive::from_usize(self.layer.activation as usize)
    }

    /// Get the weights format type.
    pub fn weights_type(&self) -> Option<WeightsType> {
        FromPrimitive::from_usize(self.layer.weights_type as usize)
    }

    /// Get the weights normalization type.
    pub fn weights_normalization(&self) -> Option<WeightsNormalizion> {
        FromPrimitive::from_usize(self.layer.weights_normalization as usize)
    }

    /// Get the non-maximum suppression (NMS) type.
    pub fn nms_kind(&self) -> Option<NmsKind> {
        FromPrimitive::from_usize(self.layer.nms_kind as usize)
    }

    /// Get the YOLO point type.
    pub fn yolo_point(&self) -> Option<YoloPoint> {
        FromPrimitive::from_usize(self.layer.yolo_point as usize)
    }

    /// Get the IoU loss type.
    pub fn iou_loss(&self) -> Option<IoULoss> {
        FromPrimitive::from_usize(self.layer.iou_loss as usize)
    }

    /// Get the IoU threshold type.
    pub fn iou_thresh_kind(&self) -> Option<IoULoss> {
        FromPrimitive::from_usize(self.layer.iou_thresh_kind as usize)
    }

    /// Get the input tensor height.
    pub fn input_height(&self) -> usize {
        self.layer.h as usize
    }

    /// Get the input tensor width.
    pub fn input_width(&self) -> usize {
        self.layer.w as usize
    }

    /// Get the input tensor channels.
    pub fn input_channels(&self) -> usize {
        self.layer.c as usize
    }

    /// Get the input shape tuple (width, height, channels).
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (
            self.input_width(),
            self.input_height(),
            self.input_channels(),
        )
    }

    /// Get the output tensor height.
    pub fn output_height(&self) -> usize {
        self.layer.out_h as usize
    }

    /// Get the output tensor width.
    pub fn output_width(&self) -> usize {
        self.layer.out_w as usize
    }

    /// Get the output tensor channels.
    pub fn output_channels(&self) -> usize {
        self.layer.out_c as usize
    }

    /// Get the output shape tuple (width, height, channels).
    pub fn output_shape(&self) -> (usize, usize, usize) {
        (
            self.output_width(),
            self.output_height(),
            self.output_channels(),
        )
    }
}
