use crate::{
    detections::Detections,
    error::Error,
    image::IntoCowImage,
    layers::{Layer, Layers},
    utils,
};
use darknet_sys as sys;

use std::{
    ffi::c_void,
    mem,
    os::raw::c_int,
    path::Path,
    ptr::{self, NonNull},
    slice,
};

/// The network wrapper type for Darknet.
pub struct Network {
    net: NonNull<sys::network>,
}

impl Network {
    /// Build the network instance from a configuration file and an optional weights file.
    pub fn load<C, W>(cfg: C, weights: Option<W>, clear: bool) -> Result<Network, Error>
    where
        C: AsRef<Path>,
        W: AsRef<Path>,
    {
        // convert paths to CString
        let weights_cstr = weights
            .map(|path| {
                utils::path_to_cstring(path.as_ref()).ok_or_else(|| Error::EncodingError {
                    reason: format!("the path {} is invalid", path.as_ref().display()),
                })
            })
            .transpose()?;
        let cfg_cstr =
            utils::path_to_cstring(cfg.as_ref()).ok_or_else(|| Error::EncodingError {
                reason: format!("the path {} is invalid", cfg.as_ref().display()),
            })?;

        let ptr = unsafe {
            let raw_weights = weights_cstr
                .as_ref()
                .map(|cstr| cstr.as_ptr() as *mut _)
                .unwrap_or(ptr::null_mut());
            sys::load_network(cfg_cstr.as_ptr() as *mut _, raw_weights, clear as c_int)
        };

        let net = NonNull::new(ptr).ok_or_else(|| Error::InternalError {
            reason: "failed to load model".into(),
        })?;

        // drop paths here to avoid early deallocation
        mem::drop(cfg_cstr);
        mem::drop(weights_cstr);

        Ok(Self { net })
    }

    /// Get network input width.
    pub fn input_width(&self) -> usize {
        unsafe { self.net.as_ref().w as usize }
    }

    /// Get network input height.
    pub fn input_height(&self) -> usize {
        unsafe { self.net.as_ref().h as usize }
    }

    /// Get network input channels.
    pub fn input_channels(&self) -> usize {
        unsafe { self.net.as_ref().c as usize }
    }

    /// Get network input shape tuple (channels, height, width).
    pub fn input_shape(&self) -> (usize, usize, usize) {
        (
            self.input_channels(),
            self.input_height(),
            self.input_width(),
        )
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        unsafe { self.net.as_ref().n as usize }
    }

    /// Get network layers.
    pub fn layers<'a>(&'a self) -> Layers<'a> {
        let layers = unsafe { slice::from_raw_parts(self.net.as_ref().layers, self.num_layers()) };
        Layers { layers }
    }

    /// Get layer by index.
    pub fn get_layer<'a>(&'a self, index: usize) -> Option<Layer<'a>> {
        if index >= self.num_layers() {
            return None;
        }

        unsafe {
            let layer = self.net.as_ref().layers.add(index).as_ref().unwrap();
            Some(Layer { layer })
        }
    }

    /// Run inference on an image.
    pub fn predict<'a, M>(
        &mut self,
        image: M,
        thresh: f32,
        hier_thres: f32,
        nms: f32,
        use_letter_box: bool,
    ) -> Detections
    where
        M: IntoCowImage<'a>,
    {
        let cow = image.into_cow_image();

        unsafe {
            let output_layer = self
                .net
                .as_ref()
                .layers
                .add(self.num_layers() - 1)
                .as_ref()
                .unwrap();

            // run prediction
            if use_letter_box {
                sys::network_predict_image_letterbox(self.net.as_ptr(), cow.image);
            } else {
                sys::network_predict_image(self.net.as_ptr(), cow.image);
            }

            let mut nboxes: c_int = 0;
            let dets_ptr = sys::get_network_boxes(
                self.net.as_mut(),
                cow.width() as c_int,
                cow.height() as c_int,
                thresh,
                hier_thres,
                ptr::null_mut(),
                1,
                &mut nboxes,
                use_letter_box as c_int,
            );
            let dets = NonNull::new(dets_ptr).unwrap();

            // NMS sort
            if nms != 0.0 {
                if output_layer.nms_kind == sys::NMS_KIND_DEFAULT_NMS {
                    sys::do_nms_sort(dets.as_ptr(), nboxes, output_layer.classes, nms);
                } else {
                    sys::diounms_sort(
                        dets.as_ptr(),
                        nboxes,
                        output_layer.classes,
                        nms,
                        output_layer.nms_kind,
                        output_layer.beta_nms,
                    );
                }
            }

            Detections {
                detections: dets,
                n_detections: nboxes as usize,
            }
        }
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            let ptr = self.net.as_ptr();
            sys::free_network(*ptr);

            // The network* pointer was allocated by calloc
            // We have to deallocate it manually
            libc::free(ptr as *mut c_void);
        }
    }
}

unsafe impl Send for Network {}
