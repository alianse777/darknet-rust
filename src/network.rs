use crate::{detections::Detections, error::Error, image::Image};
use darknet_sys as sys;

use std::{
    borrow::{Borrow, Cow},
    ffi::{c_void, CString},
    os::raw::c_int,
    path::Path,
    ptr::{self, NonNull},
};

#[cfg(unix)]
fn path_to_cstring<'a>(path: &'a Path) -> Option<CString> {
    use std::os::unix::ffi::OsStrExt;
    Some(CString::new(path.as_os_str().as_bytes()).unwrap())
}

#[cfg(not(unix))]
fn path_to_cstring<'a>(path: &'a Path) -> Option<CString> {
    path.to_str().map(|s| CString::new(s.as_bytes()).unwrap())
}

pub struct Network {
    net: NonNull<sys::network>,
}

impl Network {
    pub fn load<C: AsRef<Path>, W: AsRef<Path>>(
        cfg: C,
        weights: Option<W>,
        clear: bool,
    ) -> Result<Network, Error> {
        let weights_cstr = weights
            .map(|path| {
                path_to_cstring(path.as_ref()).ok_or_else(|| Error::EncodingError {
                    reason: format!("the path {} is invalid", path.as_ref().display()),
                })
            })
            .transpose()?;
        let cfg_cstr = path_to_cstring(cfg.as_ref()).ok_or_else(|| Error::EncodingError {
            reason: format!("the path {} is invalid", cfg.as_ref().display()),
        })?;

        let ptr = unsafe {
            let raw_weights = weights_cstr
                .map(|cstr| cstr.as_ptr() as *mut _)
                .unwrap_or(ptr::null_mut());
            sys::load_network(cfg_cstr.as_ptr() as *mut _, raw_weights, clear as c_int)
        };

        let net = NonNull::new(ptr).ok_or_else(|| Error::InternalError {
            reason: "failed to load model".into(),
        })?;

        Ok(Self { net })
    }

    /// Network input width.
    pub fn input_width(&self) -> usize {
        unsafe { self.net.as_ref().w as usize }
    }

    /// Network input height.
    pub fn input_height(&self) -> usize {
        unsafe { self.net.as_ref().h as usize }
    }

    pub fn predict<M>(
        &mut self,
        image: M,
        thresh: f32,
        hier_thres: f32,
        nms: f32,
        use_letter_box: bool,
    ) -> Detections
    where
        M: Borrow<Image>,
    {
        let borrow = image.borrow();
        let maybe_resized =
            if borrow.width() == self.input_width() && borrow.height() == self.input_height() {
                Cow::Borrowed(borrow)
            } else {
                let resized = if use_letter_box {
                    borrow.letter_box(self.input_width(), self.input_height())
                } else {
                    borrow.resize(self.input_width(), self.input_height())
                };
                Cow::Owned(resized)
            };

        unsafe {
            let output_layer = self
                .net
                .as_ref()
                .layers
                .add(self.num_layers() - 1)
                .as_ref()
                .unwrap();

            // run prediction
            sys::network_predict(*self.net.as_ref(), maybe_resized.get_raw_data());
            let mut nboxes: c_int = 0;
            let dets_ptr = sys::get_network_boxes(
                self.net.as_mut(),
                maybe_resized.width() as c_int,
                maybe_resized.height() as c_int,
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

    pub fn num_layers(&self) -> usize {
        unsafe { self.net.as_ref().n as usize }
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
