use crate::detections::Detections;
use crate::image::Image;
use darknet_sys as sys;
use std::ffi::CString;
use std::fs;
use std::io;
use std::os::raw::c_int;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

//pub type Alphabet = Box<*mut sys::image>;

#[cfg(unix)]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_ref().as_os_str().as_bytes().to_vec()
}

#[cfg(not(unix))]
fn path_to_bytes<P: AsRef<Path>>(path: P) -> Vec<u8> {
    path.as_ref().to_string_lossy().to_string().into_bytes()
}

/// Reads file line-by-line and returns vector of strings.
/// Useful for loading object labels from file.
pub fn load_labels<P: AsRef<Path> + ?Sized>(file_name: &P) -> Result<Vec<String>, io::Error> {
    Ok(fs::read_to_string(file_name)?
        .lines()
        .map(|x| x.trim().to_string())
        .collect())
}

pub struct Network {
    net: Box<sys::network>,
    labels: Arc<Vec<String>>,
}

impl Network {
    /// Load network from config file `cfg` (under cfg/ subdir) and weights file  `weights` (can be obtained from https://pjreddie.com/darknet/, optional if training).
    /// <br>`clear` - Reset network data (used for training).
    /// <br>`labels` - vector of object labels the model was trained on (i.e. vec!["car", "bird", "dog"...]).
    pub fn load<C: AsRef<Path> + ?Sized, W: AsRef<Path> + ?Sized>(
        cfg: &C,
        weights: Option<&W>,
        clear: bool,
        labels: Vec<String>,
    ) -> Option<Network> {
        let raw_weights = match weights {
            Some(w) => CString::new(path_to_bytes(w))
                .expect("CString::new(weights_file) failed")
                .into_raw(),
            None => ptr::null_mut(),
        };
        unsafe {
            let raw_cfg = CString::new(path_to_bytes(cfg))
                .expect("CString::new(config_file) failed")
                .into_raw();
            let net = sys::load_network(raw_cfg, raw_weights, clear as c_int);
            let _ = CString::from_raw(raw_cfg);
            let _ = CString::from_raw(raw_weights);
            if net != ptr::null_mut() {
                sys::set_batch_network(net, 1);
                return Some(Network {
                    net: Box::from_raw(net),
                    labels: Arc::new(labels),
                });
            } else {
                return None;
            }
        }
    }

    /// Network input width.
    pub fn get_w(&self) -> usize {
        self.net.w as usize
    }

    /// Network input height.
    pub fn get_h(&self) -> usize {
        self.net.h as usize
    }

    /// Predict and return object bboxes (with probability > 'thresh').
    /// <br>'nms' - overlap threshold for non-maximum suppression (higher = more overlapping allowed)
    pub fn predict(&mut self, image: &mut Image, thresh: f32, nms: f32) -> Detections {
        image.resize(self.get_w(), self.get_h());
        unsafe {
            sys::network_predict(&mut *self.net, image.get_raw_data());
            let mut nboxes: c_int = 0;
            let det_ptr = sys::get_network_boxes(
                &mut *self.net,
                1,
                1,
                thresh,
                0.0,
                ptr::null_mut(),
                0,
                &mut nboxes,
            );
            if nms != 0.0 {
                sys::do_nms_sort(det_ptr, nboxes, self.labels.len() as i32, nms);
            }
            Detections::new(
                Vec::from_raw_parts(det_ptr, nboxes as usize, nboxes as usize),
                &self.labels,
                thresh,
            )
        }
    }

    /// Save network weights to file
    pub fn save_weights<P: AsRef<Path> + ?Sized>(&mut self, file_name: &P) {
        let file_name = CString::new(path_to_bytes(file_name)).expect("CString::new(file_name) failed");
        unsafe {
            sys::save_weights(&mut *self.net, file_name.into_raw());
        }
    }

    /// Returns vector of object labels
    pub fn get_labels(&self) -> Vec<String> {
        self.labels.as_ref().clone()
    }
}
