use crate::image::Image;
use darknet_sys as sys;
pub use darknet_sys::box_ as BBox;
use std::ffi::CString;
use std::iter::Iterator;
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::sync::Arc;

fn get_max_prob_label(labels: &Vec<String>, det: &sys::detection) -> (String, f32) {
    let probs =
        unsafe { Vec::from_raw_parts(det.prob, det.classes as usize, det.classes as usize) };
    let (max_label, max_prob) = Iterator::zip(labels.iter(), probs.iter())
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .expect("No labels found!");
    let max_prob = *max_prob;
    mem::forget(probs);
    return (max_label.to_string(), max_prob);
}

#[derive(Debug)]
pub struct Detections {
    detections: Vec<sys::detection>,
    names: Arc<Vec<String>>,
    thresh: f32,
    mask_size: usize,
}

impl Detections {
    pub fn new(
        detections: Vec<sys::detection>,
        names: &Arc<Vec<String>>,
        thresh: f32,
        mask_size: usize,
    ) -> Detections {
        Detections {
            detections,
            names: names.clone(),
            thresh,
            mask_size,
        }
    }

    /// Returns vector of bounding boxes
    pub fn get_boxes(&self) -> Vec<BBox> {
        self.detections.iter().map(|x| x.bbox).collect()
    }

    /// Returns detected object labels
    pub fn get_labels(&self) -> Vec<String> {
        self.detections
            .iter()
            .map(|x| get_max_prob_label(&self.names, &x))
            .filter(|x| x.1 > self.thresh)
            .map(|x| x.0)
            .collect()
    }

    /// Returns vector of raw detection type
    pub fn get_raw_vec(&self) -> Vec<sys::detection> {
        self.detections.clone()
    }

    /// Draws detection boxes with labels on image
    pub fn draw_on_image(&self, image: &mut Image) {
        let mut names_raw: Vec<*mut c_char> = self
            .names
            .iter()
            .map(|x| {
                CString::new(&x[..])
                    .expect("CString::new failed")
                    .into_raw()
            })
            .collect();
        unsafe {
            sys::draw_detections(
                image.image,
                self.get_raw_vec().as_mut_ptr(),
                self.count() as i32,
                0.0,
                names_raw.as_mut_ptr(),
                sys::load_alphabet(),
                names_raw.len() as i32,
            );
        }
    }

    /// Crops all detected objects from image
    /// <br> returns Vec<label, image>
    pub fn crop_from(&self, img: &Image) -> Vec<(String, Image)> {
        self.detections
            .iter()
            .map(|x| (get_max_prob_label(&self.names, &x), x.bbox))
            .filter(|x| (x.0).1 > self.thresh)
            .map(|x| ((x.0).0.to_string(), img.crop_bbox(&x.1)))
            .collect()
    }

    /// Returns detections count
    pub fn count(&self) -> usize {
        self.detections.len()
    }
}

impl Drop for Detections {
    fn drop(&mut self) {
        unsafe {
            for d in &self.detections {
                mem::drop(Vec::from_raw_parts(
                    d.prob,
                    d.classes as usize,
                    d.classes as usize,
                ));
                if d.mask != ptr::null_mut() && self.mask_size > 4 {
                    mem::drop(Vec::from_raw_parts(
                        d.mask,
                        self.mask_size - 4,
                        self.mask_size - 4,
                    ));
                }
            }
        }
    }
}
