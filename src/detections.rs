use crate::BBox;
use darknet_sys as sys;
use std::cmp::Ordering::Less;
use std::{
    iter::{ExactSizeIterator, FusedIterator, Iterator},
    os::raw::c_int,
    ptr::NonNull,
    slice,
};

/// An instance of detection.
#[derive(Debug)]
pub struct Detection<'a> {
    detection: &'a sys::detection,
}

impl<'a> Detection<'a> {
    /// Get the bounding box of the object.
    pub fn bbox(&self) -> &BBox {
        &self.detection.bbox
    }

    /// Get the number of classes.
    pub fn num_classes(&self) -> usize {
        self.detection.classes as usize
    }

    /// Get the output probabilities of each class.
    pub fn probabilities(&self) -> &[f32] {
        unsafe { slice::from_raw_parts(self.detection.prob, self.num_classes()) }
    }

    /// Get the class index with maximum probability.
    ///
    /// The method accepts an optional [prob_threshold].
    /// If the class with maximum probability is above the [prob_threshold],
    /// it returns the tuple (class_id, corresponding_probability).
    /// Otherwise, it returns None.
    pub fn best_class(&self, prob_threshold: Option<f32>) -> Option<(usize, f32)> {
        self.probabilities()
            .iter()
            .enumerate()
            .filter(|(_, prob)| prob_threshold.map_or(true, |thresh| thresh.lt(prob)))
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Less))
            .map(|(idx, prob)| (idx, *prob))
    }

    pub fn uc(&self) -> Option<&[f32]> {
        let ptr = self.detection.uc;
        if ptr.is_null() {
            None
        } else {
            unsafe { Some(slice::from_raw_parts(ptr, 4)) }
        }
    }

    /// The the score of objectness.
    pub fn objectness(&self) -> f32 {
        self.detection.objectness
    }

    pub fn sort_class(&self) -> usize {
        self.detection.sort_class as usize
    }
}

/// A collection of detections.
#[derive(Debug)]
pub struct Detections {
    pub(crate) detections: NonNull<sys::detection>,
    pub(crate) n_detections: usize,
}

impl Detections {
    /// Get a detection instance by index.
    pub fn get(&self, index: usize) -> Option<Detection> {
        if index >= self.n_detections {
            return None;
        }

        let slice = unsafe { slice::from_raw_parts(self.detections.as_ptr(), self.n_detections) };

        Some(Detection {
            detection: &slice[index],
        })
    }

    /// Return detections count.
    pub fn len(&self) -> usize {
        self.n_detections
    }

    /// Returns `true` if the detections has a length of 0.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the iterator of a collection of detections.
    pub fn iter(&self) -> DetectionsIter {
        DetectionsIter {
            detections: self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a Detections {
    type Item = Detection<'a>;
    type IntoIter = DetectionsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Drop for Detections {
    fn drop(&mut self) {
        unsafe {
            sys::free_detections(self.detections.as_mut(), self.n_detections as c_int);
        }
    }
}

unsafe impl Send for Detections {}

/// The iterator of a collection of detections.
#[derive(Debug, Clone)]
pub struct DetectionsIter<'a> {
    detections: &'a Detections,
    index: usize,
}

impl<'a> Iterator for DetectionsIter<'a> {
    type Item = Detection<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let opt = self.detections.get(self.index);
        if opt.is_some() {
            self.index += 1;
        }
        opt
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.detections.len(), Some(self.detections.len()))
    }
}

impl<'a> FusedIterator for DetectionsIter<'a> {}

impl<'a> ExactSizeIterator for DetectionsIter<'a> {}
