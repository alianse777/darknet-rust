use crate::BBox;
use darknet_sys as sys;
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
    pub fn get<'a>(&'a self, index: usize) -> Option<Detection<'a>> {
        if index >= self.n_detections {
            return None;
        }

        let slice = unsafe { slice::from_raw_parts(self.detections.as_ptr(), self.n_detections) };

        Some(Detection {
            detection: &slice[index],
        })
    }

    /// Returns detections count.
    pub fn len(&self) -> usize {
        self.n_detections
    }

    pub fn iter<'a>(&'a self) -> DetectionsIter<'a> {
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
        if let Some(_) = opt {
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
