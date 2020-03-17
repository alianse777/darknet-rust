use darknet_sys as sys;
use darknet_sys::box_ as BBox;
pub use darknet_sys::{IMTYPE_BMP, IMTYPE_JPG, IMTYPE_PNG, IMTYPE_TGA};
use std::ffi::CString;
use std::os::raw::c_int;
use std::slice;

pub struct Image {
    pub image: sys::image,
}

impl Image {
    /// Returns empty RGB image.
    pub fn empty(w: i32, h: i32) -> Image {
        unsafe {
            Image {
                image: sys::make_image(w as c_int, h as c_int, 3 as c_int),
            }
        }
    }

    /// Returns pointer to raw image data.
    pub fn get_raw_data(&self) -> *mut f32 {
        self.image.data
    }

    /// Returns pixel values as slice.
    pub fn get_data<'a>(&'a self) -> &'a [f32] {
        return unsafe {
            slice::from_raw_parts(
                self.image.data,
                (self.image.h * self.image.w * self.image.c) as usize,
            )
        };
    }

    /// Open image from file.
    pub fn open(filename: &str) -> Image {
        let fname = CString::new(filename).expect("CString::new(filename) failed");
        unsafe {
            Image {
                image: sys::load_image_color(fname.into_raw(), 0, 0),
            }
        }
    }

    /// Show image if darknet library was compiled with OpenCV or save image as 'name'.jpg
    pub fn show_image(&self, name: &str) -> i32 {
        unsafe {
            sys::show_image(
                self.image,
                CString::new(name)
                    .expect("CString::new(name) failed")
                    .into_raw(),
                0 as c_int,
            )
        }
    }

    /// Save image to file <name>. Extension will be added automaticaly based on IMTYPE specified. Possible IMTYPEs: darknet::{IMTYPE_BMP, IMTYPE_JPG, IMTYPE_PNG, IMTYPE_TGA}
    pub fn save_image_options(&self, name: &str, img_type: sys::IMTYPE) {
        unsafe {
            sys::save_image_options(
                self.image,
                CString::new(name)
                    .expect("CString::new(name) failed")
                    .into_raw(),
                img_type,
                100 as c_int,
            );
        }
    }

    /// Resize image (uses letterbox_image internally)
    pub fn resize(&mut self, w: usize, h: usize) {
        unsafe {
            //self.image = sys::resize_image(self.image, w as c_int, h as c_int);
            self.image = sys::letterbox_image(self.image, w as c_int, h as c_int);
        }
    }

    /// Image width
    pub fn get_w(&self) -> usize {
        self.image.w as usize
    }

    ///Image height
    pub fn get_h(&self) -> usize {
        self.image.h as usize
    }

    pub fn get_wf(&self) -> f32 {
        self.image.w as f32
    }

    pub fn get_hf(&self) -> f32 {
        self.image.h as f32
    }

    pub fn crop_bbox(&self, bbox: &BBox) -> Image {
        let left = (bbox.x - bbox.w / 2.0) * self.image.w as f32;
        let right = (bbox.x + bbox.w / 2.0) * self.image.w as f32;
        let top = (bbox.y - bbox.h / 2.0) * self.image.h as f32;
        let bot = (bbox.y + bbox.h / 2.0) * self.image.h as f32;
        unsafe {
            Image {
                image: sys::crop_image(
                    self.image,
                    left as c_int,
                    top as c_int,
                    (right - left) as c_int,
                    (bot - top) as c_int,
                ),
            }
        }
    }
}

impl Clone for Image {
    /// Full copy of image
    fn clone(&self) -> Image {
        unsafe {
            let image = sys::copy_image(self.image);
            Image { image: image }
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { sys::free_image(self.image) }
    }
}
