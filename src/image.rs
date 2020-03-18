use darknet_sys as sys;
use darknet_sys::box_ as BBox;
use image::{error, open, ImageBuffer, RgbImage};
use std::ffi::CString;
use std::mem;
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

    /// Construct image from given RgbImage.
    pub fn from_image_buffer_rgb(buffer: &RgbImage) -> Image {
        let w = buffer.width() as i32;
        let h = buffer.height() as i32;
        let c = 3_i32;
        let raw_buffer = buffer.clone().into_raw();
        unsafe {
            let im = sys::make_image(w, h, c);
            let mut data = Vec::from_raw_parts(im.data, (w * h * c) as usize, (w * h * c) as usize);
            for k in 0..c {
                for j in 0..h {
                    for i in 0..w {
                        let dst_index = (i + w * j + w * h * k) as usize;
                        let src_index = (k + c * i + c * w * j) as usize;
                        data[dst_index] = raw_buffer[src_index] as f32 / 255.0;
                    }
                }
            }
            mem::forget(data);
            return Image { image: im };
        }
    }

    /// Open image from file.
    pub fn open(filename: &str) -> error::ImageResult<Image> {
        Ok(Image::from_image_buffer_rgb(
            open(filename)?.as_rgb8().unwrap(),
        ))
    }

    /// Convert image copy to RgbImage.
    pub fn to_image_buffer_rgb(&self) -> RgbImage {
        let w = self.image.w as usize;
        let h = self.image.h as usize;
        let c = self.image.c as usize;
        let size = w * h * c;
        let mut buffer = vec![0_u8; size];
        unsafe {
            let data = slice::from_raw_parts(self.image.data, size);
            for k in 0..c {
                for i in 0..w * h {
                    buffer[i * c + k] = (255.0 * data[i + k * w * h]) as u8;
                }
            }
            ImageBuffer::from_raw(w as u32, h as u32, buffer).expect("Not a valid RGB image")
        }
    }

    /// Save image to file.
    pub fn save(&self, path: &str) -> error::ImageResult<()> {
        self.to_image_buffer_rgb().save(path)
    }

    /// Resize image (uses letterbox_image internally).
    pub fn resize(&mut self, w: usize, h: usize) {
        unsafe {
            //self.image = sys::resize_image(self.image, w as c_int, h as c_int);
            self.image = sys::letterbox_image(self.image, w as c_int, h as c_int);
        }
    }

    /// Show image if darknet library was compiled with OpenCV or save image as 'name'.jpg
    pub fn show(&self, name: &str) -> i32 {
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

    /// Crop bbox from image.
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
