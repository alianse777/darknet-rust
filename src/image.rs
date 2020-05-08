use crate::{error::Error, BBox};
use darknet_sys as sys;
use image::{DynamicImage, ImageBuffer, Pixel};
use std::{convert::TryFrom, ops::Deref, os::raw::c_int, path::Path, slice};

#[derive(Debug)]
pub struct Image {
    pub image: sys::image,
}

impl Image {
    /// Returns an image filled with zeros.
    pub fn zeros(w: usize, h: usize, c: usize) -> Image {
        unsafe {
            Image {
                image: sys::make_image(w as c_int, h as c_int, c as c_int),
            }
        }
    }

    /// Open image from file.
    pub fn open<P: AsRef<Path>>(filename: P) -> Result<Self, Error> {
        let image: Self = image::open(filename)?.into();
        Ok(image)
    }

    /// Resize image without keeping the ratio.
    pub fn resize(&self, w: usize, h: usize) -> Self {
        let image = unsafe { sys::resize_image(self.image, w as c_int, h as c_int) };
        Image { image }
    }

    /// Resize image while keeping the ratio.
    pub fn letter_box(&self, w: usize, h: usize) -> Self {
        let image = unsafe { sys::letterbox_image(self.image, w as c_int, h as c_int) };
        Image { image }
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

    /// Returns pixel values as slice.
    pub fn get_data_mut<'a>(&'a self) -> &'a mut [f32] {
        return unsafe {
            slice::from_raw_parts_mut(
                self.image.data,
                (self.image.h * self.image.w * self.image.c) as usize,
            )
        };
    }

    /// Image width
    pub fn width(&self) -> usize {
        self.image.w as usize
    }

    /// Image height
    pub fn height(&self) -> usize {
        self.image.h as usize
    }

    /// Image channel
    pub fn channels(&self) -> usize {
        self.image.c as usize
    }

    /// Image shape
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.width(), self.height(), self.channels())
    }
}

impl Clone for Image {
    /// Full copy of image
    fn clone(&self) -> Image {
        let sys::image { w, h, c, .. } = self.image;
        let image = Self::zeros(w as usize, h as usize, c as usize);
        let from_slice = self.get_data();
        let to_slice = image.get_data_mut();
        to_slice.copy_from_slice(from_slice);
        image
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe { sys::free_image(self.image) }
    }
}

impl<'a> From<&'a DynamicImage> for Image {
    fn from(from: &'a DynamicImage) -> Self {
        match from {
            DynamicImage::ImageLuma8(image) => image.into(),
            DynamicImage::ImageLumaA8(image) => image.into(),
            DynamicImage::ImageRgb8(image) => image.into(),
            DynamicImage::ImageRgba8(image) => image.into(),
            DynamicImage::ImageBgr8(image) => image.into(),
            DynamicImage::ImageBgra8(image) => image.into(),
            DynamicImage::ImageLuma16(image) => image.into(),
            DynamicImage::ImageLumaA16(image) => image.into(),
            DynamicImage::ImageRgb16(image) => image.into(),
            DynamicImage::ImageRgba16(image) => image.into(),
        }
    }
}

impl From<DynamicImage> for Image {
    fn from(from: DynamicImage) -> Self {
        (&from).into()
    }
}

impl<'a, P, Container> From<&'a ImageBuffer<P, Container>> for Image
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]>,
    f32: From<P::Subpixel>,
{
    fn from(buffer: &ImageBuffer<P, Container>) -> Self {
        let w = buffer.width() as usize;
        let h = buffer.height() as usize;
        let c = P::CHANNEL_COUNT as usize;
        let n_components = w * h * c;

        let image = unsafe { sys::make_image(w as i32, h as i32, c as i32) };
        let slice = unsafe { slice::from_raw_parts_mut(image.data, n_components) };

        buffer
            .enumerate_pixels()
            .flat_map(|(x, y, pixel)| {
                pixel
                    .channels()
                    .iter()
                    .cloned()
                    .enumerate()
                    .map(move |(c, component)| (x, y, c, component))
            })
            .map(|(x, y, c, component)| {
                let converted = f32::try_from(component).ok().unwrap();
                (x as usize, y as usize, c, converted)
            })
            .for_each(|(x, y, c, component)| {
                let index = c * h * w + y * w + x;
                slice[index] = component;
            });

        Self { image }
    }
}

impl<P, Container> From<ImageBuffer<P, Container>> for Image
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]>,
    f32: From<P::Subpixel>,
{
    fn from(buffer: ImageBuffer<P, Container>) -> Self {
        (&buffer).into()
    }
}
