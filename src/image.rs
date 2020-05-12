use crate::{error::Error, BBox};
use darknet_sys as sys;
use image::{DynamicImage, ImageBuffer, Pixel};
use std::{
    borrow::{Borrow, Cow},
    ops::Deref,
    os::raw::c_int,
    path::Path,
    slice,
};

pub trait ConvertSubpixel
where
    Self: image::Primitive,
{
    fn from_subpixel(from: Self) -> f32;
    fn to_subpixel(from: f32) -> Self;
}

impl ConvertSubpixel for u8 {
    fn from_subpixel(from: Self) -> f32 {
        from as f32 / std::u8::MAX as f32
    }

    fn to_subpixel(from: f32) -> Self {
        (from * std::u8::MAX as f32) as u8
    }
}

impl ConvertSubpixel for u16 {
    fn from_subpixel(from: Self) -> f32 {
        from as f32 / std::u16::MAX as f32
    }

    fn to_subpixel(from: f32) -> Self {
        (from * std::u16::MAX as f32) as u16
    }
}

impl ConvertSubpixel for u32 {
    fn from_subpixel(from: Self) -> f32 {
        from as f32 / std::u32::MAX as f32
    }

    fn to_subpixel(from: f32) -> Self {
        (from * std::u32::MAX as f32) as u32
    }
}

impl ConvertSubpixel for u64 {
    fn from_subpixel(from: Self) -> f32 {
        from as f32 / std::u64::MAX as f32
    }

    fn to_subpixel(from: f32) -> Self {
        (from * std::u64::MAX as f32) as u64
    }
}

impl ConvertSubpixel for f32 {
    fn from_subpixel(from: Self) -> f32 {
        from
    }

    fn to_subpixel(from: f32) -> Self {
        from
    }
}

impl ConvertSubpixel for f64 {
    fn from_subpixel(from: Self) -> f32 {
        from as f32
    }

    fn to_subpixel(from: f32) -> Self {
        from as f64
    }
}

/// The image type used by darknet.
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

    /// Open image from a file.
    pub fn open<P: AsRef<Path>>(filename: P) -> Result<Self, Error> {
        let image: Self = image::open(filename)?.into();
        Ok(image)
    }

    /// Resize the image without keeping the ratio.
    pub fn resize(&self, w: usize, h: usize) -> Self {
        let image = unsafe { sys::resize_image(self.image, w as c_int, h as c_int) };
        Image { image }
    }

    /// Resize the image while keeping the ratio.
    pub fn letter_box(&self, w: usize, h: usize) -> Self {
        let image = unsafe { sys::letterbox_image(self.image, w as c_int, h as c_int) };
        Image { image }
    }

    /// Crop a bounding box from the image.
    pub fn crop_bbox<B>(&self, bbox: B) -> Image
    where
        B: Borrow<BBox>,
    {
        let BBox { x, y, w, h } = *bbox.borrow();
        let image_width = self.width() as f32;
        let image_height = self.height() as f32;

        let left = (x - w / 2.0) * image_width;
        let top = (y - h / 2.0) * image_height;
        let width = w * image_width;
        let height = h * image_height;
        unsafe {
            Image {
                image: sys::crop_image(
                    self.image,
                    left as c_int,
                    top as c_int,
                    width as c_int,
                    height as c_int,
                ),
            }
        }
    }

    /// Returns pointer to raw image data.
    pub unsafe fn get_raw_data(&self) -> *mut f32 {
        self.image.data
    }

    /// Returns pixel values as a slice.
    pub fn get_data<'a>(&'a self) -> &'a [f32] {
        return unsafe {
            slice::from_raw_parts(
                self.image.data,
                (self.image.h * self.image.w * self.image.c) as usize,
            )
        };
    }

    /// Returns pixel values as a mutable slice.
    pub fn get_data_mut<'a>(&'a self) -> &'a mut [f32] {
        return unsafe {
            slice::from_raw_parts_mut(
                self.image.data,
                (self.image.h * self.image.w * self.image.c) as usize,
            )
        };
    }

    /// Get the image width.
    pub fn width(&self) -> usize {
        self.image.w as usize
    }

    /// Get the image height.
    pub fn height(&self) -> usize {
        self.image.h as usize
    }

    /// Get the image channels.
    pub fn channels(&self) -> usize {
        self.image.c as usize
    }

    /// Get the image shape tuple (channels, height, width).
    pub fn shape(&self) -> (usize, usize, usize) {
        (self.channels(), self.height(), self.width())
    }

    /// Convert Image to ImageBuffer from 'image' crate
    pub fn to_image_buffer<P>(&self) -> Result<ImageBuffer<P, Vec<P::Subpixel>>, Error>
    where
        P: Pixel + 'static,
        P::Subpixel: 'static,
        P::Subpixel: ConvertSubpixel,
    {
        let (channels, height, width) = self.shape();
        if channels != P::CHANNEL_COUNT as usize {
            return Err(Error::ConversionError {
                reason: format!(
                    "cannot convert to a {} channel ImageBuffer from Image with {} channels",
                    P::CHANNEL_COUNT,
                    channels
                ),
            });
        }

        let mut image = ImageBuffer::<P, Vec<P::Subpixel>>::new(width as u32, height as u32);
        image.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
            pixel
                .channels_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(c, subpixel)| {
                    let value =
                        self.get_data()[c * height * width + y as usize * width + x as usize];
                    *subpixel = ConvertSubpixel::to_subpixel(value);
                });
        });

        Ok(image)
    }
}

impl Clone for Image {
    /// Make a deep-copy of the image.
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

unsafe impl Send for Image {}

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
    P::Subpixel: ConvertSubpixel,
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
                    .map(move |(c, subpixel)| (x, y, c, subpixel))
            })
            .map(|(x, y, c, subpixel)| {
                let converted = ConvertSubpixel::from_subpixel(subpixel);
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
    P::Subpixel: ConvertSubpixel,
{
    fn from(buffer: ImageBuffer<P, Container>) -> Self {
        (&buffer).into()
    }
}

// note: only traits defined in the current crate can be implemented for a type parameter (with rustc 1.40.0)
/*impl<P> TryFrom<&Image> for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    P::Subpixel: ConvertSubpixel,
{
    type Error = Error;

    fn try_from(from: &Image) -> Result<Self, Self::Error> {
        let (channels, height, width) = from.shape();
        if channels != P::CHANNEL_COUNT as usize {
            return Err(Error::ConversionError {
                reason: format!(
                    "cannot convert to a {} channel ImageBuffer from Image with {} channels",
                    P::CHANNEL_COUNT,
                    channels
                ),
            });
        }

        let mut image = ImageBuffer::<P, Vec<P::Subpixel>>::new(width as u32, height as u32);
        image.enumerate_pixels_mut().for_each(|(x, y, pixel)| {
            pixel
                .channels_mut()
                .iter_mut()
                .enumerate()
                .for_each(|(c, subpixel)| {
                    let value =
                        from.get_data()[c * height * width + y as usize * width + x as usize];
                    *subpixel = ConvertSubpixel::to_subpixel(value);
                });
        });

        Ok(image)
    }
}

impl<P> TryFrom<Image> for ImageBuffer<P, Vec<P::Subpixel>>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    P::Subpixel: ConvertSubpixel,
{
    type Error = Error;

    fn try_from(from: Image) -> Result<Self, Self::Error> {
        Self::try_from(&from)
    }
}*/

/// The traits converts input type to a copy-on-write image.
pub trait IntoCowImage<'a> {
    fn into_cow_image(self) -> Cow<'a, Image>;
}

impl<'a> IntoCowImage<'a> for Image {
    fn into_cow_image(self) -> Cow<'a, Image> {
        Cow::Owned(self)
    }
}

impl<'a> IntoCowImage<'a> for &'a Image {
    fn into_cow_image(self) -> Cow<'a, Image> {
        Cow::Borrowed(self)
    }
}

impl<'a, P, Container> IntoCowImage<'a> for &'a ImageBuffer<P, Container>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]>,
    P::Subpixel: ConvertSubpixel,
{
    fn into_cow_image(self) -> Cow<'a, Image> {
        Cow::Owned(self.into())
    }
}

impl<'a, P, Container> IntoCowImage<'a> for ImageBuffer<P, Container>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]>,
    P::Subpixel: ConvertSubpixel,
{
    fn into_cow_image(self) -> Cow<'a, Image> {
        Cow::Owned(self.into())
    }
}
