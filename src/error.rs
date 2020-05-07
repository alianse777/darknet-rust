use failure::Fail;
use image::error::ImageError;

#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "image error: {:?}", _0)]
    ImageError(ImageError),
}

impl From<ImageError> for Error {
    fn from(err: ImageError) -> Self {
        Self::ImageError(err)
    }
}
