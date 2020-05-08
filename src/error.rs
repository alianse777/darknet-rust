use failure::Fail;
use image::error::ImageError;
use std::io;

/// The error type for this crate.
#[derive(Debug, Fail)]
pub enum Error {
    #[fail(display = "image error: {:?}", _0)]
    ImageError(ImageError),
    #[fail(display = "I/O error: {:?}", _0)]
    IoError(io::Error),
    #[fail(display = "encoding error: {:?}", reason)]
    EncodingError { reason: String },
    #[fail(display = "internal error: {:?}", reason)]
    InternalError { reason: String },
}

impl From<ImageError> for Error {
    fn from(err: ImageError) -> Self {
        Self::ImageError(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Self::IoError(err)
    }
}
