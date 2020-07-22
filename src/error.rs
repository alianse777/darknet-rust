use image::error::ImageError;
use std::io;
use thiserror::Error;

/// The error type for this crate.
#[derive(Debug, Error)]
pub enum Error {
    #[error("image error: {0:?}")]
    ImageError(ImageError),
    #[error("I/O error: {0:?}")]
    IoError(io::Error),
    #[error("encoding error: {reason:?}")]
    EncodingError { reason: String },
    #[error("internal error: {reason:?}")]
    InternalError { reason: String },
    #[error("conversion error: {reason:?}")]
    ConversionError { reason: String },
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
