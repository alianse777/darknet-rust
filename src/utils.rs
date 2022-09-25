use std::os::unix::ffi::OsStrExt;
use std::{ffi::CString, path::Path};
use crate::Error;

#[cfg(unix)]
pub fn path_to_cstring(path: &Path) -> Option<CString> {
    CString::new(path.as_os_str().as_bytes()).ok()
}

pub fn path_to_cstring_or_error(path: &Path) -> Result<CString, Error> {
    path_to_cstring(path).ok_or_else(|| Error::EncodingError {
        reason: format!("the path {:?} is invalid", path),
    })
}

#[cfg(not(unix))]
pub fn path_to_cstring(path: &Path) -> Option<CString> {
    path.to_str()
        .map(|s| CString::new(s.as_bytes()).ok())
        .flatten()
}
