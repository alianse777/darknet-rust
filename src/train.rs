use crate::{error::Error, utils};
use darknet_sys as sys;
use std::{mem, os::raw::c_int, path::Path, ptr};

/// Train a detector model.
pub fn train_detector<P1, P2, P3, P4, G>(
    data_config_file: P1,
    model_config_file: P2,
    weights_file: Option<P3>,
    gpu_indexes: G,
    clear: bool,
    dont_show: bool,
    calc_map: bool,
    mjpeg_port: Option<u16>,
    show_imgs: bool,
    benchmark_layers: bool,
    chart_file: P4,
    thresh: f32,
    iou_thresh: f32,
) -> Result<(), Error>
where
    P1: AsRef<Path>,
    P2: AsRef<Path>,
    P3: AsRef<Path>,
    P4: AsRef<Path>,
    G: AsRef<[usize]>,
{
    let data_config_ctring = utils::path_to_cstring_or_error(data_config_file.as_ref())?;
    let model_config_ctring = utils::path_to_cstring_or_error(model_config_file.as_ref())?;
    let weights_ctring = weights_file
        .map(|path| utils::path_to_cstring_or_error(path.as_ref()))
        .transpose()?;
    let chart_cstring = utils::path_to_cstring_or_error(chart_file.as_ref())?;
    let gpu_indexes_c_int = gpu_indexes
        .as_ref()
        .iter()
        .cloned()
        .map(|index| index as c_int)
        .collect::<Vec<_>>();

    unsafe {
        let data_config_ptr = data_config_ctring.as_ptr() as *mut _;
        let model_config_ptr = model_config_ctring.as_ptr() as *mut _;
        let chart_ptr = chart_cstring.as_ptr() as *mut _;
        let weights_ptr = weights_ctring
            .as_ref()
            .map(|cstring| cstring.as_ptr() as *mut _)
            .unwrap_or(ptr::null_mut());
        let gpu_indexes_ptr = gpu_indexes_c_int.as_ptr() as *mut _;
        let num_gpus = gpu_indexes_c_int.len();

        sys::train_detector(
            data_config_ptr,
            model_config_ptr,
            weights_ptr,
            gpu_indexes_ptr,
            num_gpus as c_int,
            clear as c_int,
            dont_show as c_int,
            calc_map as c_int,
            thresh,
            iou_thresh,
            mjpeg_port.map(|port| port as c_int).unwrap_or(-1),
            show_imgs as c_int,
            benchmark_layers as c_int,
            chart_ptr,
        );
    }

    mem::drop(data_config_ctring);
    mem::drop(model_config_ctring);
    mem::drop(weights_ctring);
    mem::drop(chart_cstring);
    mem::drop(gpu_indexes_c_int);

    Ok(())
}
