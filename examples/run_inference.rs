use anyhow::Result;
use argh::FromArgs;
use darknet::{BBox, Image, Network};
use image::RgbImage;
use std::{
    convert::TryFrom,
    fs::{self},
    path::PathBuf,
};

/// The inference example.
#[derive(Debug, Clone, FromArgs)]
struct Args {
    /// the file including label names per class.
    #[argh(option)]
    label_file: PathBuf,
    /// the model config file, which usually has a .cfg extension.
    #[argh(option)]
    model_cfg: PathBuf,
    /// the model weights file, which usually has a .weights extension.
    #[argh(option)]
    weights: PathBuf,
    /// the output directory.
    #[argh(option, default = "PathBuf::from(\"./output\")")]
    output_dir: PathBuf,
    /// the objectness threshold.
    #[argh(option, default = "0.9")]
    objectness_threshold: f32,
    /// the class probability threshold.
    #[argh(option, default = "0.9")]
    class_prob_threshold: f32,
    /// input image files.
    #[argh(positional)]
    input_images: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let Args {
        label_file,
        model_cfg,
        weights,
        output_dir,
        objectness_threshold,
        class_prob_threshold,
        input_images,
    } = argh::from_env();

    // Load network & labels
    let object_labels = std::fs::read_to_string(label_file)?
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut net = Network::load(model_cfg, Some(weights), false)?;

    for image_path in input_images {
        // prepare data
        let image = Image::open(&image_path)?;
        let image_file_name = image_path
            .file_name()
            .expect(&format!("{} is not a valid file", image_path.display()));
        let curr_output_dir = output_dir.join(image_file_name);
        fs::create_dir_all(&curr_output_dir)?;

        // Run object detection
        let detections = net.predict(&image, 0.25, 0.5, 0.45, true);

        // show results
        println!("# {}", image_path.display());

        detections
            .iter()
            .filter(|det| det.objectness() > objectness_threshold)
            .flat_map(|det| {
                det.best_class(Some(class_prob_threshold))
                    .map(|(class_index, prob)| (det, prob, &object_labels[class_index]))
            })
            .enumerate()
            .for_each(|(index, (det, prob, label))| {
                let bbox = det.bbox();
                let BBox { x, y, w, h } = bbox;

                // Save image
                let image_path =
                    curr_output_dir.join(format!("{}-{}-{:2.2}.jpg", index, label, prob * 100.0));
                RgbImage::try_from(image.crop_bbox(bbox))
                    .unwrap()
                    .save(image_path)
                    .unwrap();

                // print result
                println!(
                    "{}\t{:.2}%\tx: {}\ty: {}\tw: {}\th: {}",
                    label,
                    prob * 100.0,
                    x,
                    y,
                    w,
                    h
                );
            });
    }
    Ok(())
}
