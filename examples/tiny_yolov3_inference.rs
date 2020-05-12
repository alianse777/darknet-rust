use darknet::{BBox, Image, Network};
use failure::Fallible;
use image::{Rgb, RgbImage};
use sha2::{Digest, Sha256};
use std::{
    convert::TryFrom,
    fs::{self, File},
    io::{prelude::*, BufReader, BufWriter},
    path::Path,
};

const LABEL_PATH: &'static str = "./darknet/data/coco.names";
const IMAGE_PATH: &'static str = "./darknet/data/person.jpg";
const CFG_PATH: &'static str = "./darknet/cfg/yolov3-tiny.cfg";
const WEIGHTS_URL: &'static str = "https://pjreddie.com/media/files/yolov3-tiny.weights";
const WEIGHTS_SHA256_HASH: &'static str =
    "dccea06f59b781ec1234ddf8d1e94b9519a97f4245748a7d4db75d5b7080a42c";
const WEIGHTS_FILE_NAME: &'static str = "./yolov3-tiny.weights";
const OUTPUT_DIR: &'static str = "./output";
const OBJECTNESS_THRESHOLD: f32 = 0.9;
const CLASS_PROB_THRESHOLD: f32 = 0.9;

fn main() -> Fallible<()> {
    // download weights file
    fs::create_dir_all(OUTPUT_DIR)?;
    let weights_path = Path::new(OUTPUT_DIR).join(WEIGHTS_FILE_NAME);

    if !weights_path.exists() {
        println!("Downloading weights file ...");
        let mut writer = BufWriter::new(File::create(&weights_path)?);
        reqwest::blocking::get(WEIGHTS_URL)?.copy_to(&mut writer)?;
    }

    // verify weights file
    {
        let mut reader = BufReader::new(File::open(&weights_path)?);
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;
        let digest = Sha256::digest(&buf);
        assert_eq!(
            digest[..],
            hex::decode(WEIGHTS_SHA256_HASH)?[..],
            "the weights file {} is corrupted. Please remove it before running the example.",
            weights_path.display()
        );
    }

    // Load network & labels
    let object_labels = std::fs::read_to_string(LABEL_PATH)?
        .lines()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut net = Network::load(CFG_PATH, Some(weights_path), false)?;

    // Run object detection
    let image = Image::open(IMAGE_PATH)?;
    let detections = net.predict(&image, 0.25, 0.5, 0.45, true);

    // show results
    detections
        .iter()
        .filter(|det| det.objectness() > OBJECTNESS_THRESHOLD)
        .flat_map(|det| {
            det.best_class(Some(CLASS_PROB_THRESHOLD))
                .map(|(class_index, prob)| (det, prob, &object_labels[class_index]))
        })
        .enumerate()
        .for_each(|(index, (det, prob, label))| {
            let bbox = det.bbox();
            let BBox { x, y, w, h } = bbox;

            // Save image
            let image_path =
                Path::new(OUTPUT_DIR).join(format!("{}-{}-{:2.2}.jpg", index, label, prob * 100.0));
            image
                .crop_bbox(bbox)
                .to_image_buffer::<Rgb<u8>>()
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

    Ok(())
}
