use darknet::{load_labels, Image, Network, IMTYPE_JPG};
use std::fs;

fn main() {
    // Load network & labels
    let object_labels = load_labels("./darknet/data/coco.names").unwrap();
    let mut net = Network::load(
        "./darknet/cfg/yolov3-tiny.cfg",
        Some("./yolov3-tiny.weights"),
        false,
        object_labels.clone(),
    )
    .unwrap();
    let mut img = Image::open("./darknet/data/person.jpg");
    // Run object detection
    let detections = net.predict(&mut img, 0.45, 0.3);
    // Print which objects where found
    println!("Found: {:?}", detections.get_labels());
    // Save detected objects as separate images
    fs::create_dir("./result");
    for (label, obj) in detections.crop_from(&img) {
        obj.save_image_options(&format!("./result/{}", label), IMTYPE_JPG);
    }
    // Annotate image with object labels and bboxes
    detections.draw_on_image(&mut img);
    img.show_image("IMG");
}
