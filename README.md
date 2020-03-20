A Rust wrapper for [Darknet](https://pjreddie.com/darknet/),  an open source neural network framework written in C and CUDA.

Currently lacks training functionality as it usually done in python. (PRs are welcome)

Features:
- [image](https://crates.io/crates/image) crate integration.

Put 'data' directory with your project if you plan to use Detections::draw_on_image method.

Example:

```rust
use darknet::{load_labels, Image, Network};
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
    let mut img = Image::open("./darknet/data/person.jpg").unwrap();
    // Run object detection
    let detections = net.predict(&mut img, 0.45, 0.3);
    // Print which objects where found
    println!("Found: {:?}", detections.get_labels());
    // Save detected objects as separate images
    fs::create_dir("./result");
    for (label, obj) in detections.crop_from(&img) {
        obj.save(&format!("./result/{}.jpg", label)).unwrap();
    }
    // Annotate image with object labels and bboxes
    detections.draw_on_image(&mut img);
    img.show("IMG");
}
```
