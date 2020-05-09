# darknet-rust: A Rust bindings for AlexeyAB's Darknet

The crate is a Rust wrapper for [AlexeyAB's Darknet](https://github.com/AlexeyAB/darknet).

It provides the following features:

- Provide both training and inference capabilities.
- Load config files and model weights from upstream without modifications.
- Safe type wrappers for C API. It includes network, detection and layer types.

## Examples

The **tiny_yolov3_inference** example automatically downloads the YOLOv3 tiny weights, and produces inference results in `output` directory.

```sh
cargo run --release --example tiny_yolov3_inference
```

The **run_inference** example is an utility program that you can test a combination of model configs and weights on image files. For example, you can test the YOLOv4 mode.

```sh
cargo run --release --example run_inference -- \
    --label-file darknet/data/coco.names \
    --model-cfg darknet/cfg/yolov4.cfg \
    --weights yolov4.weights \
    darknet/data/*.jpg
```

Read the example code in `examples/` to understand the actual usage. More model configs and weights can be found here: (https://pjreddie.com/darknet/yolo/).

## Usage

Add our crate to your `Cargo.toml`. You may take a look at the [API documentation](https://docs.rs/darknet).

```
darknet = "^0.2.0"
```

We suggest earlier users update to newer version from 0.1. There are several memory leakage and several bugs fixed.

## License

The crate is licensed under MIT.
