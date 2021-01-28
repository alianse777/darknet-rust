# darknet-rust: A Rust bindings for AlexeyAB's Darknet

[![Crates.io](https://img.shields.io/crates/v/darknet?style=for-the-badge)](https://crates.io/crates/darknet) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/alianse777/darknet-rust/Rust?style=for-the-badge)

The crate is a Rust wrapper for [AlexeyAB's Darknet](https://github.com/AlexeyAB/darknet).

It provides the following features:

- Provide both training and inference capabilities.
- Load config files and model weights from upstream without modifications.
- Safe type wrappers for C API. It includes network, detection and layer types.

Minimal rustc version: 1.43.0

## Version 0.3 changes:

- Error handling with anyhow

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
darknet = "0.3.2"
```

We suggest earlier users update to newer version from 0.1. There are several memory leakage and several bugs fixed.

## Cargo Features

- `enable-cuda`: Enable CUDA (expects CUDA 10.x and cuDNN 7.x).
- `enable-opencv`: Enable OpenCV.
- `enable-cudnn`: Enable cuDNN.
- `runtime`: Link to darknet dynamic library. For example, `libdark.so` on Linux.
- `buildtime-bindgen`: Generate bindings from darknet headers.
- `dylib`: Build dynamic library instead of static.

## License

The crate is licensed under MIT.

## Credits
Huge thanks to [jerry73204](https://github.com/jerry73204)
