# darknet-rust: A Rust bindings for AlexeyAB's Darknet

[![Crates.io](https://img.shields.io/crates/v/darknet?style=for-the-badge)](https://crates.io/crates/darknet) ![GitHub Workflow Status](https://img.shields.io/github/workflow/status/alianse777/darknet-rust/Rust?style=for-the-badge)

The crate is a Rust wrapper for [AlexeyAB's Darknet](https://github.com/AlexeyAB/darknet).

It provides the following features:

- Training and inference capabilities.
- Load config files and model weights from upstream without modifications.
- Safe type wrappers for C API. Includes network, detection and layer types.

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

[API documentation](https://docs.rs/darknet)

If you are using version 0.1, consider migrating to 0.3 or newer as several critical bugs and memory leakages were fixed.

## Build

Terms used:

darknet-sys, darknet = Rust wrappers

libdarknet = C/C++ darknet implementation

By default, darknet will compile and link libdarknet statically. You can control the feature flags to change the behavior.

## Cargo Features

- `enable-cuda`: Enable CUDA (expects CUDA 10.x and cuDNN 7.x).
- `enable-cudnn`: Enable cuDNN
- `enable-opencv`: Enable OpenCV.
- `runtime`: Link to libdarknet dynamic library. For example, `libdark.so` on Linux.
- `dylib`: Build dynamic library instead of static
- `buildtime-bindgen`: Generate bindings from libdarknet headers.

### Method 1: Download and build from source (default)

```toml
[dependencies]
darknet = "0.3.2"
```
You can optionally enable CUDA and OpenCV features. Please read [Build with CUDA](#build-with-cuda) for more info.

```toml[dependencies]
[dependencies]
darknet = {version = "0.3.2", features = ["enable-cuda", "enable-opencv"] }
```
### Method 2: Build with custom source

If you want to build with custom libdarknet source, point `DARKNET_SRC` environment variable to your source path. It should contain `CMakeLists.txt`.

```sh
export DARKNET_SRC=/path/to/your/darknet/repo
```
### Method 3: Link to libdarknet dynamic library

With `runtime` feature, darknet-sys will not compile libdarknet source code and instead links to libdarknet dynamically. If you are using Linux, make sure `libdark.so` is installed on your system.

```toml
[dependencies]
darknet = {version = "0.3.2", features = ["runtime"] }
```
### Re-generate bindings

With `buildtime-bindgen` feature, darknet-sys re-generates bindings from headers. The option is necessary only when darkent is updated or modified.

```toml
[dependencies]
darknet = {version = "0.3.2", features = ["buildtime-bindgen"] }
```
If you want to use your (possibly modified) header files, point `DARKNET_INCLUDE_PATH` environment variable to your header dir.

### Build with CUDA

Please check that both CUDA 10.x and cuDNN 7.x are installed.

Darknet reads `CUDA_PATH` environment variable (which defaults to `/opt/cuda` if not set) and assumes it can find cuda libraries at `${CUDA_PATH}/lib64`.

```sh
export CUDA_PATH=/usr/local/cuda-10.1
```

```toml
[dependencies]
darknet = {version = "0.3.2", features = ["enable-cuda", "enable-opencv"] }
```
You can also set `CUDA_ARCHITECTURES` which is passed to libdarknet's cmake. It defaults to `Auto`, which auto-detects GPU architecture based on card present in the system during build.

## License

The crate is licensed under MIT.

## Credits
Huge thanks to [jerry73204](https://github.com/jerry73204)
