use darknet_sys as sys;
use num_derive::FromPrimitive;

/// Layer types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum LayerType {
    Gru = sys::LAYER_TYPE_GRU as usize,
    Rnn = sys::LAYER_TYPE_RNN as usize,
    Sam = sys::LAYER_TYPE_SAM as usize,
    Cost = sys::LAYER_TYPE_COST as usize,
    Crnn = sys::LAYER_TYPE_CRNN as usize,
    Crop = sys::LAYER_TYPE_CROP as usize,
    Iseg = sys::LAYER_TYPE_ISEG as usize,
    Lstm = sys::LAYER_TYPE_LSTM as usize,
    Xnor = sys::LAYER_TYPE_XNOR as usize,
    Yolo = sys::LAYER_TYPE_YOLO as usize,
    Blank = sys::LAYER_TYPE_BLANK as usize,
    Empty = sys::LAYER_TYPE_EMPTY as usize,
    Local = sys::LAYER_TYPE_LOCAL as usize,
    Reorg = sys::LAYER_TYPE_REORG as usize,
    Route = sys::LAYER_TYPE_ROUTE as usize,
    Active = sys::LAYER_TYPE_ACTIVE as usize,
    L2Norm = sys::LAYER_TYPE_L2NORM as usize,
    Region = sys::LAYER_TYPE_REGION as usize,
    Avgpool = sys::LAYER_TYPE_AVGPOOL as usize,
    Dropout = sys::LAYER_TYPE_DROPOUT as usize,
    Logxent = sys::LAYER_TYPE_LOGXENT as usize,
    Maxpool = sys::LAYER_TYPE_MAXPOOL as usize,
    Network = sys::LAYER_TYPE_NETWORK as usize,
    Softmax = sys::LAYER_TYPE_SOFTMAX as usize,
    Shortcut = sys::LAYER_TYPE_SHORTCUT as usize,
    Upsample = sys::LAYER_TYPE_UPSAMPLE as usize,
    Batchnorm = sys::LAYER_TYPE_BATCHNORM as usize,
    Connected = sys::LAYER_TYPE_CONNECTED as usize,
    ConvLstm = sys::LAYER_TYPE_CONV_LSTM as usize,
    Detection = sys::LAYER_TYPE_DETECTION as usize,
    ReorgOld = sys::LAYER_TYPE_REORG_OLD as usize,
    Convolutional = sys::LAYER_TYPE_CONVOLUTIONAL as usize,
    GaussianYolo = sys::LAYER_TYPE_GAUSSIAN_YOLO as usize,
    LocalAvgpool = sys::LAYER_TYPE_LOCAL_AVGPOOL as usize,
    Normalization = sys::LAYER_TYPE_NORMALIZATION as usize,
    ScaleChannels = sys::LAYER_TYPE_SCALE_CHANNELS as usize,
    Deconvolutional = sys::LAYER_TYPE_DECONVOLUTIONAL as usize,
}

/// Activation types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum Activation {
    Elu = sys::ACTIVATION_ELU as usize,
    Gelu = sys::ACTIVATION_GELU as usize,
    Mish = sys::ACTIVATION_MISH as usize,
    Plse = sys::ACTIVATION_PLSE as usize,
    Ramp = sys::ACTIVATION_RAMP as usize,
    Relu = sys::ACTIVATION_RELU as usize,
    Selu = sys::ACTIVATION_SELU as usize,
    Tanh = sys::ACTIVATION_TANH as usize,
    Leaky = sys::ACTIVATION_LEAKY as usize,
    Lhtan = sys::ACTIVATION_LHTAN as usize,
    Loggy = sys::ACTIVATION_LOGGY as usize,
    Relie = sys::ACTIVATION_RELIE as usize,
    Relu6 = sys::ACTIVATION_RELU6 as usize,
    Stair = sys::ACTIVATION_STAIR as usize,
    Swish = sys::ACTIVATION_SWISH as usize,
    Linear = sys::ACTIVATION_LINEAR as usize,
    Hardtan = sys::ACTIVATION_HARDTAN as usize,
    Logistic = sys::ACTIVATION_LOGISTIC as usize,
    NormChan = sys::ACTIVATION_NORM_CHAN as usize,
    NormChanSoftmax = sys::ACTIVATION_NORM_CHAN_SOFTMAX as usize,
    NormChanSoftmaxMaxval = sys::ACTIVATION_NORM_CHAN_SOFTMAX_MAXVAL as usize,
}

/// Binary activation types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum BinaryActivation {
    Add = sys::BINARY_ACTIVATION_ADD as usize,
    Div = sys::BINARY_ACTIVATION_DIV as usize,
    Sub = sys::BINARY_ACTIVATION_SUB as usize,
    Mult = sys::BINARY_ACTIVATION_MULT as usize,
}

/// Cost, or namely, loss types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum CostType {
    L1 = sys::COST_TYPE_L1 as usize,
    Seg = sys::COST_TYPE_SEG as usize,
    Sse = sys::COST_TYPE_SSE as usize,
    Wgan = sys::COST_TYPE_WGAN as usize,
    Masked = sys::COST_TYPE_MASKED as usize,
    Smooth = sys::COST_TYPE_SMOOTH as usize,
}

/// Weights format types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum WeightsType {
    NoWeights = sys::WEIGHTS_TYPE_T_NO_WEIGHTS as usize,
    PerChannel = sys::WEIGHTS_TYPE_T_PER_CHANNEL as usize,
    PerFeature = sys::WEIGHTS_TYPE_T_PER_FEATURE as usize,
}

/// Weights normalizion types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum WeightsNormalizion {
    None = sys::WEIGHTS_NORMALIZATION_T_NO_NORMALIZATION as usize,
    Relu = sys::WEIGHTS_NORMALIZATION_T_RELU_NORMALIZATION as usize,
    Softmax = sys::WEIGHTS_NORMALIZATION_T_SOFTMAX_NORMALIZATION as usize,
}

/// Non-Maximum Suppression (NMS) types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum NmsKind {
    Diou = sys::NMS_KIND_DIOU_NMS as usize,
    Greedy = sys::NMS_KIND_GREEDY_NMS as usize,
    Corners = sys::NMS_KIND_CORNERS_NMS as usize,
    Default = sys::NMS_KIND_DEFAULT_NMS as usize,
}

/// IoU loss types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum IoULoss {
    IoU = sys::IOU_LOSS_IOU as usize,
    Mse = sys::IOU_LOSS_MSE as usize,
    CIoU = sys::IOU_LOSS_CIOU as usize,
    DIoU = sys::IOU_LOSS_DIOU as usize,
    GIoU = sys::IOU_LOSS_GIOU as usize,
}

/// YOLO point types.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, FromPrimitive)]
pub enum YoloPoint {
    Center = sys::YOLO_POINT_YOLO_CENTER as usize,
    LeftTop = sys::YOLO_POINT_YOLO_LEFT_TOP as usize,
    RightBottom = sys::YOLO_POINT_YOLO_RIGHT_BOTTOM as usize,
}
