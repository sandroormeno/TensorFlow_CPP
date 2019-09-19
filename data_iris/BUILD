load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "lee_datos",
    srcs = ["iris.cc"],
    deps = [
        "//tensorflow/cc:gradients",
        "//tensorflow/cc:grad_ops",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow"
    ],
    data = ["iris.csv"]
)