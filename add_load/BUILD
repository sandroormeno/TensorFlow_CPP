load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "suma_datos",
    srcs = ["suma.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
	data = ["datos.csv"]
)