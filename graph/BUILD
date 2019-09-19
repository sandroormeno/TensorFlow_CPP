load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

cc_binary(
    name = "graph_",
    srcs = ["loader.cc"],
    deps = [
        "//tensorflow/core:tensorflow",
		"//tensorflow/cc:cc_ops",
		"//tensorflow/cc:client_session",


    ],
	data = ["models/graph.pb"]
)