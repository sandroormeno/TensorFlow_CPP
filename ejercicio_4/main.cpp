#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

using namespace tensorflow;

int main(int argc, char* argv[]) {

	//std::string graph_definition = "mlp.pb";
	std::string graph_definition = "sandro.pb";
	Session* session;
	GraphDef graph_def;
	SessionOptions opts;
	std::vector<Tensor> outputs; // Store outputs
	std::vector<tensorflow::Tensor> salida;
	std::vector<tensorflow::Tensor> prediccion;

	
	TF_CHECK_OK(ReadBinaryProto(Env::Default(), graph_definition, &graph_def));

	// Set GPU options
	//graph::SetDefaultDevice("/gpu:0", &graph_def);
	//opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
	//opts.config.mutable_gpu_options()->set_allow_growth(true);

	// create a new session
	TF_CHECK_OK(NewSession(opts, &session));

	// Load graph into session
	TF_CHECK_OK(session->Create(graph_def));

	// Initialize our variables
	TF_CHECK_OK(session->Run({}, {}, {"init_all_vars_op"}, nullptr));

	Tensor x(DT_FLOAT, TensorShape({100, 32}));
	Tensor y(DT_FLOAT, TensorShape({100, 8}));
	auto _XTensor = x.matrix<float>();
	auto _YTensor = y.matrix<float>();

	_XTensor.setRandom();
	_YTensor.setRandom();
	
	

	for (int i = 0; i < 40000; ++i) {
		TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"cost"}, {}, &outputs));
		float cost = outputs[0].scalar<float>()(0);
		
		//TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"aprendizaje"}, &acc));
		//std::cout << acc[0].DebugString() << std::endl;
		TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {}, {"train"}, nullptr));
		
		//muy bien esta parte ---> y_out = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a, w2), b2), name="y_out")
		//TF_CHECK_OK(session->Run({{"x", x}}, {"y_out"}, {}, &salida));		
		//std::cout << salida[0].DebugString() << "\n";

		TF_CHECK_OK(session->Run({{"x", x}}, {"y_out"}, {}, &salida));
		TF_CHECK_OK(session->Run({{"x", x}, {"y", y}}, {"prediccion"}, {}, &prediccion));

			
		if(i % 4000 == 0){
			//std::cout << salida[0].DebugString() << "\n";
			
			//std::cout << salida[0].matrix<float>() << "\n";
			//std::cout << salida[0].matrix<float>()(0) << "\n";
			//std::cout << " Cost: " <<  cost <<  "  Acc:  " << salida <<  std::endl;
			//std::cout << "Step: " << i << " Cost: " <<  cost <<  " Acc:" << salida[0].scalar<float>()(0) << std::endl;
			std::cout << "Step: " << i << " Cost: " <<  cost <<  " Acc:"  << std::endl;

		}
		
		
		//acc.clear();
		outputs.clear();
	}
	
	
	std::cout.precision(1);
	std::cout << y.DebugString()  << "\n";
	std::cout << salida[0].DebugString()  << "\n";
	//std::cout << y.matrix<float>() (32)<< "\n";
	//std::cout << salida[0].matrix<float>() (32)<< "\n";
    //std::cout << prediccion[0].DebugString()<< "\n";
	std::cout << "------------------------------------------------------------" << "\n";
	for (int i = 0; i < 10; ++i) {
		int r = (rand() % 100) + 1;
		std::cout << "Indice: "<< r << "  valor real:  " << y.matrix<float>() (r) << "   valor predicho:  " << salida[0].matrix<float>() (r) <<  "\n";
	}
	std::cout << "------------------------------------------------------------" << "\n";
	session->Close();
	delete session;
	return 0;
}
