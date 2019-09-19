#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace tensorflow;
using namespace std;

// importante usar : "bazel run -c opt --config=monolithic //tensorflow/cc/load_graph:loader"
// importante usar : "g++ -std=c++11 -Wl,-rpath='$ORIGIN/lib' -Iinclude -Llib proyectos/ejercicio_1/load.cc -ltensorflow_cc -o load"
// https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f

vector<float> ReadCSVLine(string line) {
  vector<float> line_data;
  stringstream lineStream(line);
  string cell;
  while(std::getline(lineStream, cell, ','))
  {
    line_data.push_back(stod(cell));
  }
  return line_data;
}

vector<float> leer_file(string dir){
  // x es el vector que gurardarpá dos datos de archivo CSV
  vector<float> x;
  
  
  // leeo el archivo
  //ifstream file("proyectos/ejercicio_1/datos.csv");
  ifstream file;
/*   if (!file) {
    file.open("proyectos/ejercicio_1/datos.csv");
  } */
  // file.open("proyectos/ejercicio_1/datos.csv");
  file.open(dir);
  stringstream buffer;
  buffer << file.rdbuf();
  string line;
  vector<string> lines;
  while(getline(buffer, line, '\n')) {
    lines.push_back(line);
  }
  
  //std::cout << lines.size() << "\n";
  int n_lines = lines.size();
  for (int i = 0; i < n_lines; ++i) {
    //std::cout << lines[i] << "\n";
    vector<float> features = ReadCSVLine(lines[i]);
    x.insert(x.end(), features.begin(), features.begin() + 2);
  }
  
  //std::cout << x << "\n";
  return x;
  
  //
			
}

int main(int argc, char* argv[]) {
	
  //std::cout << features << "\n";
  // Initialize a tensorflow session
  Session* session;
  Status status = NewSession(SessionOptions(), &session);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Read in the protobuf graph we exported
  // (The path seems to be relative to the cwd. Keep this in mind
  // when using `bazel run` since the cwd isn't where you call
  // `bazel run` but from inside a temp folder.)
  GraphDef graph_def;
  status = ReadBinaryProto(Env::Default(), "models/graph.pb", &graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Add the graph to the session
  status = session->Create(graph_def);
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }
  
  vector<float> DataSet = leer_file("proyectos/ejercicio_2/datos.csv");
  Tensor x(DT_FLOAT, TensorShape{2, 2});
  copy_n(DataSet.begin(), DataSet.size(), x.flat<float>().data());
  std::cout << x.DebugString() << "\n"; // para saber cómo es este vector??
  std::cout << x.matrix<float>() << "\n";
  
  
  
  vector<float> yo = leer_file("proyectos/ejercicio_2/yo.csv");
  Tensor s(DT_FLOAT, TensorShape{2, 2});
  copy_n(yo.begin(), yo.size(), s.flat<float>().data());
  std::cout << s.DebugString() << "\n"; // para saber cómo es este vector??
  std::cout << s.matrix<float>() << "\n";

 
  // Setup inputs and outputs:

  // Our graph doesn't require any inputs, since it specifies default values,
  // but we'll change an input to demonstrate.
  
  Tensor a(DT_FLOAT, TensorShape());
  a.scalar<float>()() = 5;
  Tensor b(DT_FLOAT, TensorShape());
  b.scalar<float>()() = 2;
  
  //para el tensor u
  int d[2][2] = {2,1,3,5};
  
  // std::cout << d[1][1] << "\n";
  
  Tensor u(DT_FLOAT, TensorShape({2,2}));
  auto dst = u.tensor<float,2>();
  //auto dst = u.flat<float>(),data();
  //auto dst = u.matrix<float>();
  for (int i = 0; i < 2; ++i) {
	for(int j = 0; j < 2; ++j) {
        dst(i, j) = d[i][j];
	}
  }
  
  // para el tensor i
  int e[2][2] = {20,1,19,68};
  
  // std::cout << d[1][1] << "\n";
  
  Tensor i(DT_FLOAT, TensorShape({2,2}));
  //auto temp1 = i.flat<float>(),data();
  auto temp1 = i.tensor<float,2>();
  //auto temp1 = i.matrix<float>();
  for (int ii = 0; ii < 2; ++ii) {
	for(int jj = 0; jj < 2; ++jj) {
        temp1(ii, jj) = e[ii][jj];
	}
  }
  
  //std::cout << u.matrix<float>() << "\n"; 
  //std::cout << i.matrix<float>() << "\n"; 
  
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
    { "a", a },
    { "b", b },
  };
  // esta parte también es importante: es el formato de entrada = string, Tensor
  std::vector<std::pair<string, tensorflow::Tensor>> entrada = {
    { "i", x},
    { "u", s},
  }; 

  // The session will initialize the outputs
  
  //std::vector<Tensor> outputs;
  //std::vector<tensorflow::Tensor> outputs;
  std::vector<tensorflow::Tensor> outputs2;

  // Run the session, evaluating our "c" operation from the graph
  //status = session->Run(inputs, {"s"}, {}, &outputs);
  //status = session->Run({{"a", a}, {"b", b}}, {"s"}, {}, &outputs);
  status = session->Run(entrada, {"mul"}, {}, &outputs2);
  
  if (!status.ok()) {
    std::cout << status.ToString() << "\n";
    return 1;
  }

  // Grab the first output (we only evaluated one graph node: "c")
  // and convert the node to a scalar representation.
  //auto output_s = outputs[0].<float>();
  //auto output_so = outputs2[0].scalar<float>();
  //auto output_so = outputs2[0].flat<float>().data()
  auto output_so = outputs2[0].flat<float>();
  //auto output_s = outputs[0].matrix<float>();

  // (There are similar methods for vectors and matrices here:
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/public/tensor.h)

  // Print the results
  std::cout << "---->>  outputs[0]: " << outputs2[0].DebugString() << "\n"; // Tensor<type: float shape: [] values: 30>
  //std::cout << "---->>  Valor: " <<  output_so() << "\n"; // 30
  std::cout << "---->>  valores planos: \n" <<  output_so << "\n"; // 30

  // Free any resources used by the session
  session->Close();
  return 0;
}