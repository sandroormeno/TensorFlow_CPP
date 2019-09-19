#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

vector<int> ReadCSVLine(string line) {
  vector<int> line_data;
  stringstream lineStream(line);
  string cell;
  while(std::getline(lineStream, cell, ','))
  {
    line_data.push_back(stod(cell));
  }
  return line_data;
}



int main() {
	
  // este será el vector donde guardaré todo:
  vector<int> x_;
  
  // leeo el archivo
  ifstream file("datos.csv");
  if (!file) {
    file.open("tensorflow/cc/add_load/datos.csv");
  }
  //variable para leer las lineas
  stringstream buffer;
  buffer << file.rdbuf();
  string line;
  vector<string> lines;
  while(getline(buffer, line, '\n')) {
    lines.push_back(line);
  }
  //cout << lines[0] << endl;
	//leeco los campos, los valores de las celdas
  string valor;
  vector<int> line_data;
  for (int i = 0; i < lines.size(); ++i) {
	//stringstream lineStream(lines[i]);
    vector<int> features = ReadCSVLine(lines[i]);
	//uso solo las 3d primeras columnas
    x_.insert(x_.end(), features.begin(), features.begin() + 4);
	//cout << line_data[0] << endl; 
  }
  //cout << line_data[2] << endl;
  // verifico lo que he leido
/*   for (int u = 0; u < 8; ++u) {
	cout << x_[u] << endl;
  } */
  
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  
  // creo un tensor con las dimensiones del los datos (csv) 6 ,3 
  //  6  = cantidad de filas
  //  3  = cantidad de columnas
  //Tensor x(DataTypeToEnum<int>::v(), TensorShape{static_cast<int>(x_.size())/3, 3});
  Tensor x(DataTypeToEnum<int>::v(), TensorShape{2, 4}); // este es más comprensible
  //copy_n(data_set.x().begin(), data_set.x().size(),x_data.flat<float>().data());
  // copio el contenido leido del csv al Tensor x
  copy_n(x_.begin(), x_.size(), x.flat<int>().data());
  
  
  
  //auto a = Placeholder(root, DT_INT32);
  // [3 3; 3 3]
  //auto b = Const(root, {{21, 2}, {19, 70}});
  auto b = Const(root, x );
  auto a = Const(root, { {2, 2, 2, 2}, {1, 1, 1, 1} });
  //auto c = Mul(root, a, b);
  auto m = Add(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  // Feed a <- [1 2; 3 4]
  session.Run( {m}, &outputs);

  //LOG(INFO) << "Los valores encontrados son:  " << outputs[0].matrix<int>();
  std::cout << "Los valores leídos del CSV:  " << std::endl;
  std::cout << outputs[0].matrix<int>()  << std::endl;
  std::cout << " ----- Hoy he tenido suerte! -----" << std::endl;
  return 0;
}


