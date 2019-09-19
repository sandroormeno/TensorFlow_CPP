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
  
  
/*   string line;
  ifstream myfile ("tensorflow/cc/ver_datos/datos.csv");
  if (myfile.is_open())
  {
	while ( myfile.good() )
    {
      getline (myfile,line, ',');
	  //getline (myfile,line);
      cout << line << endl;
    }
    myfile.close();
  }
  else cout << "NO puedo ver el archivo";  */
  
  ifstream file("datos.csv");
  if (!file) {
    file.open("tensorflow/cc/ver_datos/datos.csv");
  }
  stringstream buffer;
  buffer << file.rdbuf();
  string line;
  vector<string> lines;
  while(getline(buffer, line, '\n')) {
    lines.push_back(line);
  }
  //cout << lines[0] << endl;

  string valor;
  vector<int> line_data;
  for (int i = 0; i < lines.size(); ++i) {
	stringstream lineStream(lines[i]);
	
	while(getline(lineStream, valor , ',')) {
		line_data.push_back(stod(valor));
		//cout << valor << endl;	
	}
	x_.insert(x_.end(), line_data.begin(), line_data.begin() + 1);
	
	//cout << line_data[0] << endl;


    
  }
  cout << line_data[2] << endl;
  //cout << x_[0] << endl;
  
   
  
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  //auto a = Placeholder(root, DT_INT32);
  // [3 3; 3 3]
  auto b = Const(root, {{21, 2}, {19, 70}});
  //auto c = Mul(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  // Feed a <- [1 2; 3 4]
  session.Run( {b}, &outputs);

  //LOG(INFO) << "Los valores encontrados son:  " << outputs[0].matrix<int>();
  //std::cout << "Los valores encontrados son:  " << std::endl;
  //std::cout << outputs[0].matrix<int>()  << std::endl;
  return 0;
}


