#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

//"tensorflow/cc/models/datos.csv"
using namespace std;

int main(){

  char data[200];

  
  fstream file;

  file.open ("tensorflow/cc/lee_csv/datos.csv");
  
  // Reding from file
  file >> data;
  cout << "los datos leidos son :" << endl;
  cout  << data << endl;;

  //closing the file
  file.close();
  return 0;
}