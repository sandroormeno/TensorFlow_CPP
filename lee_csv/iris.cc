#include <iostream>
#include <fstream>
#include <string>
using namespace std;

int main () {
  string line;
  ifstream myfile ("tensorflow/cc/lee_csv/datos.csv");
  if (myfile.is_open())
  {
    while ( myfile.good() )
    {
      //getline (myfile,line,',');
	  getline (myfile,line);
      cout << line << endl;
    }
    myfile.close();
  }

  else cout << "Unable to open file"; 

  return 0;
}