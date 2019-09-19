#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"


int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;

  Scope root = Scope::NewRootScope();
  auto a = Placeholder(root, DT_INT32);
  // [3 3; 3 3]
  auto b = Const(root, {{4, 4}, {4, 5}});
  auto c = Mul(root, a, b);
  ClientSession session(root);
  std::vector<Tensor> outputs;

  // Feed a <- [1 2; 3 4]
  session.Run({ {a, { {15, 2}, {3, 4} } } }, {c}, &outputs);

  //LOG(INFO) << "Los valores encontrados son:  " << outputs[0].matrix<int>();
  std::cout << "Los valores encontrados son:  " << std::endl;
  std::cout << outputs[0].matrix<int>()  << std::endl;
  return 0;
}