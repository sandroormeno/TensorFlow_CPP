# puedes usar un grafo desarrollado en python y usarlo en C++

Este ejercicio está ejecutado en una versión de tf compilada(2:30). Usé este tutorial:

https://medium.com/jim-fleming/loading-a-tensorflow-graph-with-the-c-api-4caaff88463f

Añadí algunas modificaciones (prometo hacer un tutorial actializado).

1. Complilar  : bazel run -c opt --config=monolithic //tensorflow/cc/load_graph:loader

2. Usar : g++ -std=c++11 -Wl,-rpath='$ORIGIN/lib' -Iinclude -Llib proyectos/ejercicio_2/load.cc -ltensorflow_cc -o load2