CC = g++
CFLAGS = -std=c++11

DIR = /home/sandro/tf_test_old/

INC = -I/home/sandro/tf_test_old/include

LDFLAGS = -L$(DIR)/lib/ -Wl,-rpath='/home/sandro/tf_test_old//lib'
LDFLAGS += -ltensorflow_cc

all: main

main:
	$(CC) $(CFLAGS) -o main main.cpp $(INC) $(LDFLAGS)
run:
	./main
clean:
	rm -f main