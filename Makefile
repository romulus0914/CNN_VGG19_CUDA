CC = g++
NVCC = nvcc
NVFLAGS = -O3 -std=c++11 -arch=sm_30
CXXFLAGS = -O3
LDFLAGS = -lm -lcublas -lcusparse
TARGETS = cnn_vgg19_cuda

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?

%: %.cpp
	$(CC) $(CXXFLAGS) $(LDFLAGS) -o $@ $?

vgg: vgg19.py
	python vgg19.py

image: image_converter.py
	python image_converter.py

softmax: softmax.py
	python softmax.py

clean:
	rm -rf $(TARGETS)