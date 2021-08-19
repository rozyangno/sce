nvcc sce.cu -o  sce --cudart static -O3 --relocatable-device-code=false -gencode arch=compute_75,code=sm_75 -lgsl -lgslcblas

