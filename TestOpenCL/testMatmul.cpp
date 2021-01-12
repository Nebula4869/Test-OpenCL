#include <stdio.h>
#include <windows.h>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include "openclUtils.h"


static cl_int error;
static cl_kernel kernel;
static cl_command_queue cQ;
static cl_mem memA, memB, memC, memM, memN, memK;


// for循环矩阵乘法
static void matmul(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
	for (int y = 0; y < M; y++) {
		for (int x = 0; x < N; x++) {
			int sum = 0;
			for (int i = 0; i < K; i++)
			{
				sum += A[y * K + i] * B[i * N + x];
			}
			C[y * N + x] = sum;
		}
	}
}


// OpenCV矩阵乘法
static void matmulOpenCV(cv::Mat A, cv::Mat B, cv::Mat C) {
	C = A * B;
}


// OpenCL矩阵乘法
static void matmulOpenCL(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
	size_t localThreads[2] = { 32, 32 };
	size_t globalThreads[2] = { ((N + localThreads[0] - 1) / localThreads[0])*localThreads[0], ((M + localThreads[1] - 1) / localThreads[1])*localThreads[1] };
	// 将数据写入缓冲区
	error = clEnqueueWriteBuffer(cQ, memA, CL_FALSE, 0, sizeof(float) * M * K, A, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memB, CL_FALSE, 0, sizeof(float) * K * N, B, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memM, CL_FALSE, 0, sizeof(unsigned int), &M, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memN, CL_FALSE, 0, sizeof(unsigned int), &N, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memK, CL_FALSE, 0, sizeof(unsigned int), &K, 0, NULL, NULL);

	// 启动内核
	cl_event evt;
	error = clEnqueueNDRangeKernel(cQ, kernel, 2, 0, globalThreads, localThreads, 0, NULL, &evt);  //内核执行完成后，会将evt置为CL_SUCCESS/CL_COMPLETE
	clWaitForEvents(1, &evt);   //等待命令事件发生
	clReleaseEvent(evt);

	// 将缓冲区结果拷贝回主机端
	error = clEnqueueReadBuffer(cQ, memC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);
}


// 初始化OpenCL
static void initOpenCL(int M, int N, int K, cl_int device) {
	const int numFiles = 1;
	cl_uint numPlatforms;

	// 获取平台数量
	error = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (error != CL_SUCCESS) {
		printf("Couldn't find any platforms.");
		exit(1);
	}
	printf("Total platforms: %d\n", numPlatforms);

	// 获取平台ID
	cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	clGetPlatformIDs(numPlatforms, platforms, NULL);

	// 访问设备
	cl_device_id deviceID;
	error = clGetDeviceIDs(platforms[device], CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, NULL);
	if (error != CL_SUCCESS) {
		printf("Couldn't access device %d.\n", device);
		exit(1);
	}
	// 输出平台与设备信息
	getInformation(platforms[device], deviceID);

	// 创建设备上下文
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &error);
	// 创建命令队列
	cQ = clCreateCommandQueue(context, deviceID, NULL, &error);

	// 编译内核文件
	const char *fileNames[] = { "matmul.cl" }; //待编译的内核文件名列表
	char* buffer[numFiles];
	size_t sizes[numFiles];
	loadProgramSource(fileNames, numFiles, buffer, sizes); //读取内核文件文本
	cl_program program = clCreateProgramWithSource(context, numFiles, (const char**)buffer, sizes, &error); //创建program对象
	error = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL); //编译程序
	if (error != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *programLog = (char*)malloc(log_size + 1);
		programLog[log_size] = '\0';
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, log_size + 1, programLog, NULL);
		printf(programLog);
		free(programLog);
		exit(1);
	}

	// 创建内核
	error = clCreateKernelsInProgram(program, 1, &kernel, NULL);

	// 在设备上创建缓存对象
	memA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &error);
	memB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * N, NULL, &error);
	memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &error);
	memM = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &error);
	memN = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &error);
	memK = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &error);

	// 向内核函数传递参数
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC);
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memM);
	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memN);
	error = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memK);

	clReleaseContext(context);
	clReleaseProgram(program);
	free(fileNames);
	for (int j = 0; j < numFiles; j++) {
		free(buffer[j]);
	}
}


// 比较三种矩阵乘法效率
void testMatmul(int M, int N, int K, cl_int device) {
	printf("------Testing Matmul------\n");
	initOpenCL(M, N, K, device);

	// 计时运行
	cv::Mat A = cv::Mat::ones(M, K, CV_32F);
	cv::Mat B = cv::Mat::ones(K, N, CV_32F);
	cv::Mat C = cv::Mat::ones(M, N, CV_32F);

	DWORD start;

	start = GetTickCount();
	matmul((float*)A.data, (float*)B.data, (float*)C.data, M, N, K);
	printf("Time cost of for loop matmul:\t%dms\n", GetTickCount() - start);

	start = GetTickCount();
	matmulOpenCV(A, B, C);
	printf("Time cost of OpenCV matmul:\t%dms\n", GetTickCount() - start);

	start = GetTickCount();
	matmulOpenCL((float*)A.data, (float*)B.data, (float*)C.data, M, N, K);
	printf("Time cost of OpenCL matmul:\t%dms\n", GetTickCount() - start);

	A.release();
	B.release();
	C.release();
	clReleaseCommandQueue(cQ);
	clReleaseKernel(kernel);
	clReleaseMemObject(memA);
	clReleaseMemObject(memB);
	clReleaseMemObject(memC);
	clReleaseMemObject(memM);
	clReleaseMemObject(memN);
	clReleaseMemObject(memK);
}