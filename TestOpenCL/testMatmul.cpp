#include <stdio.h>
#include <windows.h>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include "openclUtils.h"


static cl_int error;
static cl_kernel kernel;
static cl_command_queue cQ;
static cl_mem memA, memB, memC, memM, memN, memK;


// forѭ������˷�
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


// OpenCV����˷�
static void matmulOpenCV(cv::Mat A, cv::Mat B, cv::Mat C) {
	C = A * B;
}


// OpenCL����˷�
static void matmulOpenCL(float* A, float* B, float* C, unsigned int M, unsigned int N, unsigned int K) {
	size_t localThreads[2] = { 32, 32 };
	size_t globalThreads[2] = { ((N + localThreads[0] - 1) / localThreads[0])*localThreads[0], ((M + localThreads[1] - 1) / localThreads[1])*localThreads[1] };
	// ������д�뻺����
	error = clEnqueueWriteBuffer(cQ, memA, CL_FALSE, 0, sizeof(float) * M * K, A, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memB, CL_FALSE, 0, sizeof(float) * K * N, B, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memM, CL_FALSE, 0, sizeof(unsigned int), &M, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memN, CL_FALSE, 0, sizeof(unsigned int), &N, 0, NULL, NULL);
	error = clEnqueueWriteBuffer(cQ, memK, CL_FALSE, 0, sizeof(unsigned int), &K, 0, NULL, NULL);

	// �����ں�
	cl_event evt;
	error = clEnqueueNDRangeKernel(cQ, kernel, 2, 0, globalThreads, localThreads, 0, NULL, &evt);  //�ں�ִ����ɺ󣬻Ὣevt��ΪCL_SUCCESS/CL_COMPLETE
	clWaitForEvents(1, &evt);   //�ȴ������¼�����
	clReleaseEvent(evt);

	// �����������������������
	error = clEnqueueReadBuffer(cQ, memC, CL_TRUE, 0, sizeof(float) * M * N, C, 0, NULL, NULL);
}


// ��ʼ��OpenCL
static void initOpenCL(int M, int N, int K, cl_int device) {
	const int numFiles = 1;
	cl_uint numPlatforms;

	// ��ȡƽ̨����
	error = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (error != CL_SUCCESS) {
		printf("Couldn't find any platforms.");
		exit(1);
	}
	printf("Total platforms: %d\n", numPlatforms);

	// ��ȡƽ̨ID
	cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
	clGetPlatformIDs(numPlatforms, platforms, NULL);

	// �����豸
	cl_device_id deviceID;
	error = clGetDeviceIDs(platforms[device], CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, NULL);
	if (error != CL_SUCCESS) {
		printf("Couldn't access device %d.\n", device);
		exit(1);
	}
	// ���ƽ̨���豸��Ϣ
	getInformation(platforms[device], deviceID);

	// �����豸������
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &error);
	// �����������
	cQ = clCreateCommandQueue(context, deviceID, NULL, &error);

	// �����ں��ļ�
	const char *fileNames[] = { "matmul.cl" }; //��������ں��ļ����б�
	char* buffer[numFiles];
	size_t sizes[numFiles];
	loadProgramSource(fileNames, numFiles, buffer, sizes); //��ȡ�ں��ļ��ı�
	cl_program program = clCreateProgramWithSource(context, numFiles, (const char**)buffer, sizes, &error); //����program����
	error = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL); //�������
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

	// �����ں�
	error = clCreateKernelsInProgram(program, 1, &kernel, NULL);

	// ���豸�ϴ����������
	memA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * M * K, NULL, &error);
	memB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * K * N, NULL, &error);
	memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * M * N, NULL, &error);
	memM = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &error);
	memN = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &error);
	memK = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int), NULL, &error);

	// ���ں˺������ݲ���
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


// �Ƚ����־���˷�Ч��
void testMatmul(int M, int N, int K, cl_int device) {
	printf("------Testing Matmul------\n");
	initOpenCL(M, N, K, device);

	// ��ʱ����
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