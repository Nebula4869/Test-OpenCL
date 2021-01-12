#include <stdio.h>
#include <windows.h>
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include "openclUtils.h"


static cl_int error;
static cl_kernel kernel;
static cl_command_queue cQ;
static cl_mem memRgbImage, memGrayImage, memImagewidth, memImageheight;


// for循环RGB转灰度
static void RGB2Gray(unsigned char* srcImage, unsigned char* dstImage, int width, int height) {
	for (int i = 0; i < height; i++) {
		unsigned char *pRGB = srcImage + i * width * 3;
		unsigned char *pGray = dstImage + i * width;
		for (int j = 0; j < width; j++, pRGB += 3) {
			pGray[j] = int(0.114 * pRGB[0] + 0.587 * pRGB[1] + 0.299 * pRGB[2]);
		}
	}
}


// 8位精度整形运算RGB转灰度
static void RGB2GrayInt8(unsigned char* srcImage, unsigned char* dstImage, int width, int height) {
	for (int i = 0; i < height; i++) {
		unsigned char *pRGB = srcImage + i * width * 3;
		unsigned char *pGray = dstImage + i * width;
		for (int j = 0; j < width; j++, pRGB += 3) {
			pGray[j] = (30 * pRGB[0] + 150 * pRGB[1] + 76 * pRGB[2]) >> 8;
		}
	}
}


// SSE指令集优化RGB转灰度，12路并行
static void RGB2GraySSE(unsigned char* srcImage, unsigned char* dstImage, int width, int height) {
	const int bWeight = int(0.114 * 256 + 0.5);
	const int gWeight = int(0.587 * 256 + 0.5);
	const int rWeight = 256 - bWeight - gWeight; // int(0.299 * 256 + 0.5)

	for (int i = 0; i < height; i++) {
		unsigned char *pRGB = srcImage + i * width * 3;
		unsigned char *pGray = dstImage + i * width;
		for (int j = 0; j < width - 12; j += 12, pRGB += 36) {
			__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 0))), _mm_setr_epi16(bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight)); //1
			__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 1))), _mm_setr_epi16(gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight)); //2
			__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 2))), _mm_setr_epi16(rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight)); //3

			__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 8))), _mm_setr_epi16(rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight));//4
			__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 9))), _mm_setr_epi16(bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight));//5
			__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 10))), _mm_setr_epi16(gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight));//6

			__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 18))), _mm_setr_epi16(bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight));//7
			__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 19))), _mm_setr_epi16(gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight));//8
			__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 20))), _mm_setr_epi16(rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight));//9

			__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 26))), _mm_setr_epi16(rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight));//10
			__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 27))), _mm_setr_epi16(bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight));//11
			__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(pRGB + 28))), _mm_setr_epi16(gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight));//12

			__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));//13
			__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));//14
			__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));//15
			__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));//16
			__m128i sclaL = _mm_srli_epi16(sumaL, 8);//17
			__m128i sclaH = _mm_srli_epi16(sumaH, 8);//18
			__m128i sclbL = _mm_srli_epi16(sumbL, 8);//19
			__m128i sclbH = _mm_srli_epi16(sumbH, 8);//20
			__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//21
			__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//22
			__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));//23
			__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));//24
			__m128i accumL = _mm_or_si128(shftaL, shftbL);//25
			__m128i accumH = _mm_or_si128(shftaH, shftbH);//26
			__m128i h3 = _mm_or_si128(accumL, accumH);//27
													  //__m128i h3 = _mm_blendv_epi8(accumL, accumH, _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
			_mm_storeu_si128((__m128i *)(pGray + j), h3);
		}
	}
}


// AVX2指令集优化RGB转灰度，10路并行
static void  _RGB2Gray(unsigned char* srcImage, const int32_t width, const int32_t start_row, const int32_t thread_stride, const int32_t Stride, unsigned char* Dest)
{
	// AVX2
	constexpr uint16_t bWeight = static_cast<uint16_t>(32768.0 * 0.114 + 0.5);
	constexpr uint16_t gWeight = static_cast<uint16_t>(32768.0 * 0.587 + 0.5);
	constexpr uint16_t rWeight = static_cast<uint16_t>(32768.0 * 0.299 + 0.5);
	// 使用16个uint16数据构建一个256位向量
	static const __m256i weightVector = _mm256_setr_epi16(bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight, gWeight, rWeight, bWeight);

	for (int i = start_row; i < start_row + thread_stride; i++)
	{
		unsigned char *pRGB = srcImage + i * Stride;
		unsigned char *pGray = Dest + i * width;
		for (int j = 0; j < width - 10; j += 10, pRGB += 30)
		{
			// 读取内存中的128位数据（16个uint8数据）并扩展成一个256位向量（16个uint16数据）
			//B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5 G5 R5 B6 
			__m256i temp = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pRGB + 0)));
			// 两个256位向量（16个uint16数据）相乘得到256位向量（8个32位向量）
			__m256i in1 = _mm256_mulhrs_epi16(temp, weightVector);

			//B6 G6 R6 B7 G7 R7 B8 G8 R8 B9 G9 R9 B10 G10 R10 B11
			temp = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(pRGB + 15)));
			__m256i in2 = _mm256_mulhrs_epi16(temp, weightVector);


			//0  1  2  3   4  5  6  7  8  9  10 11 12 13 14 15    16 17 18 19 20 21 22 23 24 25 26 27 28   29 30  31       
			//B1 G1 R1 B2 G2 R2 B3 G3  B6 G6 R6 B7 G7 R7 B8 G8    R3 B4 G4 R4 B5 G5 R5 B6 R8 B9 G9 R9 B10 G10 R10 B11
			__m256i mul = _mm256_packus_epi16(in1, in2);

			__m256i b1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(
				//  B1 B2 B3 -1, -1, -1  B7  B8  -1, -1, -1, -1, -1, -1, -1, -1,
				0, 3, 6, -1, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, -1, B4 B5 B6 -1, -1  B9 B10 -1, -1, -1, -1, -1, -1
				-1, -1, -1, 1, 4, 7, -1, -1, 9, 12, -1, -1, -1, -1, -1, -1));

			__m256i g1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(

				// G1 G2 G3 -1, -1  G6 G7  G8  -1, -1, -1, -1, -1, -1, -1, -1, 
				1, 4, 7, -1, -1, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, -1  G4 G5 -1, -1, -1  G9  G10 -1, -1, -1, -1, -1, -1	
				-1, -1, -1, 2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1));

			__m256i r1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(

				//  R1 R2 -1  -1  -1  R6  R7  -1, -1, -1, -1, -1, -1, -1, -1, -1,	
				2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, R3 R4 R5 -1, -1, R8 R9  R10 -1, -1, -1, -1, -1, -1
				-1, -1, 0, 3, 6, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, -1));



			// B1+G1+R1  B2+G2+R2 B3+G3  0 0 G6+R6  B7+G7+R7 B8+G8 0 0 0 0 0 0 0 0 0 0 R3 B4+G4+R4 B5+G5+R5 B6 0 R8 B9+G9+R9 B10+G10+R10 0 0 0 0 0 0

			__m256i accum = _mm256_adds_epu8(r1, _mm256_adds_epu8(b1, g1));


			// _mm256_castsi256_si128(accum)
			// B1+G1+R1  B2+G2+R2 B3+G3  0 0 G6+R6  B7+G7+R7 B8+G8 0 0 0 0 0 0 0 0

			// _mm256_extracti128_si256(accum, 1)
			// 0 0 R3 B4+G4+R4 B5+G5+R5 B6 0 R8 B9+G9+R9 B10+G10+R10 0 0 0 0 0 0

			__m128i h3 = _mm_adds_epu8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1));

			_mm_storeu_si128((__m128i *)(pGray + j), h3);
		}
	}
}
static void RGB2GrayAVX2(unsigned char* srcImage, unsigned char* dstImage, int width, int height)
{
	_RGB2Gray(srcImage, width, 0, height, width * 3, dstImage);
}


// OpenCV RGB转灰度
static void RGB2GrayOpenCV(cv::Mat srcImage, cv::Mat dstImage) {
	cv::cvtColor(srcImage, dstImage, cv::COLOR_RGB2GRAY);
}


// OpenCL RGB转灰度
static void RGB2GrayOpenCL(unsigned char* srcImage, unsigned char* dstImage, int width, int height) {
	// 将数据写入缓冲区
	error = clEnqueueWriteBuffer(cQ, memRgbImage, CL_FALSE, 0, sizeof(uchar) * 3 * height * width, srcImage, 0, NULL, NULL);

	// 启动内核
	size_t globalThreads[2] = { width, height };
	cl_event evt;
	error = clEnqueueNDRangeKernel(cQ, kernel, 2, 0, globalThreads, NULL, 0, NULL, &evt);  //内核执行完成后，会将evt置为CL_SUCCESS/CL_COMPLETE
	clWaitForEvents(1, &evt);   //等待命令事件发生
	clReleaseEvent(evt);

	// 将缓冲区结果拷贝回主机端
	error = clEnqueueReadBuffer(cQ, memGrayImage, CL_FALSE, 0, sizeof(uchar) * height * width, dstImage, 0, NULL, NULL);
}


// 初始化OpenCL
static void initOpenCL(int width, int height, cl_int device) {
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
	const char *fileNames[] = { "rgb2gray.cl" }; //待编译的内核文件名列表
	char* buffer[numFiles];
	size_t sizes[numFiles];
	loadProgramSource(fileNames, numFiles, buffer, sizes); //读取内核文件文本
	cl_program program = clCreateProgramWithSource(context, numFiles, (const char**)buffer, sizes, &error); //创建program对象
	error = clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL); //编译程序
	if (error != CL_SUCCESS) {
		size_t logSize;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
		char *programLog = (char*)malloc(logSize + 1);
		programLog[logSize] = '\0';
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, logSize + 1, programLog, NULL);
		printf(programLog);
		free(programLog);
		exit(1);
	}

	// 创建内核
	error = clCreateKernelsInProgram(program, 1, &kernel, NULL);

	// 在设备上创建缓存对象
	memRgbImage = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(uchar) * 3 * height * width, NULL, &error);
	memGrayImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(uchar) * height * width, NULL, &error);
	memImageheight = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &height, &error); //CL_MEM_COPY_HOST_PTR指定创建缓存对象后拷贝数据
	memImagewidth = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), &width, &error);

	// 向内核函数传递参数
	error = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memRgbImage);
	error = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memGrayImage);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memImageheight);
	error = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memImagewidth);

	clReleaseContext(context);
	clReleaseProgram(program);
	free(fileNames);
	for (int j = 0; j < numFiles; j++) {
		free(buffer[j]);
	}
}


// RGB转GRAY测试
void testRGB2GRAY(char* imgPath, cl_int device) {
	printf("------Testing RGB2GRAY------\n");
	cv::Mat rgbImage = cv::imread(imgPath);
	int height = rgbImage.rows;
	int width = rgbImage.cols;
	initOpenCL(width, height, device);
	cv::Mat grayImage;

	DWORD start;

	grayImage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	start = GetTickCount();
	for (int i = 0; i < 100; ++i) {
		RGB2Gray((unsigned char*)rgbImage.data, (unsigned char*)grayImage.data, width, height);
	}
	printf("Time cost of for loop RGB2GRAY for 100 times:\t%dms\n", GetTickCount() - start);
	cv::imshow("grayImage", grayImage);
	cv::waitKey(1000);

	grayImage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	start = GetTickCount();
	for (int i = 0; i < 100; ++i) {
		RGB2GrayInt8((unsigned char*)rgbImage.data, (unsigned char*)grayImage.data, width, height);
	}
	printf("Time cost of Int8 RGB2GRAY for 100 times:\t%dms\n", GetTickCount() - start);
	cv::imshow("grayImage", grayImage);
	cv::waitKey(1000);

	grayImage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	start = GetTickCount();
	for (int i = 0; i < 100; ++i) {
		RGB2GraySSE((unsigned char*)rgbImage.data, (unsigned char*)grayImage.data, width, height);
	}
	printf("Time cost of SSE RGB2GRAY for 100 times:\t%dms\n", GetTickCount() - start);
	cv::imshow("grayImage", grayImage);
	cv::waitKey(1000);

	grayImage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	start = GetTickCount();
	for (int i = 0; i < 100; ++i) {
		RGB2GrayAVX2((unsigned char*)rgbImage.data, (unsigned char*)grayImage.data, width, height);
	}
	printf("Time cost of AVX2 RGB2GRAY for 100 times:\t%dms\n", GetTickCount() - start);
	cv::imshow("grayImage", grayImage);
	cv::waitKey(1000);

	grayImage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	start = GetTickCount();
	for (int i = 0; i < 100; ++i) {
		RGB2GrayOpenCV(rgbImage, grayImage);
	}
	printf("Time cost of OpenCV RGB2GRAY for 100 times:\t%dms\n", GetTickCount() - start);
	cv::imshow("grayImage", grayImage);
	cv::waitKey(1000);

	grayImage = cv::Mat(height, width, CV_8UC1, cv::Scalar(0));
	start = GetTickCount();
	for (int i = 0; i < 100; ++i) {
		RGB2GrayOpenCL((unsigned char*)rgbImage.data, (unsigned char*)grayImage.data, width, height);
	}
	printf("Time cost of OpenCL RGB2GRAY for 100 times:\t%dms\n", GetTickCount() - start);
	cv::imshow("grayImage", grayImage);
	cv::waitKey(1000);

	rgbImage.release();
	grayImage.release();
	clReleaseCommandQueue(cQ);
	clReleaseKernel(kernel);
	clReleaseMemObject(memRgbImage);
	clReleaseMemObject(memGrayImage);
	clReleaseMemObject(memImageheight);
	clReleaseMemObject(memImagewidth);
}