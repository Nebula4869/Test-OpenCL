#include <stdio.h>
#include <windows.h>
#include <CL/cl.h>


// 输出平台与设备信息
void getInformation(cl_platform_id platform_id, cl_device_id device_id) {
	size_t extSize;
	const char icdExt[] = "cl_khr_icd";

	cl_int error = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, 0, NULL, &extSize);
	if (error != CL_SUCCESS)
	{
		perror("Couldn't read extension data.");
		return;
	}

	char* extData = (char*)malloc(extSize);
	clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, extSize, extData, NULL);

	if (strstr(extData, icdExt) != NULL) {
		printf("Supports the %s extension.\n", icdExt);
	}

	char *name = (char*)malloc(extSize);
	clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, extSize, name, NULL);
	printf("Name                    : %s\n", name);

	char *vendor = (char*)malloc(extSize);
	clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, extSize, vendor, NULL);
	printf("Vendor                  : %s\n", vendor);

	char *version = (char*)malloc(extSize);
	clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, extSize, version, NULL);
	printf("Version                 : %s\n", version);

	char *profile = (char*)malloc(extSize);
	clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, extSize, profile, NULL);
	printf("Profile                 : %s\n", profile);

	cl_uint workItemDim;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), (void *)&workItemDim, NULL);
	printf("Max work-item dimensions: %d\n", workItemDim);

	int workItemSizes[3];
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workItemSizes), (void *)workItemSizes, NULL);
	printf("Max work-item sizes     : %d %d %d\n", workItemSizes[0], workItemSizes[1], workItemSizes[2]);

	int workGroupSize;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(int), (void *)&workGroupSize, NULL);
	printf("Max work-group sizes    : %d\n", workGroupSize);

	cl_uint ucomputUnit;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), (void *)&ucomputUnit, NULL);
	printf("Max comput_uint         : %u\n", ucomputUnit);

	cl_uint uconstantArgs = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), (void *)&uconstantArgs, NULL);
	printf("Max constant_args       : %u\n", uconstantArgs);

	cl_ulong uconstantBufferSize = 0;
	clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), (void *)&uconstantBufferSize, NULL);
	printf("Max constant_buffer_size: %u\n", uconstantBufferSize);

	free(extData);
	free(name);
	free(vendor);
	free(version);
	free(profile);
}


// 读取内核文件
void loadProgramSource(const char** files, int length, char** buffer, size_t* sizes) {
	for (int i = 0; i < length; i++) {
		FILE* file = fopen(files[i], "rb");
		if (file == NULL) {
			printf("Couldn't read the program file");
			exit(1);
		}
		fseek(file, 0, SEEK_END);
		sizes[i] = ftell(file);
		rewind(file);
		buffer[i] = (char*)malloc(sizes[i] + 1);
		buffer[i][sizes[i]] = '\0';
		fread(buffer[i], sizeof(char), sizes[i], file);
		fclose(file);
	}
}