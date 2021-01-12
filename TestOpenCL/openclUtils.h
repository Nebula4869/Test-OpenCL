#pragma once
#include <CL/cl.h>
void getInformation(cl_platform_id platform_id, cl_device_id device_id);
void loadProgramSource(const char** files, int length, char** buffer, size_t* sizes);