#include "testRGB2GRAY.h"
#include "testMatmul.h"


int main() {
	testRGB2GRAY("test.jpg", 0);
	testMatmul(1000, 1000, 500, 0);
	return 0;
}