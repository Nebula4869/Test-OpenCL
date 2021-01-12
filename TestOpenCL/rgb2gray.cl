__kernel void kernel_rgb2gray(__global unsigned char * rgbImage, __global unsigned char * grayImage, __global unsigned * const p_height, __global unsigned * const p_width)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int height = *p_height;
	int width = *p_width;
	if (x < width && y < height)
	{
		int index = y * width + x;
		grayImage[index] = (30 * rgbImage[index * 3] + 150 * rgbImage[index * 3 + 1] + 76 * rgbImage[index * 3 + 2]) >> 8;
	}
}