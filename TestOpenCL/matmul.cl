__kernel void hello_kernel(const __global float* A, const __global float* B, __global float* C, __global unsigned int *pM, __global unsigned int *pN, __global unsigned int *pK) {
	#define TS 32
	int M = *pM;
	int N = *pN;
	int K = *pK;

	// 获取工作组中当前工作项所在的2d索引
	const int local_x = get_local_id(0);
	const int local_y = get_local_id(1);
	// 通过当前工作项所在的工作组ID和自己在此工作组中的索引计算出其在全局的索引
	const int global_x = TS * get_group_id(0) + local_x;
	const int global_y = TS * get_group_id(1) + local_y;
	// 定义local 内存，在同一个在工作组对所有工作项可见
	__local int Asub[TS][TS];
	__local int Bsub[TS][TS];

	int sum = 0;
	// 一个工作项需要循环计算的group的数量
	const int numTiles = K / TS;
	for (int t = 0; t < numTiles; t++) {
		// 计算此时工作项在当前分块的索引位置
		const int tiled_x = TS * t + local_x;
		const int tiled_y = TS * t + local_y;

		// 从global内存中把数组A和B中每块大小为32*32的值储存到local内存
		Asub[local_y][local_x] = A[tiled_y * M + global_x];
		Bsub[local_y][local_x] = B[global_y * K + tiled_x];

		// 同步以保证Asub和Bsub均完成赋值
		barrier(CLK_LOCAL_MEM_FENCE);

		// 对单个分块计算加和
		for (int k = 0; k < TS; k++) {
			sum += Asub[k][local_x] * Bsub[local_y][k];
		}

		// 同步以保证这一个分块全部计算完毕
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[global_y * M + global_x] = sum;
}