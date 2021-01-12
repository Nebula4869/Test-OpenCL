__kernel void hello_kernel(const __global float* A, const __global float* B, __global float* C, __global unsigned int *pM, __global unsigned int *pN, __global unsigned int *pK) {
	#define TS 32
	int M = *pM;
	int N = *pN;
	int K = *pK;

	// ��ȡ�������е�ǰ���������ڵ�2d����
	const int local_x = get_local_id(0);
	const int local_y = get_local_id(1);
	// ͨ����ǰ���������ڵĹ�����ID���Լ��ڴ˹������е��������������ȫ�ֵ�����
	const int global_x = TS * get_group_id(0) + local_x;
	const int global_y = TS * get_group_id(1) + local_y;
	// ����local �ڴ棬��ͬһ���ڹ���������й�����ɼ�
	__local int Asub[TS][TS];
	__local int Bsub[TS][TS];

	int sum = 0;
	// һ����������Ҫѭ�������group������
	const int numTiles = K / TS;
	for (int t = 0; t < numTiles; t++) {
		// �����ʱ�������ڵ�ǰ�ֿ������λ��
		const int tiled_x = TS * t + local_x;
		const int tiled_y = TS * t + local_y;

		// ��global�ڴ��а�����A��B��ÿ���СΪ32*32��ֵ���浽local�ڴ�
		Asub[local_y][local_x] = A[tiled_y * M + global_x];
		Bsub[local_y][local_x] = B[global_y * K + tiled_x];

		// ͬ���Ա�֤Asub��Bsub����ɸ�ֵ
		barrier(CLK_LOCAL_MEM_FENCE);

		// �Ե����ֿ����Ӻ�
		for (int k = 0; k < TS; k++) {
			sum += Asub[k][local_x] * Bsub[local_y][k];
		}

		// ͬ���Ա�֤��һ���ֿ�ȫ���������
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	C[global_y * M + global_x] = sum;
}