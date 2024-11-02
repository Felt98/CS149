#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"
using namespace std;
#define THREADS_PER_BLOCK 256
int threadNum=36;
void printCudaInfo();

// helper function to round an integer up to the next power of 2, 向上取为2的幂
static inline int nextPow2(int n) {     
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upsweep(int* result,int N, int offset ){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    i*=2*offset;                            //错：别忘了内循环的步长的是2*offset，也就是说线程编号要乘2*offset，才是其真正有用的编号
    if(i+2*offset-1<N){
        // printf("threadId: %d , index: %d ,i+offset-1: %d\n",blockIdx.x*blockDim.x+threadIdx.x,i,i+offset-1);
        result[i+2*offset-1]+=result[i+offset-1];       //线程计算走一步，线程切换走两步
    }
        
} 

__global__ void downsweep(int* result,int N, int offset){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    i*=2*offset;
    if(i+2*offset-1<N){
        // printf("threadId: %d , index: %d ,i+offset-1: %d\n",blockIdx.x*blockDim.x+threadIdx.x,i,i+offset-1);
        int t=result[i+offset-1];
        result[i+offset-1]=result[i+2*offset-1];
        result[i+2*offset-1]+=t;
    }
        
}

void exclusive_scan(int* input, int N, int* result)  //折半threadNum的写法
{
    
    N = nextPow2(N);                                //N扩充为2的幂
    int total_threadNum=N;                          //总线程数
    int block_num;
    for(int offset=1;offset<=N/2;offset*=2){        //线程之间计算的步数每次多跨一倍，最大可以跨一半的数
        total_threadNum/=2;                              //每次都只需要计算一半的数据，依次减半N/2,N/4, N/8 ,直到1
        block_num=(total_threadNum+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;      //注意向上取整
        upsweep<<<block_num,min(THREADS_PER_BLOCK,total_threadNum)>>>(result,N,offset);
        cudaDeviceSynchronize();
    }

    int resultarray[N];

    cudaMemset(&result[N-1],0,sizeof(int));         //result[N-1]=0;错误写法,不能直接对gpu变量赋值

    for(int offset=N/2;offset>=1;offset/=2){        //下扫就反过来遍历
        total_threadNum*=2;                         //与上扫相反，线程数从1开始增倍
        block_num=(total_threadNum+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;      //注意向上取整
        downsweep<<<block_num,min(THREADS_PER_BLOCK,total_threadNum)>>>(result,N,offset);
        cudaDeviceSynchronize();
    }   
    //int resultarray[N];
    // cudaMemcpy(resultarray, result, N * sizeof(int), cudaMemcpyDeviceToHost);
    // printf("result: ");
    // for(int i=0;i<N;i++){
    //     printf("%d--",resultarray[i]);
    // }
    // printf("\n");
}


//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    printCudaInfo();
    int* device_result;
    int* device_input;
    int N = end - inarray;  
    printf("N: %d\n",N);
    /*
    此代码将提供给exclusive_scan的数组向上取为到2的幂，但原始输入结束后的元素将保持未初始化状态，也不会检查其正确性。 
    为了简单起见，exclusive_scan的学生实现可能会假设数组的分配长度是2的幂。这将导致在非2次方输入上的额外工作，但值得采用仅2次方解决方案的简单性。
    */
    int rounded_length = nextPow2(end - inarray);
    printf("rounded_length: %d\n",rounded_length);

    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    double startTime = CycleTimer::currentSeconds();
    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);      //只复制end - inarray，N多余的就不复制了
  
    cudaFree(device_result);
    cudaFree(device_input);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //    
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    return 0; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
