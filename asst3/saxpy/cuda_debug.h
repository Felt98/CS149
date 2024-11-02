#ifndef CUDA_DEBUG_H
#define CUDA_DEBUG_H

#include <stdio.h>
#include <cuda_runtime.h>  // 确保包含 CUDA 的运行时 API

// 如果定义了 DEBUG，则使用 cudaCheckError 包装 CUDA 调用
#ifdef DEBUG
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr, "CUDA Error: %s at %s:%d\n", 
        cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#else
// 如果没有定义 DEBUG，cudaCheckError 就直接返回 ans
#define cudaCheckError(ans) ans
#endif

#endif // CUDA_DEBUG_H
