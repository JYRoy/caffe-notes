#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
#ifdef USE_MKL
  *ptr = mkl_malloc(size ? size:1, 64);
#else
  *ptr = malloc(size);
#endif
  *use_cuda = false;
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
#ifdef USE_MKL
  mkl_free(ptr);
#else
  free(ptr);
#endif
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
// 负责caffe底层的数据的内存管理
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();  // 获取cpu上的data地址
  void set_cpu_data(void* data);  // 将私有成员cpu_data指定为外部的data ptr
  const void* gpu_data();  // 获取gpu上的data地址
  void set_gpu_data(void* data);  // 将私有成员gpu_data指定为外部的data ptr
  void* mutable_cpu_data();  // 指示用户可修改的cpu数据内容
  void* mutable_gpu_data();  // 指示用户可修改的gpu数据内容
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };  // 数据同步状态，本类中操作数据的方法会根据synchead来进行逻辑判断
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void* cpu_ptr_;  // cpu内存地址，可以由对象自己分配，也可以外部指定
  void* gpu_ptr_;  // gpu内存地址，可以由对象自己分配，也可以外部指定
  size_t size_;  // 数据大小
  SyncedHead head_;  // 数据同步状态
  bool own_cpu_data_;  // 指示是否由对象内部调用CaffeMoallocHost分配的内存
  bool cpu_malloc_use_cuda_;  // 指示是否使用cudaMallocHost分配pinned memory
  bool own_gpu_data_;  // 指示是否由对象内部调用cudaMalloc分配的显存
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
