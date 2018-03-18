#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
class DoubleBuffer
{
public:
    DoubleBuffer();
    DoubleBuffer(int nelements);
    T* readTarget();
    T* writeTarget();
    void swap();
    int byteCount();
    ~DoubleBuffer();
private:
    int nbytes;
    T* A;
    T* B;
};
