#include "double_buffer.h"
#include <utility>

template <typename T>
DoubleBuffer<T>::DoubleBuffer()
{
    A = 0;
    B = 0;
    nbytes = 0;
}

template <typename T>
DoubleBuffer<T>::DoubleBuffer(int nelements)
{
    nbytes = sizeof(T)*nelements;
    cudaMalloc( (void**)&A, nbytes );
    cudaMalloc( (void**)&B, nbytes );
    if( 0 == A || 0 == B )
    {
        printf("couldn't allocate GPU memory\n");
        return;
    }
    printf("Allocated %.2f MB on GPU\n", 2*nbytes/(1024.f*1024.f));
}

template <typename T>
T* DoubleBuffer<T>::readTarget()
{
    return A;
}

template <typename T>
T* DoubleBuffer<T>::writeTarget()
{
    return B;
}

template <typename T>
void DoubleBuffer<T>::swap()
{
    std::swap(A,B);
}

template <typename T>
int DoubleBuffer<T>::byteCount()
{
    return nbytes;
}

template <typename T>
DoubleBuffer<T>::~DoubleBuffer()
{
    cudaFree(A);
    cudaFree(B);
}
