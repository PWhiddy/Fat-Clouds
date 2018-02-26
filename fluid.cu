#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

struct dims {
    int x, y, z;
};

struct fdims {
    float x, y, z;
};

void render_fluid(uint8_t *render_target, dims img_dims, float *d_volume /*pointer to gpu mem?*/, dims vol_dims, float step_size, fdims light_dir, fdims cam_pos, float rotation) {

	float measured_time=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( 32, 32 );
	dim3 grid( (img_dims.x+32-1)/32, (img_dims.y+32-1)/32 );

	cudaEventRecord( start, 0 );
	
	// Allocate device memory for image
	int img_bytes = sizeof(uint8_t)*img_dims.x*img_dims.y;
	uint8_t *device_img;
	cudaMalloc( (void**)&device_img, img_bytes );
        if( 0 == device_img )
        {
                printf("couldn't allocate GPU memory\n");
   		return;
        }

	//render_pixel<<<grid,block>>>( 
	//	device_img, d_volume, img_dims, vol_dims, 
	//	step_size, light_dir, cam_pos, rotation);

	// Read image back
	cudaMemcpy( render_target, device_img, img_bytes, cudaMemcpyDeviceToHost );

	cudaEventRecord( stop, 0 );
	cudaThreadSynchronize();
	cudaEventElapsedTime( &measured_time, start, stop );

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	std::cout << "Render Time: " << measured_time << "\n";
	cudaFree(device_img);
}

void save_image(uint8_t *pixels, dims img_dims, std::string name) {
    std::ofstream file(name, std::ofstream::binary);
    if (file.is_open()) {
        file << "P6\n" << img_dims.x << " " << img_dims.y << "\n" << "255\n";
        file.write((char *)pixels, img_dims.x*img_dims.y*3);
        file.close();
    } else {
        std::cout << "Could not open file :(\n";
    }
}

// Avoid reallocating volume buffer each step by swapping?
void simulate_fluid(float *volume, dims dimensions, float time_step);

__global__ void kernel_A( float *g_data, int dimx, int dimy )
{
	int ix  = blockIdx.x;
    int iy  = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*dimx + ix;

    float value = g_data[idx];
	
    value = sinf(value);

    g_data[idx] = value;
}

float run_kernel( void (*kernel)( float*, int,int), float *d_data, int dimx, int dimy, int nreps, int blockx, int blocky )
{
	float measured_time=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( blockx, blocky );
	dim3 grid( dimx/block.x, dimy/block.y );

	cudaEventRecord( start, 0 );
	for(int i=0; i<nreps; i++)
		kernel<<<grid,block>>>( d_data, dimx,dimy );
	cudaEventRecord( stop, 0 );
	cudaThreadSynchronize();
	cudaEventElapsedTime( &measured_time, start, stop );
	measured_time /= nreps;

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	return measured_time;
}

int main()
{
	int dimx = 2*1024;
	int dimy = 2*1024;
	
	int nreps = 10;

        int nbytes = dimx*dimy*sizeof(float);

        float *d_data=0, *h_data=0;
        cudaMalloc( (void**)&d_data, nbytes );
        if( 0 == d_data )
        {
                printf("couldn't allocate GPU memory\n");
                return -1;
        }
        printf("allocated %.2f MB on GPU\n", nbytes/(1024.f*1024.f) );
        h_data = (float*)malloc( nbytes );
        if( 0 == h_data )
        {
                printf("couldn't allocate CPU memory\n");
                return -2;
        }
        printf("allocated %.2f MB on CPU\n", nbytes/(1024.f*1024.f) );
        for(int i=0; i<dimx*dimy; i++)
                h_data[i] = 10.f + rand() % 256;
        cudaMemcpy( d_data, h_data, nbytes, cudaMemcpyHostToDevice );

        float measured_time=0.0f;

        measured_time = run_kernel( kernel_A, d_data, dimx,dimy, nreps, 1, 512 );
        printf("A:  %8.2f ms\n", measured_time );

        printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

        if( d_data )
                cudaFree( d_data );
        if( h_data )
                free( h_data );

        cudaThreadExit();

        return 0;
}
