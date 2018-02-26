#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include "cutil_math.h"

struct dims {
    int x, y, z;
};

struct fdims {
    float x, y, z;
};

void save_image(uint8_t *pixels, dims img_dims, std::string name) {
    std::ofstream file("output/"+name, std::ofstream::binary);
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

__device__ inline float get_density(float3 p, int3 d, float* vol) {
	int3 c = make_int3(p);
	if (c.x < 0 || c.y < 0 || c.z < 0 ||
	    c.x >= d.x || c.y >= d.y || c.z >= d.z) {
		return 0.0;
	} else {
		return vol[c.z*d.y*d.x + c.y*d.x + c.x];
	}
	
}

__global__ void render_pixel( uint8_t *image, float *volume, 
	dims img_dims, dims vol_dims, float step_size, 
	fdims light_dir, fdims cam_pos, float rotation)
{
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	if (x >= img_dims.x || y >= img_dims.y) return;
	
	int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);
	// Create Normalized UV image coordinates
	float uvx = float(x)/float(img_dims.x)-0.5;
	float uvy = float(y)/float(img_dims.y)-0.5;
	uvx *= float(img_dims.x)/float(img_dims.y);	

	// Set up ray originating from camera
	float3 ray_pos = make_float3(cam_pos.x, cam_pos.y, cam_pos.z);
	float3 ray_dir = normalize(make_float3(uvx,uvy,0.4));
	float3 dir_to_light = normalize(
		make_float3(light_dir.x, light_dir.y, light_dir.z));
	float accum = 0.0;
	
	// Trace ray through volume
	for (int step=0; step<64; step++) {
	// At each step, cast occlusion ray towards light source	
		float3 occ_pos = ray_pos;
		float occlusion = 1.0;
		for (int occ=0; occ<32; occ++) {
			occlusion *= (1.0-get_density(occ_pos, vd, volume));
			occ_pos += dir_to_light*step_size;
		}
		accum += get_density(ray_pos, vd, volume)*occlusion;
		ray_pos += ray_dir*step_size;
	}

	int pixel = 3*(y*img_dims.x+x);
	image[pixel+0] = (uint8_t)(50.0*accum);
	image[pixel+1] = (uint8_t)(50.0*accum);
	image[pixel+2] = (uint8_t)(50.0*accum);
}

void render_fluid(uint8_t *render_target, dims img_dims, 
	float *d_volume /*pointer to gpu mem?*/, dims vol_dims, 
	float step_size, fdims light_dir, fdims cam_pos, float rotation) {

	float measured_time=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

	dim3 block( 32, 32 );
	dim3 grid( (img_dims.x+32-1)/32, (img_dims.y+32-1)/32 );

	cudaEventRecord( start, 0 );
	
	// Allocate device memory for image
	int img_bytes = 3*sizeof(uint8_t)*img_dims.x*img_dims.y;
	uint8_t *device_img;
	cudaMalloc( (void**)&device_img, img_bytes );
        if( 0 == device_img )
        {
                printf("couldn't allocate GPU memory\n");
   		return;
        }

	render_pixel<<<grid,block>>>( 
		device_img, d_volume, img_dims, vol_dims, 
		step_size, light_dir, cam_pos, rotation);

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

__global__ void kernel_A( float *g_data, int dimx, int dimy )
{
	int ix  = blockIdx.x;
    int iy  = blockIdx.y*blockDim.y + threadIdx.y;
    int idx = iy*dimx + ix;

    float value = g_data[idx];
	
    value = sinf(value);

    g_data[idx] = value;
}

float run_kernel( void (*kernel)( float*, int,int), float *d_data, 
	int dimx, int dimy, int nreps, int blockx, int blocky )
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
	
	dims vol_d;
	vol_d.x = 64;
	vol_d.y = 64;
	vol_d.z = 64;
	dims img_d;
	img_d.x = 800;
	img_d.y = 600;

	fdims cam;
	cam.x = 0.0;
	cam.y = 0.0;
	cam.z = -1.0;
	fdims light;
	light.x = -0.1;
	light.y = -0.9;
	light.z = 0.2;

	uint8_t *img = new uint8_t[3*img_d.x*img_d.y];
	int vol_bytes = vol_d.x*vol_d.y*vol_d.z*sizeof(float);
	float *d_vol = 0;
        cudaMalloc( (void**)&d_vol, vol_bytes );
        if( 0 == d_vol )
        {
                printf("couldn't allocate GPU memory\n");
                return -1;
        }

	render_fluid(img, img_d, d_vol, vol_d, 1.0, light, cam, 0.0);

	save_image(img, img_d, "test.ppm");

	delete[] img;

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
