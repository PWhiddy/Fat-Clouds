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

__device__ inline int get_voxel(int x, int y, int z, int3 d)
{
	return z*d.y*d.x + y*d.x + x;
}

__device__ inline float get_density(int3 c, int3 d, float* vol) {
	if (c.x < 0 || c.y < 0 || c.z < 0 ||
	    c.x >= d.x || c.y >= d.y || c.z >= d.z) {
		return 0.0;
	} else {
		return vol[ get_voxel( c.x, c.y, c.z, d ) ];
	}
}

__device__ inline float get_density(float3 p, int3 d, float* vol) {
	return get_density(make_int3(p), d, vol);
}

__global__ void initialize_volume(float *volume, dims vd)
{
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	int z = blockDim.z*blockIdx.z+threadIdx.z;
	if (x >= vd.x || y >= vd.y || z >= vd.z) return;
	const float width = 128.0;
	const float den = 0.018;
	float dx = float(x-vd.x/2);
	float dy = float(y-vd.y/2);
	float dz = float(z-vd.z/2);
	float dist = sqrtf(dx*dx+dy*dy+dz*dz);
	float density = den/(1.0+pow(1.2,dist-width));
	volume[ get_voxel( x, y, z, make_int3(vd.x, vd.y, vd.z)) ] 
		= density;
}

__global__ void diffuse(float *volume, dims vd)
{
	__shared__ float loc[1024];
	int x = blockDim.x*blockIdx.x+threadIdx.x;
	int y = blockDim.y*blockIdx.y+threadIdx.y;
	int z = blockDim.z*blockIdx.z+threadIdx.z;
	if (x >= vd.x || y >= vd.y || z >= vd.z) return;
	// Fill shared memory with 1 extra pixel on each side
	int shared_pixel = z*blockDim.y*blockDim.x+y*blockDim.x+x;
	//loc[shared_pixel] = 
	// 
	int3 d = make_int3(vd.x, vd.y, vd.z);
	//float up = volume[ get_voxel(x,y,z, 
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
	float3 ray_dir = normalize(make_float3(uvx,uvy,0.5));
	float3 dir_to_light = normalize(
		make_float3(light_dir.x, light_dir.y, light_dir.z));
	float d_accum = 1.0;
	float light_accum = 0.0;
	
	// Trace ray through volume
	for (int step=0; step<512; step++) {
	// At each step, cast occlusion ray towards light source	
		float3 occ_pos = ray_pos;
		float occlusion = 1.0;
		for (int occ=0; occ<512; occ++) {
			occlusion *= fmax(1.0-get_density(occ_pos, vd, volume),0.0);
			occ_pos += dir_to_light*step_size;
		}
		float c_density = get_density(ray_pos, vd, volume);
		d_accum *= fmax(1.0-c_density,0.0);
		light_accum += d_accum*c_density*occlusion;
		ray_pos += ray_dir*step_size;
	}

	int pixel = 3*(y*img_dims.x+x);
	image[pixel+0] = (uint8_t)(fmin(255.0*light_accum, 255.0));
	image[pixel+1] = (uint8_t)(fmin(255.0*light_accum, 255.0));
	image[pixel+2] = (uint8_t)(fmin(255.0*light_accum, 255.0));
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

int main(int argc, char* args[])
{
	
	dims vol_d;
	vol_d.x = 512;
	vol_d.y = 512;
	vol_d.z = 512;
	dims img_d;
	img_d.x = 800;
	img_d.y = 600;

	fdims cam;
	cam.x = static_cast<float>(vol_d.x)*0.5;
	cam.y = static_cast<float>(vol_d.y)*0.5;
	cam.z = 0.0;
	fdims light;
	light.x = -0.2;
	light.y = -0.9;
	light.z =  0.2;

	uint8_t *img = new uint8_t[3*img_d.x*img_d.y];
	int vol_bytes = vol_d.x*vol_d.y*vol_d.z*sizeof(float);
	float *d_vol = 0;
        cudaMalloc( (void**)&d_vol, vol_bytes );
        if( 0 == d_vol )
        {
                printf("couldn't allocate GPU memory\n");
                return -1;
        }

        printf("Allocated %.2f MB on GPU\n", vol_bytes/(1024.f*1024.f) );

	initialize_volume<<<dim3(vol_d.x/8, vol_d.y/8, vol_d.z/8), 
		dim3(8,8,8)>>>(d_vol, vol_d);

	render_fluid(img, img_d, d_vol, vol_d, 1.0, light, cam, 0.0);

	save_image(img, img_d, "test.ppm");

	delete[] img;
	cudaFree(d_vol);

        printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

        cudaThreadExit();

        return 0;
}
