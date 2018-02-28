#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "cutil_math.h"

// Same as a dim3
struct dims {
    int x, y, z;
};

struct fdims {
    float x, y, z;
};

// Stores value contained in a cell as well
// as the values of cells adjacent to it
struct adjacent_cells {
    float o,
    xn, xp,
    yn, yp,
    zn, zp;      
};

std::string pad_number(int n)
{
    std::ostringstream ss;
    ss << std::setw( 7 ) << std::setfill( '0' ) << n;
    return ss.str();
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

inline __device__ int3 operator*(const dim3 a, const uint3 b) {
    return make_int3(a.x*b.x, a.y*b.y, a.z*b.z);
}

inline __device__ int3 operator+(dim3 a, int3 b) {
    return make_int3(a.x+b.x, a.y+b.y, a.z+b.z);
}

inline __device__ int get_voxel(int x, int y, int z, int3 d)
{
	return z*d.y*d.x + y*d.x + x;
}

inline __device__ float get_density(int3 c, int3 d, float* vol) {
	if (c.x < 0 || c.y < 0 || c.z < 0 ||
	    c.x >= d.x || c.y >= d.y || c.z >= d.z) {
		return 0.0;
	} else {
		return vol[ get_voxel( c.x, c.y, c.z, d ) ];
	}
}

inline __device__ float get_density(float3 p, int3 d, float* vol) {
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

// Convert single index into 3D coordinates
inline __device__ int3 mod_coords(int i, int d) {
    return make_int3( i%d, (i/d) % d, (i/(d*d)) );
}

__device__ adjacent_cells read_adjacent(dim3 blkDim, dim3 blkIdx, 
    dim3 thrIdx, int3 vd, float *shared, float *v_src) 
{
    const int padding = 2;
	const int sdim = blkDim.x+padding; // 10
	int t_idx = thrIdx.z*blkDim.y*blkDim.x 
		+ thrIdx.y*blkDim.x + thrIdx.x; 
    // Load sdim*sdim*sdim cube of memory into shared array 
    const int cutoff = (sdim*sdim*sdim)/2;
	if (t_idx < cutoff) {
        int3 sp = mod_coords(t_idx, sdim);
        sp = sp + blkDim*blkIdx - 1;
        shared[t_idx] = get_density( sp, vd, v_src);
        sp = mod_coords(t_idx+cutoff, sdim);
        sp = sp + blkDim*blkIdx - 1;
        shared[t_idx+cutoff] = get_density( sp, vd, v_src);
    }
    __syncthreads();
    
    adjacent_cells aj;
	
    int3 sc = make_int3( thrIdx.x, thrIdx.y, thrIdx.z );
    int3 blk_dim = make_int3(sdim, sdim, sdim);
    aj.o  = shared[ get_voxel(sc.x+1,sc.y+1,sc.z+1, blk_dim) ];
	aj.yp = shared[ get_voxel(sc.x+1,sc.y+2,sc.z+1, blk_dim) ];
    aj.yn = shared[ get_voxel(sc.x+1,sc.y  ,sc.z+1, blk_dim) ];
    aj.xn = shared[ get_voxel(sc.x  ,sc.y+1,sc.z+1, blk_dim) ];
    aj.xp = shared[ get_voxel(sc.x+2,sc.y+1,sc.z+1, blk_dim) ];
    aj.zp = shared[ get_voxel(sc.x+1,sc.y+1,sc.z+2, blk_dim) ];
    aj.zn = shared[ get_voxel(sc.x+1,sc.y+1,sc.z  , blk_dim) ];

    return aj;
}

__global__ void diffusion(float *v_src, float *v_dst, dims vol_dims, float amount)
{
	__shared__ float loc[1024];
	const int x = blockDim.x*blockIdx.x+threadIdx.x;
	const int y = blockDim.y*blockIdx.y+threadIdx.y;
	const int z = blockDim.z*blockIdx.z+threadIdx.z;

    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);
    /*
    const int padding = 2;
	const int sdim = blockDim.x+padding; // 10
	int t_idx = threadIdx.z*blockDim.y*blockDim.x 
		+ threadIdx.y*blockDim.x + threadIdx.x; 
    // Load sdim*sdim*sdim cube of memory into shared array 
    const int cutoff = sdim*sdim*sdim/2;
	if (t_idx < cutoff) {
        int3 sp = mod_coords(t_idx, sdim);
        sp = sp + blockDim*blockIdx - 1;
        loc[t_idx] = get_density( sp, vd, v_src);
        sp = mod_coords(t_idx+cutoff, sdim);
        sp = sp + blockDim*blockIdx - 1;
        loc[t_idx+cutoff] = get_density( sp, vd, v_src);
    }
    __syncthreads();

	if (x >= vd.x || y >= vd.y || z >= vd.z) return;
	
    int3 sc = make_int3( threadIdx.x, threadIdx.y, threadIdx.z );
    int3 blk_dim = make_int3(sdim, sdim, sdim);
    float cent  = loc[ get_voxel(sc.x+1,sc.y+1,sc.z+1, blk_dim) ];
	float up    = loc[ get_voxel(sc.x+1,sc.y+2,sc.z+1, blk_dim) ];
    float down  = loc[ get_voxel(sc.x+1,sc.y  ,sc.z+1, blk_dim) ];
    float left  = loc[ get_voxel(sc.x  ,sc.y+1,sc.z+1, blk_dim) ];
    float right = loc[ get_voxel(sc.x+2,sc.y+1,sc.z+1, blk_dim) ];
    float front = loc[ get_voxel(sc.x+1,sc.y+1,sc.z+2, blk_dim) ];
    float back  = loc[ get_voxel(sc.x+1,sc.y+1,sc.z  , blk_dim) ];
    
    float avg = (up+down+left+right+front+back)/6.0;
    */

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    adjacent_cells c = read_adjacent(
        blockDim, blockIdx, threadIdx, vd, loc, v_src); 

    float avg = (c.xp+c.xn+c.yp+c.yn+c.zp+c.zn)/6.0;

    avg = avg - c.o;//cent;
    v_dst[ get_voxel(x,y,z, vd) ] = c.o/*cent*/ + avg*amount;
}

// Avoid reallocating volume buffer each step by swapping?
void simulate_fluid(float *v_src, float *v_dst, dims vol_dim, float time_step)
{

    float measured_time=0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop  );

    int s = 8;
	dim3 block( s, s, s );
	dim3 grid( (vol_dim.x+s-1)/s, (vol_dim.y+s-1)/s, (vol_dim.z+s-1)/s );

	cudaEventRecord( start, 0 );
	
	diffusion<<<grid,block>>>( v_src, v_dst, vol_dim, time_step);

	cudaEventRecord( stop, 0 );
	cudaThreadSynchronize();
	cudaEventElapsedTime( &measured_time, start, stop );

	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	std::cout << "Simulation Time: " << measured_time << "\n";
}

__global__ void render_pixel( uint8_t *image, float *volume, 
	dims img_dims, dims vol_dims, float step_size, 
	fdims light_dir, fdims cam_pos, float rotation)
{
	const int x = blockDim.x*blockIdx.x+threadIdx.x;
	const int y = blockDim.y*blockIdx.y+threadIdx.y;
	if (x >= img_dims.x || y >= img_dims.y) return;
	
	int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);
	// Create Normalized UV image coordinates
	float uvx = float(x)/float(img_dims.x)-0.5;
	float uvy = float(y)/float(img_dims.y)-0.5;
	uvx *= float(img_dims.x)/float(img_dims.y);	

	// Set up ray originating from camera
	float3 ray_pos = make_float3(cam_pos.x, cam_pos.y, cam_pos.z);
	const float3 ray_dir = normalize(make_float3(uvx,uvy,0.5));
	const float3 dir_to_light = normalize(
		make_float3(light_dir.x, light_dir.y, light_dir.z));
	float d_accum = 1.0;
	float light_accum = 0.0;
	
	// Trace ray through volume
	for (int step=0; step<512; step++) {
	// At each step, cast occlusion ray towards light source
        float c_density = get_density(ray_pos, vd, volume);
        float3 occ_pos = ray_pos;
        ray_pos += ray_dir*step_size;
        // Don't bother with occlusion ray if theres nothing there
        if (c_density < 0.001) continue;
		float occlusion = 1.0;
		for (int occ=0; occ<512; occ++) {
			occlusion *= fmax(1.0-get_density(occ_pos, vd, volume),0.0);
			occ_pos += dir_to_light*step_size;
		}
		d_accum *= fmax(1.0-c_density,0.0);
		light_accum += d_accum*c_density*occlusion;
	}

	const int pixel = 3*(y*img_dims.x+x);
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
	float *d_volA = 0;
    float *d_volB = 0;
    cudaMalloc( (void**)&d_volA, vol_bytes );
    cudaMalloc( (void**)&d_volB, vol_bytes );   
    if( 0 == d_volA || 0 == d_volB )
    {
        printf("couldn't allocate GPU memory\n");
        return -1;
    }
 
    printf("Allocated %.2f MB on GPU\n", 2*vol_bytes/(1024.f*1024.f) );

	initialize_volume<<<dim3(vol_d.x/8, vol_d.y/8, vol_d.z/8), 
		dim3(8,8,8)>>>(d_volA, vol_d);

    for (int f=0; f<=800; f++) {
        std::cout << "Step " << f+1 << "\n";
        render_fluid(img, img_d, d_volA, vol_d, 1.0, light, cam, 0.0);
        save_image(img, img_d, "output/R" + pad_number(f+1) + ".ppm");
        for (int st=0; st<30; st++) {
            simulate_fluid(d_volA, d_volB, vol_d, 0.7);
            std::swap(d_volA, d_volB);
        }
    }

	delete[] img;
	cudaFree(d_volA);
    cudaFree(d_volB);

    printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

    cudaThreadExit();

    return 0;
}
