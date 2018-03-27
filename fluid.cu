#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cuda_fp16.h>
#include "cutil_math.h"
#include "double_buffer.cpp"

struct fluid_state {

    int3 dimensions;
    int64_t nelems;
    DoubleBuffer<float3> *velocity;
    DoubleBuffer<float> *density;
    DoubleBuffer<float> *temperature;
    DoubleBuffer<float> *pressure;
    float *diverge;

    fluid_state(int3 dims) {
        dimensions = dims;
        nelems = dims.x*dims.y*dims.z;
        velocity = new DoubleBuffer<float3>(nelems);
        density = new DoubleBuffer<float>(nelems);
        temperature = new DoubleBuffer<float>(nelems);
        pressure = new DoubleBuffer<float>(nelems);
        cudaMalloc( (void**) &diverge, sizeof(float)*nelems);
    }

    ~fluid_state() {
        delete velocity;
        delete density;
        delete temperature;
        delete pressure;
        cudaFree(diverge);
    }
};

std::string pad_number(int n)
{
    std::ostringstream ss;
    ss << std::setw( 7 ) << std::setfill( '0' ) << n;
    return ss.str();
}

void save_image(uint8_t *pixels, int3 img_dims, std::string name) {
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

template <typename T>
inline __device__ T get_cell(int3 c, int3 d, T *vol) {
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= d.x || c.y >= d.y || c.z >= d.z) {
    return 0.0;
    } else {
        return vol[ get_voxel( c.x, c.y, c.z, d ) ];
    }
}

template <typename T>
inline __device__ T get_cell(float3 p, int3 d, T *vol) {
    return get_cell<T>(make_int3(p), d, vol);
}
   
__global__ void initialize_volume(float *volume, int3 vd)
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

template <typename T>
inline __device__ float read_shared(T *mem, dim3 c, 
    int3 blk_dim, int pad, int x, int y, int z)
{
    return mem[ get_voxel(c.x+pad+x, c.y+pad+y, c.z+pad+z, blk_dim) ];
}

template <typename T>
__device__ void load_shared(dim3 blkDim, dim3 blkIdx, 
    dim3 thrIdx, int3 vd, int sdim, T *shared, T *src) 
{
    int t_idx = thrIdx.z*blkDim.y*blkDim.x 
        + thrIdx.y*blkDim.x + thrIdx.x; 
    // Load sdim*sdim*sdim cube of memory into shared array 
    const int cutoff = (sdim*sdim*sdim)/2;
    if (t_idx < cutoff) {
        int3 sp = mod_coords(t_idx, sdim);
        sp = sp + blkDim*blkIdx - 1;
        shared[t_idx] = get_cell( sp, vd, src);
        sp = mod_coords(t_idx+cutoff, sdim);
        sp = sp + blkDim*blkIdx - 1;
        shared[t_idx+cutoff] = get_cell( sp, vd, src);
    }
}

template <typename T>
__global__ void pressure_solve(T *v_src, T *v_dst, int3 vol_dims, float amount)
{
    __shared__ T loc[1024];
    const int padding = 1; // How far to load past end of cube
    const int sdim = blockDim.x+2*padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;
    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, v_src); 
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    T o = 
           read_shared(loc, threadIdx, s_dims, padding,  0,  0,  0);
    T avg = 
           read_shared(loc, threadIdx, s_dims, padding, -1,  0,  0);
    avg += read_shared(loc, threadIdx, s_dims, padding,  1,  0,  0);
    avg += read_shared(loc, threadIdx, s_dims, padding,  0, -1,  0);
    avg += read_shared(loc, threadIdx, s_dims, padding,  0,  1,  0);
    avg += read_shared(loc, threadIdx, s_dims, padding,  0,  0, -1);
    avg += read_shared(loc, threadIdx, s_dims, padding,  0,  0,  1);
    avg /= 6.0;
    avg -= o;

    v_dst[ get_voxel(x,y,z, vd) ] = o + avg*amount;
}

template <typename V, typename T>
__global__ void divergence(V *velocity, T *div, int3 vol_dims)
{
    __shared__ V loc[1024];
    const int padding = 1; // How far to load past end of cube
    const int sdim = blockDim.x+2*padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;
    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, velocity); 
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    T d = 
         read_shared(loc, threadIdx, s_dims, padding,  1,  0,  0);
    d -= read_shared(loc, threadIdx, s_dims, padding, -1,  0,  0);
    d += read_shared(loc, threadIdx, s_dims, padding,  0,  1,  0);
    d -= read_shared(loc, threadIdx, s_dims, padding,  0, -1,  0);
    d += read_shared(loc, threadIdx, s_dims, padding,  0,  0,  1);
    d -= read_shared(loc, threadIdx, s_dims, padding,  0,  0, -1);
    d *= 0.5;

    div[ get_voxel(x,y,z, vd) ] = d;
}

template <typename V, typename T>
__global__ void subtract_pressure(V *v_src, V *v_dest, T *pressure, 
    int3 vol_dims, float grad_scale)
{
    __shared__ V loc[1024];
    const int padding = 1; // How far to load past end of cube
    const int sdim = blockDim.x+2*padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;
    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, pressure); 
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    V old_v = get_cell(make_int3(x,y,z), vd, v_src);

    V grad;
    grad.x = 
        read_shared(loc, threadIdx, s_dims, padding,  1,  0,  0) - 
        read_shared(loc, threadIdx, s_dims, padding, -1,  0,  0);
    grad.y =
        read_shared(loc, threadIdx, s_dims, padding,  0,  1,  0) -
        read_shared(loc, threadIdx, s_dims, padding,  0, -1,  0);
    grad.z = 
        read_shared(loc, threadIdx, s_dims, padding,  0,  0,  1) -
        read_shared(loc, threadIdx, s_dims, padding,  0,  0, -1);

    v_dest[ get_voxel(x,y,z, vd) ] = old_v - grad*grad_scale;
}

template <typename V, typename T>
__global__ void advection( V *velocity, T *source, T *dest, int3 vol_dims, 
    float time_step, float dissipation)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;
    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    float3 p = make_float3(float(x),float(y),float(z));

    V vel = velocity[ get_voxel(x,y,z,vd) ];
    float3 np = make_float3(float(x),float(y),float(z)) - time_step*vel;

    dest[ get_voxel(x,y,z, vd) ] = dissipation * get_cell(np, vd, source);
}

template <typename T>
__global__ void impulse( T *target, float xp, float yp, float zp, 
    float radius, T val, int3 vol_dims)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;
    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    float3 p = make_float3(float(x),float(y),float(z));
    
    float dist = sqrt(pow(p.x-xp,2.0)+pow(p.y-yp,2.0)+pow(p.z-zp,2.0));

    if (dist < radius) {
        target[ get_voxel(x,y,z, vd) ] = val;
    }
}

template <typename T>
__global__ void clear( T *target, T val, int3 vol_dims)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;
    const int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    target[ get_voxel(x,y,z, vd) ] = val;
}

// void time_kernel(void *kernel, grid, block, params) ?? 

void simulate_fluid( fluid_state& state, int3 vol_dim, float time_step)
{

    float measured_time=0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );

    const int s = 8;
    dim3 block( s, s, s );
    dim3 grid( (vol_dim.x+s-1)/s, (vol_dim.y+s-1)/s, (vol_dim.z+s-1)/s );

    cudaEventRecord( start, 0 );
        
    pressure_solve<<<grid,block>>>( 
            state.density->readTarget(),
            state.density->writeTarget(), 
            vol_dim, time_step);

    cudaEventRecord( stop, 0 );
    cudaThreadSynchronize();
    cudaEventElapsedTime( &measured_time, start, stop );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    std::cout << "Simulation Time: " << measured_time << "\n";
}

__global__ void render_pixel( uint8_t *image, float *volume, 
        int3 img_dims, int3 vol_dims, float step_size, 
        float3 light_dir, float3 cam_pos, float rotation)
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
    float3 ray_pos = cam_pos;
    const float3 ray_dir = normalize(make_float3(uvx,uvy,0.5));
    const float3 dir_to_light = normalize(light_dir);
    const float occ_thresh = 0.001;
    float d_accum = 1.0;
    float light_accum = 0.0;

    // Trace ray through volume
    for (int step=0; step<512; step++) {
        // At each step, cast occlusion ray towards light source
        float c_density = get_cell(ray_pos, vd, volume);
        float3 occ_pos = ray_pos;
        ray_pos += ray_dir*step_size;
        // Don't bother with occlusion ray if theres nothing there
        if (c_density < occ_thresh) continue;
        float transparency = 1.0;
        for (int occ=0; occ<512; occ++) {
            transparency *= fmax(1.0-get_cell(occ_pos, vd, volume),0.0);
            if (transparency < occ_thresh) break;
            occ_pos += dir_to_light*step_size;
        }
        d_accum *= fmax(1.0-c_density,0.0);
        light_accum += d_accum*c_density*transparency;
        if (d_accum < occ_thresh) break;
    }

    const int pixel = 3*(y*img_dims.x+x);
    image[pixel+0] = (uint8_t)(fmin(255.0*light_accum, 255.0));
    image[pixel+1] = (uint8_t)(fmin(255.0*light_accum, 255.0));
    image[pixel+2] = (uint8_t)(fmin(255.0*light_accum, 255.0));
}

void render_fluid(uint8_t *render_target, int3 img_dims, 
    float *d_volume /*pointer to gpu mem?*/, int3 vol_dims, 
    float step_size, float3 light_dir, float3 cam_pos, float rotation) {

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

    const int3 vol_d = make_int3(512,512,512);
    const int3 img_d = make_int3(800,600,0);

    float3 cam;
    cam.x = static_cast<float>(vol_d.x)*0.5;
    cam.y = static_cast<float>(vol_d.y)*0.5;
    cam.z = 0.0;
    float3 light;
    light.x = -0.2;
    light.y = -0.9;
    light.z =  0.2;

    uint8_t *img = new uint8_t[3*img_d.x*img_d.y];
   
    fluid_state state(vol_d);

    initialize_volume<<<dim3(vol_d.x/8, vol_d.y/8, vol_d.z/8), 
                dim3(8,8,8)>>>(state.density->writeTarget(), vol_d);
    state.density->swap();

    for (int f=0; f<=800; f++) {
        std::cout << "Step " << f+1 << "\n";
        render_fluid(img, img_d, state.density->readTarget(), vol_d, 1.0, light, cam, 0.0);
        save_image(img, img_d, "output/R" + pad_number(f+1) + ".ppm");
        for (int st=0; st<30; st++) {
            simulate_fluid(state, vol_d, 0.7);
            state.density->swap();
        }
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

    cudaThreadExit();

    return 0;
}
