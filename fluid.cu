#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cuda_fp16.h>
#include "cutil_math.h"
#include "double_buffer.cpp"


// Container for simulation state
struct fluid_state {

    float3 impulseLoc;
    float impulseTemp;
    float impulseDensity;
    float impulseRadius;
    float f_weight;
    float cell_size;
    float time_step;
    int3 dim;
    int64_t nelems;
    int step;
    DoubleBuffer<float3> *velocity;
    DoubleBuffer<float> *density;
    DoubleBuffer<float> *temperature;
    DoubleBuffer<float> *pressure;
    float *diverge;

    fluid_state(int3 dims) {
        step = 0;
        dim = dims;
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


// A couple IO utility functions
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

// GPU helper functions
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

template <typename T> inline __device__ T zero() { return 0.0; }

template <> inline __device__ float  zero<float>() { return 0.0f; }
template <> inline __device__ float3 zero<float3>() { return make_float3(0.0f); }

template <typename T>
inline __device__ T get_cell(int3 c, int3 d, T *vol) {
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= d.x || c.y >= d.y || c.z >= d.z) {
        return zero<T>();
    } else {
        return vol[ get_voxel( c.x, c.y, c.z, d ) ];
    }
}

template <typename T>
inline __device__ T get_cellF(float3 p, int3 d, T *vol) {
    
    // bilinear interpolation
    float3 l = floor(p);
    int3 rp = make_int3(l);
    float3 dif = p-l;
    T sum = zero<T>();

    #pragma unroll
    for (int a=0; a<=1; a++) 
    {
        #pragma unroll
        for (int b=0; b<=1; b++)
        {
            #pragma unroll
            for (int c=0; c<=1; c++)
            {
                sum += abs(float(1-a)-dif.x) *
                       abs(float(1-b)-dif.y) *
                       abs(float(1-c)-dif.z) *
                    get_cell( make_int3( rp.x+a, rp.y+b, rp.z+c ), d, vol);
            }
        }
    }

    return sum;
}
   
// Convert single index into 3D coordinates
inline __device__ int3 mod_coords(int i, int d) {
    return make_int3( i%d, (i/d) % d, (i/(d*d)) );
}

template <typename T>
inline __device__ T read_shared(T *mem, dim3 c, 
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

// Simulation compute kernels
template <typename T>
__global__ void pressure_solve(T *div, T *p_src, T *p_dst, 
        int3 vd, float amount)
{
    __shared__ T loc[1024];
    const int padding = 1; // How far to load past end of cube
    const int sdim = blockDim.x+2*padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, p_src); 
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    T d = div[get_voxel(x,y,z, vd)];

    T p_sum = 
             read_shared(loc, threadIdx, s_dims, padding, -1,  0,  0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding,  1,  0,  0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding,  0, -1,  0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding,  0,  1,  0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding,  0,  0, -1);
    p_sum += read_shared(loc, threadIdx, s_dims, padding,  0,  0,  1);
    //avg /= 6.0;
    //avg -= o;

    p_dst[ get_voxel(x,y,z, vd) ] = (p_sum+amount*d)*0.166667;//o + avg*amount;
}

template <typename V, typename T>
__global__ void divergence(V *velocity, T *div, int3 vd, float half_cell)
{
    __shared__ V loc[1024];
    const int padding = 1; // How far to load past end of cube
    const int sdim = blockDim.x+2*padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, velocity); 
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    T d = 
         read_shared(loc, threadIdx, s_dims, padding,  1,  0,  0).x;
    d -= read_shared(loc, threadIdx, s_dims, padding, -1,  0,  0).x;
    d += read_shared(loc, threadIdx, s_dims, padding,  0,  1,  0).y;
    d -= read_shared(loc, threadIdx, s_dims, padding,  0, -1,  0).y;
    d += read_shared(loc, threadIdx, s_dims, padding,  0,  0,  1).z;
    d -= read_shared(loc, threadIdx, s_dims, padding,  0,  0, -1).z;
    d *= half_cell;

    div[ get_voxel(x,y,z, vd) ] = d;
}

template <typename V, typename T>
__global__ void subtract_pressure(V *v_src, V *v_dest, T *pressure, 
    int3 vd, float grad_scale)
{
    __shared__ T loc[1024];
    const int padding = 1; // How far to load past end of cube
    const int sdim = blockDim.x+2*padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

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
__global__ void advection( V *velocity, T *source, T *dest, int3 vd, 
    float time_step, float dissipation)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    V vel = velocity[ get_voxel(x,y,z,vd) ];

    float3 np = make_float3(float(x),float(y),float(z)) - time_step*vel;
    
    dest[ get_voxel(x,y,z, vd) ] = dissipation * get_cellF(np, vd, source);
}

template <typename T>
__global__ void impulse( T *target, float3 c,
    float radius, T val, int3 vd)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    float3 p = make_float3(float(x),float(y),float(z));
    
    float dist = length(p-c);

    if (dist < radius) {
        target[ get_voxel(x,y,z, vd) ] = val;
    }
}

template <typename T>
__global__ void soft_impulse( T *target, float3 c,
    float radius, T val, float speed, int3 vd)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    float3 p = make_float3(float(x),float(y),float(z));
    
    float dist = length(p-c);

    T cur = target[ get_voxel(x,y,z, vd) ];

    if (dist < radius && cur < val) {
        target[ get_voxel(x,y,z, vd) ] = cur + speed*val;
    }
}

template <typename T>
__global__ void wavey_impulse( T *target, float3 c,
    float3 size, T base, float amp, float freq, int3 vd)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
    
    float3 p = make_float3(float(x),float(y),float(z));
    
    //float dist = length(p-c);
    float3 minC = c-size;
    float3 maxC = c+size;

    //T cur = target[ get_voxel(x,y,z, vd) ];

    if (p.x>minC.x && p.y>minC.y && p.z>minC.z &&
        p.x<maxC.x && p.y<maxC.y && p.z<maxC.z ) {
        float v = 0.5*(sin(freq*p.x)+sin(freq*p.z)+0.0);
        v = v*v*v*v*v;
        target[ get_voxel(x,y,z, vd) ] = base + amp*v;
    }
}


template <typename V, typename T>
__global__ void buoyancy( V *v_src, T *t_src, T *d_src, V *v_dest, 
    float amb_temp, float time_step, float buoy, float weight, int3 vd)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    const int z = blockDim.z*blockIdx.z+threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;
   
    T temp = t_src[ get_voxel(x,y,z, vd)];
    V vel = v_src[ get_voxel(x,y,z, vd)];

    if (temp > amb_temp)
    {
        T dense = d_src[ get_voxel(x,y,z, vd)];
        vel.y += (time_step * (temp - amb_temp) * buoy - dense * weight);
    }
    
    v_dest[ get_voxel(x,y,z, vd)] = vel;
}

// Runs a single iteration of the simulation
void simulate_fluid( fluid_state& state)
{

    float measured_time=0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );

    const int s = 8;
    dim3 block( s, s, s );
    dim3 grid( (state.dim.x+s-1)/s, 
               (state.dim.y+s-1)/s, 
               (state.dim.z+s-1)/s );

    cudaEventRecord( start, 0 );
    
    advection<<<grid,block>>>(
            state.velocity->readTarget(),
            state.velocity->readTarget(),
            state.velocity->writeTarget(),
            state.dim, state.time_step, 1.0);
    state.velocity->swap();

    advection<<<grid,block>>>(
            state.velocity->readTarget(),
            state.temperature->readTarget(),
            state.temperature->writeTarget(),
            state.dim, state.time_step, 0.998);
    state.temperature->swap();

    advection<<<grid,block>>>(
            state.velocity->readTarget(),
            state.density->readTarget(),
            state.density->writeTarget(),
            state.dim, state.time_step, 0.9999);
    state.density->swap();

    buoyancy<<<grid,block>>>( 
            state.velocity->readTarget(), 
            state.temperature->readTarget(),
            state.density->readTarget(),
            state.velocity->writeTarget(), 
            0.0f, state.time_step, 1.0f, state.f_weight, state.dim);
    state.velocity->swap();

    float3 location = state.impulseLoc;
    location.x += 75.0*sinf(-0.003f*float(state.step));
    location.y += 75.0*cosf(-0.003f*float(state.step));

    /*
    soft_impulse<<<grid,block>>>( 
            state.temperature->readTarget(), 
            location, state.impulseRadius, 
            state.impulseTemp, 0.01, state.dim);

    soft_impulse<<<grid,block>>>(
            state.density->readTarget(), 
            location, state.impulseRadius, 
            state.impulseDensity, 0.01, state.dim);
    */    

    divergence<<<grid,block>>>(
            state.velocity->readTarget(),
            state.diverge, state.dim, 0.5);

    // clear pressure
    impulse<<<grid,block>>>(
            state.pressure->readTarget(),
            make_float3(0.0), 1000000.0f,
            0.0f, state.dim);
    
    for (int i=0; i<35; i++)
    {
        pressure_solve<<<grid,block>>>( 
                state.diverge,
                state.pressure->readTarget(),
                state.pressure->writeTarget(), 
                state.dim, -1.0);
        state.pressure->swap();
    }

    subtract_pressure<<<grid,block>>>(
            state.velocity->readTarget(),
            state.velocity->writeTarget(),
            state.pressure->readTarget(), 
            state.dim, 1.0);
    state.velocity->swap();

    cudaEventRecord( stop, 0 );
    cudaThreadSynchronize();
    cudaEventElapsedTime( &measured_time, start, stop );

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    std::cout << "Simulation Time: " << measured_time << "\n";
}

__device__ float2 rotate(float2 p, float a)
{
    return make_float2(p.x*cos(a) - p.y*sin(a),
                       p.y*cos(a) + p.x*sin(a));
}

// GPU volumetric raymarcher
__global__ void render_pixel( uint8_t *image, float *volume, 
        float *temper, int3 img_dims, int3 vol_dims, float step_size, 
        float3 light_dir, float3 cam_pos, float rotation)
{
    const int x = blockDim.x*blockIdx.x+threadIdx.x;
    const int y = blockDim.y*blockIdx.y+threadIdx.y;
    if (x >= img_dims.x || y >= img_dims.y) return;

    int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);
    // Create Normalized UV image coordinates
    float uvx =  float(x)/float(img_dims.x)-0.5;
    float uvy = -float(y)/float(img_dims.y)+0.5;
    uvx *= float(img_dims.x)/float(img_dims.y);     

    float3 v_center = make_float3(
            0.5*float(vol_dims.x),
            0.5*float(vol_dims.y),
            0.5*float(vol_dims.z));

    // Set up ray originating from camera
    float3 ray_pos = cam_pos-v_center;
    float2 pos_rot = rotate(make_float2(ray_pos.x, ray_pos.z), rotation);
    ray_pos.x = pos_rot.x;
    ray_pos.z = pos_rot.y;
    ray_pos += v_center;
    float3 ray_dir = normalize(make_float3(uvx,uvy,0.5));
    float2 dir_rot = rotate(make_float2(ray_dir.x, ray_dir.z), rotation);
    ray_dir.x = dir_rot.x;
    ray_dir.z = dir_rot.y;
    const float3 dir_to_light = normalize(light_dir);
    const float occ_thresh = 0.001;
    float d_accum = 1.0;
    float light_accum = 0.0;
    float temp_accum = 0.0;

    // Trace ray through volume
    for (int step=0; step<512; step++) {
        // At each step, cast occlusion ray towards light source
        float c_density = get_cellF(ray_pos, vd, volume);
        float3 occ_pos = ray_pos;
        ray_pos += ray_dir*step_size;
        // Don't bother with occlusion ray if theres nothing there
        if (c_density < occ_thresh) continue;
        float transparency = 1.0;
        for (int occ=0; occ<512; occ++) {
            transparency *= fmax(1.0-get_cellF(occ_pos, vd, volume),0.0);
            if (transparency < occ_thresh) break;
            occ_pos += dir_to_light*step_size;
        }
        d_accum *= fmax(1.0-c_density,0.0);
        light_accum += d_accum*c_density*transparency;
        if (d_accum < occ_thresh) break;
    }

    // gamma correction
    light_accum = pow(light_accum, 0.45);
    const int pixel = 3*(y*img_dims.x+x);
    image[pixel+0] = (uint8_t)(fmin(255.0*light_accum, 255.0));
    image[pixel+1] = (uint8_t)(fmin(255.0*light_accum, 255.0));
    image[pixel+2] = (uint8_t)(fmin(255.0*light_accum, 255.0));
}

void render_fluid(uint8_t *render_target, int3 img_dims, 
    float *d_volume, float *temper, int3 vol_dims, 
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
        device_img, d_volume, temper, img_dims, vol_dims, 
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
    const int3 img_d = make_int3(1920,1080,0);

    float3 cam;
    cam.x = static_cast<float>(vol_d.x)*0.5;
    cam.y = static_cast<float>(vol_d.y)*0.5;
    cam.z = 0.0;
    float3 light;
    light.x =  0.1;
    light.y =  1.0;
    light.z = -0.5;

    uint8_t *img = new uint8_t[3*img_d.x*img_d.y];
   
    fluid_state state(vol_d);
    
    state.impulseLoc = make_float3(0.5*float(vol_d.x),
                                   0.5*float(vol_d.y)-170.0,
                                   0.5*float(vol_d.z));
    state.impulseTemp = 4.0;
    state.impulseDensity = 0.35;
    state.impulseRadius = 18.0;
    state.f_weight = 0.05;
    state.time_step = 0.1;

    dim3 full_grid(vol_d.x/8+1, vol_d.y/8+1, vol_d.z/8+1);
    dim3 full_block(8,8,8);

    // zero out buffers
    impulse<<<full_grid, full_block>>>( state.velocity->readTarget(), 
        make_float3(0.0), 100000.0f, make_float3(0.0), vol_d);

    impulse<<<full_grid, full_block>>>( state.temperature->readTarget(), 
        make_float3(0.0), 100000.0f, 0.0f, vol_d);

    impulse<<<full_grid, full_block>>>( state.density->readTarget(), 
        make_float3(0.0), 100000.0f, 0.0f, vol_d);

    wavey_impulse<<<full_grid, full_block>>>( state.density->readTarget(),
        state.impulseLoc + make_float3(0.0, 70.0, 0.0),
        make_float3(100.0, 15.0, 100.0), 0.25f, 0.0f, 1.0f, vol_d);

    wavey_impulse<<<full_grid, full_block>>>( state.temperature->readTarget(),
        state.impulseLoc + make_float3(0.0, 70.0, 0.0),
        make_float3(100.0, 15.0, 100.0), 0.0f, 3.0f, 0.15f, vol_d);

    for (int f=0; f<=3000; f++) {
        
        std::cout << "Step " << f+1 << "\n";
        
        light.x = 1.05*sinf(0.006*float(state.step));
        light.z = 1.05*cosf(0.006*float(state.step));
        render_fluid(
                img, img_d, 
                state.density->readTarget(), 
                state.temperature->readTarget(),
                vol_d, 1.0, light, cam, 0.0*float(state.step));

        save_image(img, img_d, "output/R" + pad_number(f+1) + ".ppm");
        for (int st=0; st<1; st++) {
            simulate_fluid(state);
            state.step++;
        }
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString( cudaGetLastError() ) );

    cudaThreadExit();

    return 0;
}
