#ifndef IGNORE_INCLUDES
#include <metal_stdlib>
#include "shader_types.h"
#endif

using namespace metal;


#define GRAVITY -9.81
#define delta_time (as_type<float>(atomic_load_explicit(&compute_args.time_step, memory_order_relaxed)))
#define EPS 1.0e-5

#define CONCAT1(x,y) x##y
#define CONCAT(x,y) CONCAT1(x,y)
#define PREFIX(x) CONCAT(x, __LINE__)
#define for_neighbor(position, func) { \
int3 PREFIX(cell) = (int3) floor(position/compute_args.kernel_radius); \
uint PREFIX(grid_count) = compute_args.grid_dims.x * compute_args.grid_dims.y * compute_args.grid_dims.z; \
for (int z = -1; z < 2; z++) { \
    for (int y = -1; y < 2; y++) { \
        for (int x = -1; x < 2; x++) { \
            int3 PREFIX(new_cell) = PREFIX(cell) + int3(x, y, z); \
            if (PREFIX(new_cell).x < 0 || PREFIX(new_cell).y < 0 || PREFIX(new_cell).z < 0 || PREFIX(new_cell).x >= compute_args.grid_dims.x || PREFIX(new_cell).y >= compute_args.grid_dims.y || PREFIX(new_cell).z >= compute_args.grid_dims.z) { \
                continue; \
            } \
            uint PREFIX(new_cell_hash) = hash_cell(PREFIX(new_cell)) % PREFIX(grid_count); \
            int PREFIX(i) = cell_indices[PREFIX(new_cell_hash)]; \
            if (PREFIX(i) == -1) { \
                continue; \
            } \
            ParticleLocation PREFIX(location) = cells[PREFIX(i)]; \
            do { \
                os_log_default.log_debug("neighbor %d", PREFIX(i)); \
                int j_id = PREFIX(location).index; \
                PREFIX(i)++; \
                PREFIX(location) = cells[PREFIX(i)]; \
                func \
            } while (PREFIX(location).cell == PREFIX(new_cell_hash)); \
        } \
    } \
} }

device Particle* constant particles [[buffer(0)]];
device ComputeArguments& constant compute_args [[buffer(1)]];
device ParticleLocation* constant cells [[buffer(2)]];
device uint* constant cell_indices [[buffer(3)]];
uint id [[thread_position_in_grid]];


float kernal(device Particle& i, Particle j, float H) {
    float res = 0.0f;
    float dist = length(i.position-j.position);
    float m_K = 8 / (M_PI_F * pow(H, 3));
    const float q = dist / H;

    if (q <= 1.0f) {
        if (q <= 0.5f) {
            const float q2 = q * q;
            const float q3 = q2 * q;
            res = m_K * (6.0f * q3 - 6.0f * q2 + 1.0f);
        }
        else {
            const float q1 = 1.0f - q;
            const float q3 = q1 * q1 * q1;
            res = m_K * (2.0f * q3);
        }
    }

    return res;
}

float3 gradient_kernel(Particle i, Particle j, float H) {
    float3 res;
    float dist = length(i.position-j.position);
    float q = dist / H;
    float m_L = 48 / (M_PI_F * pow(H, 3));
    if (dist > EPS && (q <= 1.0f))
    {
        float3 grad_q = (i.position-j.position) * (1.0f / (dist * H));
        if (q <= 0.5f)
        {
            res = m_L * q * (3.0f * q - 2.0f) * grad_q;
        }
        else
        {
            float factor = 1.0f - q;
            res = m_L * (-factor * factor) * grad_q;
        }
    }
    else {
        res = { 0.0f, 0.0f, 0.0f };
    }
    return res;
}

float density_pressure_force(uint id) {
    Particle i = particles[id];
    float pressure_force = 0.0;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        pressure_force += dot(i.pressure_acceleration - j.pressure_acceleration, gradient_kernel(i, j, compute_args.kernel_radius));
    }
    return pressure_force * compute_args.volume;
}

uint hash_cell(int3 cell)
{
     const uint hashK1 = 15823;
     const uint hashK2 = 9737333;
     const uint hashK3 = 440817757;
    const uint blockSize = 50;
    uint3 ucell = (uint3) (cell + blockSize / 2);

    uint3 localCell = ucell % blockSize;
    uint3 blockID = ucell / blockSize;
    uint blockHash = blockID.x * hashK1 + blockID.y * hashK2 + blockID.z * hashK3;
    return localCell.x + blockSize * (localCell.y + blockSize * localCell.z) + blockHash;
}

kernel void compute_cell() {
    device Particle& i = particles[id];
    int3 cell = (int3) floor(i.position/compute_args.kernel_radius);
    uint grid_count = compute_args.grid_dims.x * compute_args.grid_dims.y * compute_args.grid_dims.z;
    cells[id] = ParticleLocation {id, hash_cell(cell) % grid_count};
   // cells[id] = ParticleLocation { id, hash_cell(cell) % grid_count};
}

//template<void (*func)(uint)> v
template<typename F, F func, typename... Args> void get_neighbors(float3 position, Args... args) {
    int3 cell = (int3) floor(position/compute_args.kernel_radius);
    uint grid_count = compute_args.grid_dims.x * compute_args.grid_dims.y * compute_args.grid_dims.z;
    for (int z = -1; z < 2; z++) {
        for (int y = -1; y < 2; y++) {
            for (int x = -1; x < 2; x++) {
                int3 new_cell = cell + int3(x, y, z);
                if (new_cell.x < 0 || new_cell.y < 0 || new_cell.z < 0 || new_cell.x >= compute_args.grid_dims.x || new_cell.y >= compute_args.grid_dims.y || new_cell.z >= compute_args.grid_dims.z) {
                    continue;
                }
                uint new_cell_hash = hash_cell(new_cell) % grid_count;
                int i = cell_indices[new_cell_hash];
                if (i == -1) {
                    continue;
                }
                ParticleLocation location = cells[i];
                do {
                    os_log_default.log_debug("neighbor %d", i);
                    func(location.index, args...);
                    i++;
                    location = cells[i];
                } while (location.cell == new_cell_hash);
            }
        }
    }
}


kernel void compute_pressure_accel() {
    device Particle& i = particles[id];
    i.pressure_acceleration = float3(0);
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        float sum = i.pressure_rho2 + j.pressure_rho2;
        if (fabs(sum) > EPS) {
            i.pressure_acceleration += sum * -compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
        }
    }
}

kernel void compute_pressure_accel2() {
    device Particle& i = particles[id];
    i.pressure_acceleration = float3(0);
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        float sum = i.pressure_rho2v + j.pressure_rho2v;
        if (fabs(sum) > EPS) {
            i.pressure_acceleration += sum * -compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
        }
    }
}



kernel void compute_densities_and_factors2() {
    device Particle& i = particles[id];
    i.density = 0;
    float3 sum = 0;
    float sum_squared = 0;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        
        Particle j = particles[j_id];
        i.density += compute_args.volume * kernal(i, j, compute_args.kernel_radius);
        if (j_id == id) {
            continue;
        }
        float3 gradient = -compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
        sum += gradient;
        sum_squared += length_squared(gradient);
    }
    i.density *= compute_args.rest_density;
    if (dot(sum, sum) + sum_squared > EPS) {
        i.factor = 1.0 / (dot(sum, sum) + sum_squared);
    } else {
        i.factor = 0.0;
    }
}

//template<typename... Args, void (*func)(uint)> struct{
//
//
//};

//void compute_density_factor(uint index, thread float3* sum, thread float* sum_squared) {
//    device Particle& i = particles[id];
//    Particle j = particles[index];
//    i.density += compute_args.volume * kernal(i, j, compute_args.kernel_radius);
//    if (index == id) return;
//    float3 gradient = -compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
//    *sum += gradient;
//    *sum_squared += length_squared(gradient);
//}
//
//
//kernel void compute_densities_and_factors() {
//    device Particle& i = particles[id];
//    i.density = 0;
//    float3 sum = 0;
//    float sum_squared = 0;
//    get_neighbors<decltype(compute_density_factor), compute_density_factor>(i.position, &sum, &sum_squared);
//    i.density *= compute_args.rest_density;
//    if (dot(sum, sum) + sum_squared > EPS) {
//        i.factor = 1.0 / (dot(sum, sum) + sum_squared);
//    } else {
//        i.factor = 0.0;
//    }
//}

void compute_density_factor(uint index, thread float3* sum, thread float* sum_squared) {
    device Particle& i = particles[id];
    Particle j = particles[index];
    i.density += compute_args.volume * kernal(i, j, compute_args.kernel_radius);
    if (index == id) return;
    float3 gradient = -compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
    *sum += gradient;
    *sum_squared += length_squared(gradient);
}


kernel void compute_densities_and_factors() {
    device Particle& i = particles[id];
    i.density = 0;
    float3 sum = 0;
    float sum_squared = 0;
    for_neighbor(i.position, {
        Particle j = particles[j_id];
        i.density += compute_args.volume * kernal(i, j, compute_args.kernel_radius);
        if (j_id == id) {
            continue;;
        }
        float3 gradient = -compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
        sum += gradient;
        sum_squared += length_squared(gradient);
    });
    //get_neighbors<decltype(compute_density_factor), compute_density_factor>(i.position, &sum, &sum_squared);
    i.density *= compute_args.rest_density;
    if (dot(sum, sum) + sum_squared > EPS) {
        i.factor = 1.0 / (dot(sum, sum) + sum_squared);
    } else {
        i.factor = 0.0;
    }
}

kernel void adapt_timestep() {
    device Particle& i = particles[id];
    float k = 0.4;
    if (length(i.velocity) > 0) {
        atomic_fetch_max_explicit(&compute_args.time_step, as_type<uint>(min(k * 2 * compute_args.particle_radius / length(i.velocity), 0.0005)), memory_order_relaxed);
    }
        
}

kernel void non_pressure_forces() {
    particles[id].acceleration = float3(0, 1 * GRAVITY, 0);
}

kernel void update_positions() {
    particles[id].position += particles[id].velocity * delta_time;
    float y_edge = compute_args.grid_dims.y;
    if (particles[id].position.y - compute_args.kernel_radius < 0) {
        particles[id].position.y = compute_args.kernel_radius;
        particles[id].velocity.y *= -0.5;
    }
    if (particles[id].position.y + compute_args.kernel_radius > y_edge) {
        particles[id].position.y = y_edge - compute_args.kernel_radius;
        particles[id].velocity.y *= -0.5;
    }
    float x_edge = compute_args.grid_dims.x;
    if (particles[id].position.x - compute_args.kernel_radius < 0) {
        particles[id].position.x = compute_args.kernel_radius;
        particles[id].velocity.x *= -0.5;
    }
    if (particles[id].position.x + compute_args.kernel_radius > x_edge) {
        particles[id].position.x = x_edge - compute_args.kernel_radius;
        particles[id].velocity.x *= -0.5;
    }
    float z_edge = compute_args.grid_dims.z;
    if (particles[id].position.z - compute_args.kernel_radius < 0) {
        particles[id].position.z = compute_args.kernel_radius;
        particles[id].velocity.z *= -0.5;
    }
    if (particles[id].position.z + compute_args.kernel_radius > z_edge) {
        particles[id].position.z = z_edge - compute_args.kernel_radius;
        particles[id].velocity.z *= -0.5;
    }
}

kernel void update_velocities() {
    particles[id].velocity += delta_time * particles[id].acceleration;
}

kernel void correct_density_error() {
    device Particle& i = particles[id];
    float pressure_force = delta_time * delta_time * density_pressure_force(id);
    float residiuum = min(1.0 - i.density_adv - pressure_force, 0.0f);
    //float residiuum = 1 - i.density_adv - pressure_force;
    i.pressure_rho2 -= residiuum * i.factor;
    atomic_fetch_sub_explicit(&compute_args.density_error, compute_args.rest_density * residiuum/compute_args.num_particles, memory_order_relaxed);
}

kernel void compute_density_adv() {
    device Particle& i = particles[id];
    float delta = 0.0f;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        delta += compute_args.volume * dot(gradient_kernel(i, j, compute_args.kernel_radius), i.velocity-j.velocity);
    }
    
    i.density_adv = i.density / compute_args.rest_density + delta_time * delta;
    
    i.factor *= (1.0/pow(delta_time, 2.0));
    float residiuum = min(0.0, 1.0 - i.density_adv);
    i.pressure_rho2 =  -residiuum * i.factor;
}

kernel void compute_density_change() {
    device Particle& i = particles[id];
    i.density_adv = 0.0f;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        i.density_adv += compute_args.volume * dot(gradient_kernel(i, j, compute_args.kernel_radius), i.velocity-j.velocity);
    }
    
    i.factor *= 1/delta_time;
    i.pressure_rho2v = max(i.density_adv, 0.0) * i.factor;
}

kernel void correct_divergence_error() {
    device Particle& i = particles[id];
    float pressure_force = delta_time * density_pressure_force(id);
    float residiuum = min(-i.density_adv - pressure_force, 0.0);
    //float residiuum = 1 - i.density_adv - pressure_force;
    i.pressure_rho2v -= residiuum * i.factor;
    atomic_fetch_sub_explicit(&compute_args.density_error, compute_args.rest_density * residiuum/compute_args.num_particles, memory_order_relaxed);
}

kernel void update_velocities_from_pressure() {
    particles[id].velocity += delta_time * particles[id].pressure_acceleration;
    particles[id].factor *= delta_time;
}
