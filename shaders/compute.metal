#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

#define GRAVITY -9.81
#define delta_time as_type<float>(atomic_load_explicit(&compute_args.time_step, memory_order_relaxed))
#define EPS 1.0e-5

float kernal(device Particle& i, Particle j, float H) {
    float dist = length(i.position-j.position);
    if (dist > H) {
        return 0.0;
    }
    float q = dist/H;
    return (1.0/(H*H*H*M_PI_F))*(1.0 - 1.5*q*q + 0.75*q*q*q);
}

float3 gradient_kernel(device Particle& i, Particle j, float H) {
    float dist = length(i.position-j.position);
    if (dist > H) {
        return float3(0.0);
    }
    if (dist == 0) {
        return float3(1.0);
    }
    float q = dist/H;
    return (i.position-j.position)*(1.0/(H*H*H*H*M_PI_F*dist))*(- 3.0*q + 2.25*q*q);
}

float density_pressure_force(uint id, device Particle *particles, device ComputeArguments& compute_args) {
    device Particle& i = particles[id];
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


kernel void compute_pressure_accel(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
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


kernel void compute_densities_and_factors(device Particle *particles [[buffer(0)]], device ComputeArguments& compute_args [[buffer(1)]], uint id [[thread_position_in_grid]]) {
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
        float3 gradient = compute_args.volume * gradient_kernel(i, j, compute_args.kernel_radius);
        sum += gradient;
        sum_squared += dot(gradient, gradient);
    }
    i.density *= compute_args.rest_density;
    if (dot(sum, sum) + sum_squared > EPS) {
        i.factor = 1.0 / (dot(sum, sum) + sum_squared);
    } else {
        i.factor = 0.0;
    }
}

kernel void adapt_timestep(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    float k = 0.4;
    if (length(i.velocity) > 0) {
        atomic_fetch_max_explicit(&compute_args.time_step, as_type<uint>(min(k * 2 * compute_args.particle_radius / length(i.velocity), 0.0005)), memory_order_relaxed);
    }
        
}

kernel void non_pressure_forces(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    particles[id].acceleration = float3(0, GRAVITY, 0);
}

kernel void update_positions(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    particles[id].position += particles[id].velocity * delta_time;
    float y_edge = 1.0;
    if (particles[id].position.y - compute_args.kernel_radius < 0) {
        particles[id].position.y = compute_args.kernel_radius;
        particles[id].velocity.y *= -0.5;
    }
    if (particles[id].position.y + compute_args.kernel_radius > y_edge) {
        particles[id].position.y = y_edge - compute_args.kernel_radius;
        particles[id].velocity.y *= -0.5;
    }
    float x_edge = 0.6;
    if (particles[id].position.x - compute_args.kernel_radius < 0) {
        particles[id].position.x = compute_args.kernel_radius;
        particles[id].velocity.x *= -0.5;
    }
    if (particles[id].position.x + compute_args.kernel_radius > x_edge) {
        particles[id].position.x = x_edge - compute_args.kernel_radius;
        particles[id].velocity.x *= -0.5;
    }
    float z_edge = 0.6;
    if (particles[id].position.z - compute_args.kernel_radius < 0) {
        particles[id].position.z = compute_args.kernel_radius;
        particles[id].velocity.z *= -0.5;
    }
    if (particles[id].position.z + compute_args.kernel_radius > z_edge) {
        particles[id].position.z = z_edge - compute_args.kernel_radius;
        particles[id].velocity.z *= -0.5;
    }
}

kernel void update_velocities(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    particles[id].velocity += delta_time * particles[id].acceleration;
}

kernel void correct_density_error(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    float pressure_force = delta_time * delta_time * density_pressure_force(id, particles, compute_args);
    //float residiuum = min(1 - i.advected_density - pressure_force, 0.0f);
    float residiuum = 1 - i.advected_density - pressure_force;
    i.pressure_rho2 -= residiuum * i.factor;
    atomic_fetch_sub_explicit(&compute_args.density_error, compute_args.rest_density * residiuum/compute_args.num_particles, memory_order_relaxed);
}

kernel void compute_density_adv(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    float delta = 0.0f;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        delta += compute_args.volume * dot(gradient_kernel(i, j, compute_args.kernel_radius), i.velocity-j.velocity);
    }
    
    i.advected_density = i.density / compute_args.rest_density + delta_time * delta;
    
    i.factor *= (1/pow(delta_time, 2.0));
    i.pressure_rho2 = -min(0.0, 1.0 - i.advected_density) * i.factor;
}

kernel void update_velocities_from_pressure(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    particles[id].velocity += delta_time * particles[id].pressure_acceleration;
}
