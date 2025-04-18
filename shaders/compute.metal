#include <metal_stdlib>
using namespace metal;
#include "shader_types.h"

#define mass3 0
#define GRAVITY -9.8
#define num3_particles 1000
//#define time_step 0.000001
//#define H 16
#define delta_time as_type<float>(  atomic_load_explicit(&compute_args.time_step, memory_order_relaxed))


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


kernel void compute_densities_and_factors(device Particle *particles [[buffer(0)]], device ComputeArguments& compute_args [[buffer(1)]], uint id [[thread_position_in_grid]]) {
    device Particle& i = particles[id];
    i.density = 0;
    float3 sum = 0;
    float sum_squared = 0;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        
        Particle j = particles[j_id];
        i.density += compute_args.mass * kernal(i, j, compute_args.kernel_radius);
        if (j_id == id) {
            continue;
        }
        float3 gradient = compute_args.mass * gradient_kernel(i, j, compute_args.kernel_radius);
        sum += gradient;
        sum_squared += dot(gradient, gradient);
    }
    i.factor = 1./max(dot(sum, sum) + sum_squared, 1e-6f);
}

kernel void adapt_timestep(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    float k = 0.4;
    float particle_radius = 0.01 * 2; // ???
    if (length(i.velocity) > 0) {
        atomic_fetch_max_explicit(&compute_args.time_step, as_type<uint>(min(k * particle_radius / length(i.velocity), 0.001)), memory_order_relaxed);
    }
        
}

kernel void non_pressure_forces(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    particles[id].force.y = compute_args.mass * GRAVITY;
}

kernel void update_positions(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    particles[id].position += particles[id].pred_velocity * delta_time;
    if (particles[id].position.y - compute_args.kernel_radius < 0) {
        particles[id].position.y = compute_args.kernel_radius;
        particles[id].pred_velocity.y *= -0.5;
    }
    if (particles[id].position.x - compute_args.kernel_radius < 0) {
        particles[id].position.x = compute_args.kernel_radius;
        particles[id].pred_velocity.x *= -0.5;
    }
    float x_edge = 1;
    if (particles[id].position.x + compute_args.kernel_radius > x_edge) {
        particles[id].position.x = x_edge- compute_args.kernel_radius;
        particles[id].pred_velocity.x *= -0.5;
    }
    if (particles[id].position.z - compute_args.kernel_radius < 0) {
            particles[id].position.z = compute_args.kernel_radius;
            particles[id].pred_velocity.z *= -0.5;
    }
    float z_edge = 1;
    if (particles[id].position.z + compute_args.kernel_radius > z_edge) {
        particles[id].position.z = z_edge- compute_args.kernel_radius;
        particles[id].pred_velocity.z *= -0.5;
    }
}

kernel void update_velocities(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    particles[id].velocity = particles[id].pred_velocity;
}

kernel void predict_velocities(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    i.pred_velocity = i.velocity + delta_time * i.force / compute_args.mass;
}

kernel void correct_divergence_error(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    float k_i = 1/delta_time * i.density_derivative * i.factor / i.density;
    float3 acceleration = 0;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        float k_j = 1/delta_time * j.density_derivative * j.factor / j.density;
        acceleration += compute_args.mass * (k_i + k_j) * gradient_kernel(i, j, compute_args.kernel_radius);
    }
    i.pred_velocity = i.pred_velocity - acceleration * delta_time;
}

kernel void correct_density_error(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    float k_i = (1./pow(delta_time, 2)) * (i.pred_density - compute_args.rest_density) * i.factor / i.density;
    //float k_i = fmax((1./pow(delta_time, 2)) * (i.pred_density - compute_args.rest_density) * i.factor, 0.25)/i.density;
    float3 acceleration = 0;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        float k_j = (1/pow(delta_time, 2)) * (j.pred_density - compute_args.rest_density) * j.factor / j.density;
        acceleration += compute_args.mass * (k_i + k_j) * gradient_kernel(i, j, compute_args.kernel_radius);
    }
    i.pred_velocity = i.pred_velocity - acceleration * delta_time;
}

kernel void compute_density_pred_derivative(device Particle *particles [[buffer(0)]], uint id [[thread_position_in_grid]], device ComputeArguments& compute_args [[buffer(1)]]) {
    device Particle& i = particles[id];
    i.density_derivative = 0;
    for (int j_id = 0; j_id < compute_args.num_particles; j_id++) {
        if (j_id == id) {
            continue;
        }
        Particle j = particles[j_id];
        i.density_derivative += compute_args.mass * dot(gradient_kernel(i, j, compute_args.kernel_radius), i.pred_velocity-j.pred_velocity);
    }
    //i.pred_density = max(i.density + delta_time * i.density_derivative, compute_args.rest_density);
    i.pred_density = i.density + delta_time * i.density_derivative;
    atomic_fetch_add_explicit(&compute_args.avg_density_derivative, i.density_derivative/compute_args.num_particles, memory_order_relaxed);
    atomic_fetch_add_explicit(&compute_args.avg_pred_density, i.pred_density/compute_args.num_particles, memory_order_relaxed);
}

