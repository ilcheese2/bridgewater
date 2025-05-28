#pragma once
#include <simd/vector_types.h>

#ifndef __METAL__
#define atomic_float float
#define atomic_uint uint32_t
#endif

struct Particle { // https://gist.github.com/entropylost/75a3ff4e0fae22a27b408968de31c5d1
    vector_float3 position;
    vector_float3 velocity;
    vector_float3 pressure_acceleration;
    vector_float3 acceleration;
    float density;
    float density_adv; // ρ* / ρ_0
    float factor;// s p / ρ^2 * ρ_0
    float pressure_rho2;
    float pressure_rho2v;
    int neighbors;
};

struct ComputeArguments {
    float kernel_radius;
    float particle_radius;
    float rest_density;
    float volume;
    int num_particles;
    atomic_uint time_step;
    atomic_float avg_density_derivative;
    atomic_float density_error;
    vector_float3 size;
    vector_int3 grid_dims;
};

struct ParticleLocation {
    unsigned int index;
    unsigned int cell;
};
