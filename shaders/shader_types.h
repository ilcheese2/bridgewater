#pragma once
#include <simd/vector_types.h>

#ifndef __METAL__
#define atomic_float float
#define atomic_uint uint32_t
#endif

struct Particle {
    vector_float3 position;
    vector_float3 velocity;
    vector_float3 pred_velocity;
    vector_float3 force;
    float density;
    float density_derivative;
    float pred_density;
    float factor;
};

struct ComputeArguments {
    float kernel_radius;
    float mass;
    int num_particles;
    float rest_density;
    atomic_uint time_step;
    atomic_float avg_density_derivative;
    atomic_float avg_pred_density;
};
