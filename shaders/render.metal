#include <metal_stdlib>
using namespace metal;

#include "shader_types.h"

struct VertexOut {
    float4 position [[position]];
    float4 color;
};

vertex VertexOut vertexShader(uint vertexID [[vertex_id]],
                            uint instanceID [[instance_id]],
             constant float3* vertexData [[buffer(0)]],
             constant Particle* particles [[buffer(1)]],
             constant float4x4& matrix [[buffer(2)]]) {
    VertexOut out;
    out.position = matrix * float4(vertexData[vertexID] + particles[instanceID].position, 1);
    out.color = float4(0, particles[instanceID].density/1000, 0, 1);
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    return in.color;
}
