

#include <metal_stdlib>
using namespace metal;

#include "shader_types.h"

struct VertexOut {
    float4 position [[position]];
};

vertex VertexOut vertexShader(uint vertexID [[vertex_id]],
                            uint instanceID [[instance_id]],
             constant float3* vertexData [[buffer(0)]],
             constant Particle* particles [[buffer(1)]],
             constant float4x4& matrix [[buffer(2)]])
{
    VertexOut out;
    out.position = matrix * float4(vertexData[vertexID] + particles[instanceID].position, 1);
    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]
                               ) {
    const float4 colorSample = float4(0,1,0,1);
    return colorSample;
}