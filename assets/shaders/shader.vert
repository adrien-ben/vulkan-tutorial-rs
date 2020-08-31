#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 vPosition;
layout(location = 1) in vec3 vColor;
layout(location = 2) in vec2 vCoords;

layout(binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
} camera;

layout(push_constant) uniform Model {
    mat4 matrix;
} model;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragCoords;

void main() {
    gl_Position = camera.proj * camera.view * model.matrix * vec4(vPosition, 1.0);
    fragColor = vColor;
    fragCoords = vCoords;
}
