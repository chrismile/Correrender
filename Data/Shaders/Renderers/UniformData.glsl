// Binding 0-7 is reserved for the individual rendering modes.
#define VIEW_UNIFORM_DATA_BUFFER_BINDING 8
#define MIN_MAX_BUFFER_BINDING 9 // for TransferFunction.glsl
#define TRANSFER_FUNCTION_TEXTURE_BINDING 10 // for TransferFunction.glsl

/*layout(binding = VIEW_UNIFORM_DATA_BUFFER_BINDING) uniform ViewUniformDataBuffer {
    // Camera data.
    vec3 cameraPosition;
    float fieldOfViewY;
    mat4 viewMatrix;
    mat4 projectionMatrix;
    mat4 inverseViewMatrix;
    mat4 inverseProjectionMatrix;
    vec4 backgroundColor;
    vec4 foregroundColor;

    // Antialiasing.glsl needs a viewport size. Change this value if downsampling/upscaling is used!
    uvec2 viewportSize;
};*/
