struct RequestData {
    uint xi, yi, zi, i, xj, yj, zj, j;
};
layout(std430, binding = 5) readonly buffer RequestsBuffer {
    RequestData requests[];
};
layout(push_constant) uniform PushConstants {
    uint numRequests, paddingPushConstants;
#ifdef NEEDS_MIN_MAX_FIELD_VALUE
    float minFieldVal, maxFieldVal;
#endif
};
layout(std430, binding = 1) writeonly buffer OutputBuffer {
    float outputBuffer[];
};
