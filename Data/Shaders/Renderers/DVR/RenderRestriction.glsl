#ifdef USE_RENDER_RESTRICTION
layout(binding = RENDER_RESTRICTION_BUFFER_BINDING) uniform RenderRestrictionUniformBuffer {
    vec3 renderRestrictionCenter;
    float renderRestrictionRadius;
};
bool getShouldRender(vec3 position) {
    vec3 diff = position - renderRestrictionCenter;
#ifdef RENDER_RESTRICTION_CHEBYSHEV_DISTANCE
    diff = abs(diff);
    float restrictionDist = max(diff.x, max(diff.y, diff.z));
#else // RENDER_RESTRICTION_EUCLIDEAN_DISTANCE
    float restrictionDist = length(diff);
#endif
    return restrictionDist <= renderRestrictionRadius;
}
#else
#define getShouldRender(position) true
#endif
