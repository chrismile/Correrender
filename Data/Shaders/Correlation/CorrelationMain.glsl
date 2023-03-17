    // Should be appended to the beginning of the main function of correlation computation code.
#ifdef USE_REQUESTS_BUFFER
    uint requestIdx = gl_GlobalInvocationID.x;
    if (requestIdx >= numRequests) {
        return;
    }
    RequestData request = requests[requestIdx];
    ivec3 referencePointIdx = ivec3(request.xi, request.yi, request.zi);
    ivec3 currentPointIdx = ivec3(request.xj, request.yj, request.zj);
#else
    ivec3 currentPointIdx = ivec3(gl_GlobalInvocationID.xyz + batchOffset);
    if (currentPointIdx.x >= xs || currentPointIdx.y >= ys || currentPointIdx.z >= zs) {
        return;
    }
#endif
