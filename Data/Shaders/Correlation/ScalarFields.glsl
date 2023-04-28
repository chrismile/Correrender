#ifndef COMBINE_CORRELATION_MEMBERS
layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, cs;
#ifdef USE_SECONDARY_FIELDS
    uint xsr, ysr, zsr, paddingUniform0;
    uint xsq, ysq, zsq, paddingUniform1;
#endif
};
#else
layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, cs;
    vec3 boundingBoxMin;
    float minFieldVal;
    vec3 boundingBoxMax;
    float maxFieldVal;
};
#endif
#if !defined(COMBINE_CORRELATION_MEMBERS) && !defined(USE_SECONDARY_FIELDS)
#define xsr xs
#define ysr ys
#define zsr zs
#define xsq xs
#define ysq ys
#define zsq zs
#endif

#ifdef SUPPORT_TILING

#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define TILE_SIZE_Z 4
uint IDXS(uint x, uint y, uint z) {
    uint xst = (xs - 1) / TILE_SIZE_X + 1;
    uint yst = (ys - 1) / TILE_SIZE_Y + 1;
    //uint zst = (zs - 1) / TILE_SIZE_Z + 1;
    uint xt = x / TILE_SIZE_X;
    uint yt = y / TILE_SIZE_Y;
    uint zt = z / TILE_SIZE_Z;
    uint tileAddressLinear = (xt + yt * xst + zt * xst * yst) * (TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z);
    uint vx = x & (TILE_SIZE_X - 1u);
    uint vy = y & (TILE_SIZE_Y - 1u);
    uint vz = z & (TILE_SIZE_Z - 1u);
    uint voxelAddressLinear = vx + vy * TILE_SIZE_X + vz * TILE_SIZE_X * TILE_SIZE_Y;
    return tileAddressLinear | voxelAddressLinear;
}

#ifdef USE_SECONDARY_FIELDS
uint IDXSR(uint x, uint y, uint z) {
    uint xst = (xsr - 1) / TILE_SIZE_X + 1;
    uint yst = (ysr - 1) / TILE_SIZE_Y + 1;
    //uint zst = (zs - 1) / TILE_SIZE_Z + 1;
    uint xt = x / TILE_SIZE_X;
    uint yt = y / TILE_SIZE_Y;
    uint zt = z / TILE_SIZE_Z;
    uint tileAddressLinear = (xt + yt * xst + zt * xst * yst) * (TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z);
    uint vx = x & (TILE_SIZE_X - 1u);
    uint vy = y & (TILE_SIZE_Y - 1u);
    uint vz = z & (TILE_SIZE_Z - 1u);
    uint voxelAddressLinear = vx + vy * TILE_SIZE_X + vz * TILE_SIZE_X * TILE_SIZE_Y;
    return tileAddressLinear | voxelAddressLinear;
}
uint IDXSQ(uint x, uint y, uint z) {
    uint xst = (xsq - 1) / TILE_SIZE_X + 1;
    uint yst = (ysq - 1) / TILE_SIZE_Y + 1;
    //uint zst = (zs - 1) / TILE_SIZE_Z + 1;
    uint xt = x / TILE_SIZE_X;
    uint yt = y / TILE_SIZE_Y;
    uint zt = z / TILE_SIZE_Z;
    uint tileAddressLinear = (xt + yt * xst + zt * xst * yst) * (TILE_SIZE_X * TILE_SIZE_Y * TILE_SIZE_Z);
    uint vx = x & (TILE_SIZE_X - 1u);
    uint vy = y & (TILE_SIZE_Y - 1u);
    uint vz = z & (TILE_SIZE_Z - 1u);
    uint voxelAddressLinear = vx + vy * TILE_SIZE_X + vz * TILE_SIZE_X * TILE_SIZE_Y;
    return tileAddressLinear | voxelAddressLinear;
}
#else
#define IDXSQ IDXS
#define IDXSR IDXS
#endif

#else

#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))
#ifdef USE_SECONDARY_FIELDS
#define IDXSR(x,y,z) ((z)*xsr*ysr + (y)*xsr + (x))
#define IDXSQ(x,y,z) ((z)*xsq*ysq + (y)*xsq + (x))
#else
#define IDXSR(x,y,z) ((z)*xs*ys + (y)*xs + (x))
#define IDXSQ(x,y,z) ((z)*xs*ys + (y)*xs + (x))
#endif

#endif

#ifdef USE_SCALAR_FIELD_IMAGES
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFields[MEMBER_COUNT];
#else
layout (binding = 2, std430) readonly buffer ScalarFieldBuffers {
    float values[];
} scalarFields[MEMBER_COUNT];
#endif
#define scalarFieldsRef scalarFields

#ifdef USE_SECONDARY_FIELDS
#ifdef USE_SCALAR_FIELD_IMAGES
layout (binding = 7) uniform texture3D scalarFieldsSecondary[MEMBER_COUNT];
#else
layout (binding = 7, std430) readonly buffer ScalarFieldBuffersSecondary {
    float values[];
} scalarFieldsSecondary[MEMBER_COUNT];
#endif
#define scalarFieldsQuery scalarFieldsSecondary
#else
#define scalarFieldsQuery scalarFields
#endif

#ifdef SEPARATE_REFERENCE_AND_QUERY_FIELDS
layout (binding = 4, std430) readonly buffer ReferenceValuesBuffer {
#if !defined(KENDALL_RANK_CORRELATION) && !defined(MUTUAL_INFORMATION_KRASKOV)
    float referenceValues[MEMBER_COUNT];
#else
    float referenceValuesOrig[MEMBER_COUNT];
#endif
};
#endif
