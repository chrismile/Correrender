layout (binding = 0) uniform UniformBuffer {
    uint xs, ys, zs, cs;
};

#ifdef SUPPORT_TILING
#define TILE_SIZE_X 8
#define TILE_SIZE_Y 8
#define TILE_SIZE_Z 4
uint IDXS(uint x, uint y, uint z) {
    uint xst = (xs - 1) / TILE_SIZE_X + 1;
    uint yst = (ys - 1) / TILE_SIZE_Y + 1;
    uint zst = (zs - 1) / TILE_SIZE_Z + 1;
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
#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))
#endif

#ifdef USE_SCALAR_FIELD_IMAGES
layout (binding = 2) uniform sampler scalarFieldSampler;
layout (binding = 3) uniform texture3D scalarFields[MEMBER_COUNT];
#else
layout (binding = 2, std430) readonly buffer ScalarFieldBuffers {
    float values[];
} scalarFields[MEMBER_COUNT];
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
