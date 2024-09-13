/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

-- Compute

#version 450 core

/*
 * - Overview over different analytic and iterative solvers:
 *   https://www.sci.utah.edu/~wald/Publications/2004/iso/IsoIsec_VMV2004.pdf
 * - Overview of analytic cubic solver with algorithm by Schwarze:
 *   https://www.ppsloan.org/publications/iso98.pdf
     For more details on solver, see: https://dl.acm.org/doi/abs/10.5555/90767.90877
 */

layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0) uniform RendererUniformDataBuffer {
    mat4 inverseViewMatrix;
    mat4 inverseProjectionMatrix;
    vec3 cameraPosition;
    float dx;
    float dy;
    float dz;
    float zNear;
    float zFar;
    vec3 minBoundingBox;
    float isoValue;
    vec3 maxBoundingBox;
    float stepSize;
    vec4 isoSurfaceColor;
    vec3 voxelTexelSize;
};

layout (binding = 1, rgba32f) uniform image2D outputImage;
layout (binding = 2) uniform sampler3D scalarField;

#ifdef SUPPORT_DEPTH_BUFFER

layout (binding = 3, r32f) uniform image2D depthBuffer;
float closestDepth, closestDepthNew;
#include "DepthHelper.glsl"

#ifdef SUPPORT_NORMAL_BUFFER
layout (binding = 4, rgba32f) uniform image2D normalBuffer;
vec3 surfaceNormal;
bool normalSet;
#endif

#endif

ivec3 gridSize;

#include "RayIntersectionTests.glsl"
#include "Blending.glsl"
#include "UniformData.glsl"
#include "Lighting.glsl"

#ifdef USE_RENDER_RESTRICTION
#include "RenderRestriction.glsl"
#endif

#define DIFFERENCES_NEIGHBOR
vec3 computeGradient(vec3 texCoords) {
#ifdef DIFFERENCES_NEIGHBOR
    float gradX =
            (textureOffset(scalarField, texCoords, ivec3(-1, 0, 0)).r
            - textureOffset(scalarField, texCoords, ivec3(1, 0, 0)).r) * 0.5 / dx;
    float gradY =
            (textureOffset(scalarField, texCoords, ivec3(0, -1, 0)).r
            - textureOffset(scalarField, texCoords, ivec3(0, 1, 0)).r) * 0.5 / dy;
    float gradZ =
            (textureOffset(scalarField, texCoords, ivec3(0, 0, -1)).r
            - textureOffset(scalarField, texCoords, ivec3(0, 0, 1)).r) * 0.5 / dz;
#else
    const float dxp = voxelTexelSize.x * 1;
    const float dyp = voxelTexelSize.y * 1;
    const float dzp = voxelTexelSize.z * 1;
    float gradX =
            (texture(scalarField, texCoords - vec3(dxp, 0.0, 0.0)).r
            - texture(scalarField, texCoords + vec3(dxp, 0.0, 0.0)).r) * 0.5;
    float gradY =
            (texture(scalarField, texCoords - vec3(0.0, dyp, 0.0)).r
            - texture(scalarField, texCoords + vec3(0.0, dyp, 0.0)).r) * 0.5;
    float gradZ =
            (texture(scalarField, texCoords - vec3(0.0, 0.0, dzp)).r
            - texture(scalarField, texCoords + vec3(0.0, 0.0, dzp)).r) * 0.5;
#endif
    return normalize(vec3(gradX, gradY, gradZ));
}

const int MAX_NUM_REFINEMENT_STEPS = 8;

void refineIsoSurfaceHit(inout vec3 currentPoint, vec3 rayDirection, float stepSign) {
    vec3 lastPoint = currentPoint - stepSize * rayDirection;
    for (int i = 0; i < MAX_NUM_REFINEMENT_STEPS; i++) {
        vec3 midPoint = (currentPoint + lastPoint) * 0.5;
        vec3 texCoordsMidPoint = (midPoint - minBoundingBox) / (maxBoundingBox - minBoundingBox);
        float scalarValueMidPoint = texture(scalarField, texCoordsMidPoint).r;
        if ((scalarValueMidPoint - isoValue) * stepSign >= 0.0) {
            currentPoint = midPoint;
        } else {
            lastPoint = midPoint;
        }
    }
}

vec4 getIsoSurfaceHitColor(
        vec3 currentPoint
#if defined(USE_INTERPOLATION_NEAREST_NEIGHBOR) && defined(ANALYTIC_INTERSECTIONS)
        , vec3 gradient
#endif
#if defined(CLOSE_ISOSURFACES) && !defined(ANALYTIC_INTERSECTIONS)
        , bool isFirstPoint, vec3 entryNormal
#endif
) {
    //vec4 volumeColor = transferFunction(scalarValue);
    vec3 texCoords = (currentPoint - minBoundingBox) / (maxBoundingBox - minBoundingBox);
#if !defined(USE_INTERPOLATION_NEAREST_NEIGHBOR) || !defined(ANALYTIC_INTERSECTIONS)
    vec3 gradient = computeGradient(texCoords);
#endif
#if defined(CLOSE_ISOSURFACES) && !defined(ANALYTIC_INTERSECTIONS)
    if (isFirstPoint) {
        gradient = entryNormal;
    }
#endif
#ifdef SUPPORT_NORMAL_BUFFER
    if (!normalSet) {
        surfaceNormal = gradient;
        normalSet = true;
    }
#endif
    
    vec4 color = blinnPhongShadingSurface(isoSurfaceColor, currentPoint, gradient);
    //return vec4(vec3(0.5) + currentPoint * 2.0, 1.0);
#ifdef SUPPORT_DEPTH_BUFFER
    closestDepthNew = min(closestDepthNew, length(currentPoint - cameraPosition));
#endif
    return color;
}

#ifdef ANALYTIC_INTERSECTIONS
vec3 worldToGridPos(vec3 worldPos) {
    return (worldPos - minBoundingBox) / (maxBoundingBox - minBoundingBox) * gridSize - vec3(0.5);
}
vec3 worldToGridDir(vec3 worldDir) {
    return normalize(worldDir / (maxBoundingBox - minBoundingBox) * gridSize);
}
vec3 gridToWorldPos(vec3 gridPos) {
    return (gridPos + vec3(0.5)) / gridSize * (maxBoundingBox - minBoundingBox) + minBoundingBox;
}
vec3 gridToWorldDir(vec3 gridDir) {
    return normalize(gridDir / gridSize * (maxBoundingBox - minBoundingBox));
}
vec3 gridToTexCoord(vec3 gridPos) {
    return (gridPos + vec3(0.5)) / gridSize;
}

/**
 * Solves a cubic equation. Source: https://github.com/erich666/GraphicsGems/blob/master/gems/Roots3And4.c
 *
 * License:
 * "EULA: The Graphics Gems code is copyright-protected. In other words, you cannot claim the text of the code as your
 * own and resell it. Using the code is permitted in any program, product, or library, non-commercial or commercial.
 * Giving credit is not required, though is a nice gesture. The code comes as-is, and if there are any flaws or problems
 * with any Gems code, nobody involved with Gems - authors, editors, publishers, or webmasters - are to be held
 * responsible. Basically, don't be a jerk, and remember that anything free comes with no guarantee."
 */
#define     M_PI        3.14159265358979323846
#define     EQN_EPS     1e-5
#define	    IsZero(x)	((x) > -EQN_EPS && (x) < EQN_EPS)
#define     cbrt(x)     ((x) > 0.0 ? pow(x, 1.0/3.0) : \
                          ((x) < 0.0 ? -pow(-(x), 1.0/3.0) : 0.0))
int solveCubic(float c[4], float s[3]) {
    int     i, num;
    float  sub;
    float  A, B, C;
    float  sq_A, p, q;
    float  cb_p, D;

    /* normal form: x^3 + Ax^2 + Bx + C = 0 */
    A = c[ 2 ] / c[ 3 ];
    B = c[ 1 ] / c[ 3 ];
    C = c[ 0 ] / c[ 3 ];

    /*  substitute x = y - A/3 to eliminate quadric term:
    x^3 +px + q = 0 */
    sq_A = A * A;
    p = 1.0/3 * (- 1.0/3 * sq_A + B);
    q = 1.0/2 * (2.0/27 * A * sq_A - 1.0/3 * A * B + C);

    /* use Cardano's formula */
    cb_p = p * p * p;
    D = q * q + cb_p;

    if (IsZero(D))
    {
        if (IsZero(q)) /* one triple solution */
        {
            s[ 0 ] = 0;
            num = 1;
        }
        else /* one single and one float solution */
        {
            float u = cbrt(-q);
            s[ 0 ] = 2 * u;
            s[ 1 ] = - u;
            num = 2;
        }
    }
    else if (D < 0) /* Casus irreducibilis: three real solutions */
    {
        float phi = 1.0/3 * acos(-q / sqrt(-cb_p));
        float t = 2 * sqrt(-p);

        s[ 0 ] =   t * cos(phi);
        s[ 1 ] = - t * cos(phi + M_PI / 3);
        s[ 2 ] = - t * cos(phi - M_PI / 3);
        num = 3;
    }
    else /* one real solution */
    {
        float sqrt_D = sqrt(D);
        float u = cbrt(sqrt_D - q);
        float v = - cbrt(sqrt_D + q);

        s[ 0 ] = u + v;
        num = 1;
    }

    /* resubstitute */
    sub = 1.0/3 * A;
    for (i = 0; i < num; ++i)
        s[ i ] -= sub;

    return num;
}

bool findHit(
        out float tHit, float t0, float t1, float f0, float f1, vec3 startPoint, vec3 rayDirection,
        float A, float B, float C) {
    float majorant = 4.0 * B * B - 6.0 * A * C;
    if (majorant >= 0.0) {
        float majorantSqrt = sqrt(majorant);
        float root0 = (-2.0 * B - majorantSqrt) / (6.0 * A);
        float root1 = (-2.0 * B + majorantSqrt) / (6.0 * A);

        float e0 = min(root0, root1);
        float f_e0 = texture(scalarField, gridToTexCoord(startPoint + e0 * rayDirection)).r - isoValue;
        if (t0 <= e0 && e0 <= t1) {
            if (sign(f_e0) == sign(f0)) {
                t0 = e0;
                f0 = f_e0;
            } else {
                t1 = e0;
                f1 = f_e0;
            }
        }

        float e1 = max(root0, root1);
        float f_e1 = texture(scalarField, gridToTexCoord(startPoint + e1 * rayDirection)).r - isoValue;
        if (t0 <= e1 && e1 <= t1) {
            if (sign(f_e1) == sign(f0)) {
                t0 = e1;
                f0 = f_e1;
            } else {
                t1 = e1;
                f1 = f_e1;
            }
        }
    }

    if (sign(f0) == sign(f1)) {
        return false;
    }

    const int N = 10;
    for (int i= 0; i < N; i++) {
        float t = t0 + (t1 - t0) * (-f0) / (f1 - f0);
        float f_Rt = texture(scalarField, gridToTexCoord(startPoint + t * rayDirection)).r - isoValue;
        if (sign(f_Rt) == sign(f0)) {
            t0 = t;
            f0 = f_Rt;
        } else {
            t1 = t;
            f1 = f_Rt;
        }
    }

    tHit = t0 + (t1 - t0) * (-f0) / (f1 - f0);
    return true;
}

bool findHitManual(
        out float tHit, float t0, float t1, float f0, float f1,
        float A, float B, float C, float D) {
    float majorant = 4.0 * B * B - 6.0 * A * C;
    if (majorant >= 0.0) {
        float majorantSqrt = sqrt(majorant);
        float root0 = (-2.0 * B - majorantSqrt) / (6.0 * A);
        float root1 = (-2.0 * B + majorantSqrt) / (6.0 * A);

        float e0 = min(root0, root1);
        float f_e0 = ((A * e0 + B) * e0 + C) * e0 + D;
        if (t0 <= e0 && e0 <= t1) {
            if (sign(f_e0) == sign(f0)) {
                t0 = e0;
                f0 = f_e0;
            } else {
                t1 = e0;
                f1 = f_e0;
            }
        }

        float e1 = max(root0, root1);
        float f_e1 = ((A * e1 + B) * e1 + C) * e1 + D;
        if (t0 <= e1 && e1 <= t1) {
            if (sign(f_e1) == sign(f0)) {
                t0 = e1;
                f0 = f_e1;
            } else {
                t1 = e1;
                f1 = f_e1;
            }
        }
    }

    if (sign(f0) == sign(f1)) {
        return false;
    }

    const int N = 10;
    for (int i= 0; i < N; i++) {
        float t = t0 + (t1 - t0) * (-f0) / (f1 - f0);
        float f_Rt = ((A * t + B) * t + C) * t + D;
        if (sign(f_Rt) == sign(f0)) {
            t0 = t;
            f0 = f_Rt;
        } else {
            t1 = t;
            f1 = f_Rt;
        }
    }

    tHit = t0 + (t1 - t0) * (-f0) / (f1 - f0);
    return true;
}

vec4 texelFetchZeroBorder(ivec3 voxelIndexOffset, int lodLevel) {
    if (all(greaterThanEqual(voxelIndexOffset, ivec3(0))) && all(lessThan(voxelIndexOffset, gridSize))) {
        return texelFetch(scalarField, voxelIndexOffset, lodLevel);
    }
    return vec4(0.0);
}

vec4 texelFetchClamp(ivec3 voxelIndexOffset, int lodLevel) {
    voxelIndexOffset = clamp(voxelIndexOffset, ivec3(0), gridSize - ivec3(1));
    return texelFetch(scalarField, voxelIndexOffset, lodLevel);
}

vec4 traverseVoxelGridAnalytic(vec3 rayDirection, vec3 startPoint, vec3 endPoint) {
    vec4 outputColor = vec4(0.0);
    float maxDist = length(endPoint - startPoint);

    float tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ;
    ivec3 voxelIndex;

    int stepX = int(sign(endPoint.x - startPoint.x));
    if (stepX != 0)
        tDeltaX = min(stepX / (endPoint.x - startPoint.x), 1e7);
    else
        tDeltaX = 1e7; // inf
    if (stepX > 0)
        tMaxX = tDeltaX * (1.0 - fract(startPoint.x));
    else
        tMaxX = tDeltaX * fract(startPoint.x);
    voxelIndex.x = int(floor(startPoint.x));

    int stepY = int(sign(endPoint.y - startPoint.y));
    if (stepY != 0)
        tDeltaY = min(stepY / (endPoint.y - startPoint.y), 1e7);
    else
        tDeltaY = 1e7; // inf
    if (stepY > 0)
        tMaxY = tDeltaY * (1.0 - fract(startPoint.y));
    else
        tMaxY = tDeltaY * fract(startPoint.y);
    voxelIndex.y = int(floor(startPoint.y));

    int stepZ = int(sign(endPoint.z - startPoint.z));
    if (stepZ != 0)
        tDeltaZ = min(stepZ / (endPoint.z - startPoint.z), 1e7);
    else
        tDeltaZ = 1e7; // inf
    if (stepZ > 0)
        tMaxZ = tDeltaZ * (1.0 - fract(startPoint.z));
    else
        tMaxZ = tDeltaZ * fract(startPoint.z);
    voxelIndex.z = int(floor(startPoint.z));

    if (stepX == 0 && stepY == 0 && stepZ == 0) {
        return vec4(0.0);
    }
    ivec3 step = ivec3(stepX, stepY, stepZ);
    vec3 tMax = vec3(tMaxX, tMaxY, tMaxZ);
    vec3 tDelta = vec3(tDeltaX, tDeltaY, tDeltaZ);

#ifdef USE_INTERPOLATION_NEAREST_NEIGHBOR
    ivec3 stepDirCurr = ivec3(0, 0, 0);
    int dirInit;
    vec3 voxelGridLower = minBoundingBox;
    vec3 voxelGridUpper = maxBoundingBox;
    if (rayBoxIntersectionRayCoordsDir(startPoint, rayDirection, voxelGridLower, voxelGridUpper, dirInit)) {
        stepDirCurr[dirInit] = step[dirInit];
    }
#endif

    int testIt = 0;

    while (all(greaterThanEqual(voxelIndex, ivec3(-1))) && all(lessThan(voxelIndex, gridSize))) {
#ifdef SUPPORT_DEPTH_BUFFER
        vec3 currentPoint = gridToWorldPos(voxelIndex);
        if (length(currentPoint - cameraPosition) >= closestDepth) {
            break;
        }
#endif

#ifndef USE_INTERPOLATION_NEAREST_NEIGHBOR
        float f000 = texelFetchClamp(voxelIndex + ivec3(0,0,0), 0).r;
        float f100 = texelFetchClamp(voxelIndex + ivec3(1,0,0), 0).r;
        float f010 = texelFetchClamp(voxelIndex + ivec3(0,1,0), 0).r;
        float f110 = texelFetchClamp(voxelIndex + ivec3(1,1,0), 0).r;
        float f001 = texelFetchClamp(voxelIndex + ivec3(0,0,1), 0).r;
        float f101 = texelFetchClamp(voxelIndex + ivec3(1,0,1), 0).r;
        float f011 = texelFetchClamp(voxelIndex + ivec3(0,1,1), 0).r;
        float f111 = texelFetchClamp(voxelIndex + ivec3(1,1,1), 0).r;
        //float f000 = texelFetch(scalarField, voxelIndex + ivec3(0,0,0), 0).r;
        //float f100 = texelFetch(scalarField, voxelIndex + ivec3(1,0,0), 0).r;
        //float f010 = texelFetch(scalarField, voxelIndex + ivec3(0,1,0), 0).r;
        //float f110 = texelFetch(scalarField, voxelIndex + ivec3(1,1,0), 0).r;
        //float f001 = texelFetch(scalarField, voxelIndex + ivec3(0,0,1), 0).r;
        //float f101 = texelFetch(scalarField, voxelIndex + ivec3(1,0,1), 0).r;
        //float f011 = texelFetch(scalarField, voxelIndex + ivec3(0,1,1), 0).r;
        //float f111 = texelFetch(scalarField, voxelIndex + ivec3(1,1,1), 0).r;
        bool hasHit =
                (f000 <= isoValue || f100 <= isoValue || f010 <= isoValue || f110 <= isoValue || f001 <= isoValue || f101 <= isoValue || f011 <= isoValue || f111 <= isoValue)
                && (f000 >= isoValue || f100 >= isoValue || f010 >= isoValue || f110 >= isoValue || f001 >= isoValue || f101 >= isoValue || f011 >= isoValue || f111 >= isoValue);
        if (hasHit) {
            float tNearHit = 0.0;
            float tFarHit = 0.0;
            bool hit2 = rayBoxIntersectionRayCoords(startPoint, rayDirection, voxelIndex, voxelIndex + ivec3(1), tNearHit, tFarHit);
            vec3 p0 = startPoint + tNearHit * rayDirection;
            vec3 p1 = startPoint + tFarHit * rayDirection;
            //if (!hit2) {
            //    return vec4(1.0, 1.0, 0.0, 1.0);
            //}

            vec3 a0 = voxelIndex + vec3(1.0) - startPoint;
            vec3 b0 = -rayDirection;
            vec3 a1 = startPoint - voxelIndex;
            vec3 b1 = rayDirection;

            float A =
                      b0.x * b0.y * b0.z * f000
                    + b1.x * b0.y * b0.z * f100
                    + b0.x * b1.y * b0.z * f010
                    + b1.x * b1.y * b0.z * f110
                    + b0.x * b0.y * b1.z * f001
                    + b1.x * b0.y * b1.z * f101
                    + b0.x * b1.y * b1.z * f011
                    + b1.x * b1.y * b1.z * f111;
            float B =
                      (a0.x * b0.y * b0.z + b0.x * a0.y * b0.z + b0.x * b0.y * a0.z) * f000
                    + (a1.x * b0.y * b0.z + b1.x * a0.y * b0.z + b1.x * b0.y * a0.z) * f100
                    + (a0.x * b1.y * b0.z + b0.x * a1.y * b0.z + b0.x * b1.y * a0.z) * f010
                    + (a1.x * b1.y * b0.z + b1.x * a1.y * b0.z + b1.x * b1.y * a0.z) * f110
                    + (a0.x * b0.y * b1.z + b0.x * a0.y * b1.z + b0.x * b0.y * a1.z) * f001
                    + (a1.x * b0.y * b1.z + b1.x * a0.y * b1.z + b1.x * b0.y * a1.z) * f101
                    + (a0.x * b1.y * b1.z + b0.x * a1.y * b1.z + b0.x * b1.y * a1.z) * f011
                    + (a1.x * b1.y * b1.z + b1.x * a1.y * b1.z + b1.x * b1.y * a1.z) * f111;
            float C =
                      (b0.x * a0.y * a0.z + a0.x * b0.y * a0.z + a0.x * a0.y * b0.z) * f000
                    + (b1.x * a0.y * a0.z + a1.x * b0.y * a0.z + a1.x * a0.y * b0.z) * f100
                    + (b0.x * a1.y * a0.z + a0.x * b1.y * a0.z + a0.x * a1.y * b0.z) * f010
                    + (b1.x * a1.y * a0.z + a1.x * b1.y * a0.z + a1.x * a1.y * b0.z) * f110
                    + (b0.x * a0.y * a1.z + a0.x * b0.y * a1.z + a0.x * a0.y * b1.z) * f001
                    + (b1.x * a0.y * a1.z + a1.x * b0.y * a1.z + a1.x * a0.y * b1.z) * f101
                    + (b0.x * a1.y * a1.z + a0.x * b1.y * a1.z + a0.x * a1.y * b1.z) * f011
                    + (b1.x * a1.y * a1.z + a1.x * b1.y * a1.z + a1.x * a1.y * b1.z) * f111;
            float D =
                    -isoValue
                    + a0.x * a0.y * a0.z * f000
                    + a1.x * a0.y * a0.z * f100
                    + a0.x * a1.y * a0.z * f010
                    + a1.x * a1.y * a0.z * f110
                    + a0.x * a0.y * a1.z * f001
                    + a1.x * a0.y * a1.z * f101
                    + a0.x * a1.y * a1.z * f011
                    + a1.x * a1.y * a1.z * f111;

//#define SOLVER_SCHWARZE
//#define SOLVER_LINEAR_INTERPOLATION
//#define SOLVER_LINEAR_INTERPOLATION_MANUAL
//#define SOLVER_NEUBAUER
//#define SOLVER_MARMITT
#if defined(SOLVER_SCHWARZE)
            float c[4];
            float s[3];
            c[0] = D;
            c[1] = C;
            c[2] = B;
            c[3] = A;
            int numSolutions = solveCubic(c, s);
            if (numSolutions > 0) {
                //vec4 color = getIsoSurfaceHitColor(currentPoint);
                //vec4 color = vec4(0.5, 0.5, 0.5, 1.0);
                vec4 color = vec4(vec3(abs(s[0])), 1.0);
                if (blend(color, outputColor)) {
                    // Early ray termination.
                    return outputColor;
                }
            }
#elif defined(SOLVER_LINEAR_INTERPOLATION)
            vec3 tex0 = gridToTexCoord(p0);
            vec3 tex1 = gridToTexCoord(p1);
            float rho0 = texture(scalarField, tex0).r;
            float rho1 = texture(scalarField, tex1).r;
            if (sign(rho0 - isoValue) != sign(rho1 - isoValue)) {
                float tHit = tNearHit + (tFarHit - tNearHit) * (isoValue - rho0) / (rho1 - rho0);
                vec3 currentPoint = gridToWorldPos(startPoint + tHit * rayDirection);
                if (all(greaterThan(currentPoint, minBoundingBox)) && all(lessThan(currentPoint, maxBoundingBox))) {
#ifdef USE_RENDER_RESTRICTION
                    if (getShouldRender(currentPoint)) {
#endif
                    vec4 color = getIsoSurfaceHitColor(currentPoint);
                    if (blend(color, outputColor)) {
                        // Early ray termination.
                        return outputColor;
                    }
#ifdef USE_RENDER_RESTRICTION
                    }
#endif
                }
            }
#elif defined(SOLVER_LINEAR_INTERPOLATION_MANUAL)
            float tFarHitLocal = tFarHit - tNearHit;
            float tNearHitLocal = 0.0;
            float f0 = ((A * tNearHit + B) * tNearHit + C) * tNearHit + D;
            float f1 = ((A * tFarHit + B) * tFarHit + C) * tFarHit + D;
            if (sign(f0) != sign(f1)) {
                float tHit = tNearHit + (tFarHit - tNearHit) * f0 / (f0 - f1);
                vec3 currentPoint = gridToWorldPos(startPoint + tHit * rayDirection);
                if (all(greaterThan(currentPoint, minBoundingBox)) && all(lessThan(currentPoint, maxBoundingBox))) {
#ifdef USE_RENDER_RESTRICTION
                    if (getShouldRender(currentPoint)) {
#endif
                    vec4 color = getIsoSurfaceHitColor(currentPoint);
                    if (blend(color, outputColor)) {
                        // Early ray termination.
                        return outputColor;
                    }
#ifdef USE_RENDER_RESTRICTION
                    }
#endif
                }
            }
#elif defined(SOLVER_NEUBAUER)
            float t0 = tNearHit;
            float t1 = tFarHit;
            vec3 tex0 = gridToTexCoord(p0);
            vec3 tex1 = gridToTexCoord(p1);
            float rho0 = texture(scalarField, tex0).r;
            float rho1 = texture(scalarField, tex1).r;
            if (sign(rho0 - isoValue) != sign(rho1 - isoValue)) {
                const int N = 4;
                for (int i = 0; i < N; i++) {
                    float tHit = t0 + (t1 - t0) * (isoValue - rho0) / (rho1 - rho0);
                    float rhoT = texture(scalarField, gridToTexCoord(startPoint + tHit * rayDirection)).r;
                    if (sign(rhoT - isoValue) == sign(rho0 - isoValue)) {
                        t0 = tHit;
                        rho0 = rhoT;
                    } else {
                        t1 = tHit;
                        rho1 = rhoT;
                    }
                }
                float tHit = t0 + (t1 - t0) * (isoValue - rho0) / (rho1 - rho0);
                vec3 currentPoint = gridToWorldPos(startPoint + tHit * rayDirection);
                if (all(greaterThan(currentPoint, minBoundingBox)) && all(lessThan(currentPoint, maxBoundingBox))) {
#ifdef USE_RENDER_RESTRICTION
                    if (getShouldRender(currentPoint)) {
#endif
                    vec4 color = getIsoSurfaceHitColor(currentPoint);
                    if (blend(color, outputColor)) {
                        // Early ray termination.
                        return outputColor;
                    }
#ifdef USE_RENDER_RESTRICTION
                    }
#endif
                }
            }
#elif defined(SOLVER_MARMITT)
            float tHit;
            float f0 = texture(scalarField, gridToTexCoord(p0)).r - isoValue;
            float f1 = texture(scalarField, gridToTexCoord(p1)).r - isoValue;
            bool hasSolution = findHit(tHit, tNearHit, tFarHit, f0, f1, startPoint, rayDirection, A, B, C);
            if (hasSolution) {
                vec3 currentPoint = gridToWorldPos(startPoint + tHit * rayDirection);
                if (all(greaterThan(currentPoint, minBoundingBox)) && all(lessThan(currentPoint, maxBoundingBox))) {
#ifdef USE_RENDER_RESTRICTION
                    if (getShouldRender(currentPoint)) {
#endif
                    vec4 color = getIsoSurfaceHitColor(currentPoint);
                    if (blend(color, outputColor)) {
                        // Early ray termination.
                        return outputColor;
                    }
#ifdef USE_RENDER_RESTRICTION
                    }
#endif
                }
            }
#elif defined(SOLVER_MARMITT_MANUAL)
            float tHit;
            float f0 = ((A * tNearHit + B) * tNearHit + C) * tNearHit + D;
            float f1 = ((A * tFarHit + B) * tFarHit + C) * tFarHit + D;
            bool hasSolution = findHitManual(tHit, tNearHit, tFarHit, f0, f1, A, B, C, D);
            if (hasSolution) {
                vec3 currentPoint = gridToWorldPos(startPoint + tHit * rayDirection);
                if (all(greaterThan(currentPoint, minBoundingBox)) && all(lessThan(currentPoint, maxBoundingBox))) {
#ifdef USE_RENDER_RESTRICTION
                    if (getShouldRender(currentPoint)) {
#endif
                    vec4 color = getIsoSurfaceHitColor(currentPoint);
                    if (blend(color, outputColor)) {
                        // Early ray termination.
                        return outputColor;
                    }
#ifdef USE_RENDER_RESTRICTION
                    }
#endif
                }
            }
#endif
        }

#else // !defined(USE_INTERPOLATION_NEAREST_NEIGHBOR)

        float f000 = texelFetchClamp(voxelIndex, 0).r;
        if (f000 >= isoValue) {
#ifdef USE_RENDER_RESTRICTION
            if (getShouldRender(currentPoint)) {
#endif
                vec3 gradient = vec3(stepDirCurr);
                vec4 color = getIsoSurfaceHitColor(currentPoint, gradient);
                if (blend(color, outputColor)) {
                    // Early ray termination.
                    return outputColor;
                }
#ifdef USE_RENDER_RESTRICTION
            }
#endif
        }

#endif
        
        float newDist = length(vec3(voxelIndex) + vec3(0.5) - startPoint); // diff > sqrt(3)/2 (i.e., half diagonal)?
        if (newDist - maxDist > 0.866025404) {
            break;
        }

        if (tMaxX < tMaxY) {
            if (tMaxX < tMaxZ) {
                voxelIndex.x += stepX;
                tMaxX += tDeltaX;
#ifdef USE_INTERPOLATION_NEAREST_NEIGHBOR
                stepDirCurr = ivec3(stepX, 0, 0);
#endif
            } else {
                voxelIndex.z += stepZ;
                tMaxZ += tDeltaZ;
#ifdef USE_INTERPOLATION_NEAREST_NEIGHBOR
                stepDirCurr = ivec3(0, 0, stepZ);
#endif
            }
        } else {
            if (tMaxY < tMaxZ) {
                voxelIndex.y += stepY;
                tMaxY += tDeltaY;
#ifdef USE_INTERPOLATION_NEAREST_NEIGHBOR
                stepDirCurr = ivec3(0, stepY, 0);
#endif
            } else {
                voxelIndex.z += stepZ;
                tMaxZ += tDeltaZ;
#ifdef USE_INTERPOLATION_NEAREST_NEIGHBOR
                stepDirCurr = ivec3(0, 0, stepZ);
#endif
            }
        }
    }

    return outputColor;
}
#else
vec4 traverseVoxelGridRayMarching(
        vec3 currentPoint, vec3 rayDirection, vec3 entrancePoint, vec3 exitPoint, bool startsInVolume
#ifdef CLOSE_ISOSURFACES
        , vec3 boxNormal
#endif
) {
    vec4 outputColor = vec4(0.0);
    float lastScalarSign, currentScalarSign;
#ifdef CLOSE_ISOSURFACES
    bool isFirstPoint = !startsInVolume;
#else
    bool isFirstPoint = true;
#endif
    float volumeDepth = length(exitPoint - entrancePoint);
    while (length(currentPoint - entrancePoint) < volumeDepth) {
#ifdef SUPPORT_DEPTH_BUFFER
        if (length(currentPoint - cameraPosition) >= closestDepth) {
            break;
        }
#endif

        vec3 texCoords = (currentPoint - minBoundingBox) / (maxBoundingBox - minBoundingBox);

        float scalarValue = texture(scalarField, texCoords).r;
        if (!isnan(scalarValue)) {
            currentScalarSign = sign(scalarValue - isoValue);
#ifdef CLOSE_ISOSURFACES
            if (isFirstPoint) {
                lastScalarSign = sign(-isoValue);
            }
#else
            if (isFirstPoint) {
                isFirstPoint = false;
                lastScalarSign = currentScalarSign;
            }
#endif

            if (lastScalarSign != currentScalarSign) {
                refineIsoSurfaceHit(currentPoint, rayDirection, currentScalarSign);
#ifdef USE_RENDER_RESTRICTION
                if (getShouldRender(currentPoint)) {
#endif
                vec4 color = getIsoSurfaceHitColor(
                    currentPoint
#ifdef CLOSE_ISOSURFACES
                    , isFirstPoint, boxNormal
#endif
                );
                if (blend(color, outputColor)) {
                    break;
                }
#ifdef USE_RENDER_RESTRICTION
                }
#endif
            }

#ifdef CLOSE_ISOSURFACES
            isFirstPoint = false;
#endif
        }

        currentPoint += rayDirection * stepSize;
        lastScalarSign = currentScalarSign;
    }

    return outputColor;
}
#endif

void main() {
    ivec2 outputImageSize = imageSize(outputImage);
    ivec2 imageCoords = ivec2(gl_GlobalInvocationID.xy);
    if (imageCoords.x >= outputImageSize.x || imageCoords.y >= outputImageSize.y) {
        return;
    }

    gridSize = textureSize(scalarField, 0);

#ifdef SUPPORT_NORMAL_BUFFER
    normalSet = false;
    surfaceNormal = vec3(0.0);
#endif

    vec3 rayOrigin = (inverseViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec2 fragNdc = 2.0 * ((vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) / vec2(outputImageSize)) - 1.0;
    vec3 rayTarget = (inverseProjectionMatrix * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 normalizedTarget = normalize(rayTarget.xyz);
    vec3 rayDirection = (inverseViewMatrix * vec4(normalizedTarget, 0.0)).xyz;
    float zFactor = abs(normalizedTarget.z);

    float tNear, tFar;
#ifdef ANALYTIC_INTERSECTIONS
    vec3 voxelGridLower = vec3(-1.0 + 1e-4);
    vec3 voxelGridUpper = vec3(gridSize) + vec3(-1e-4);
    rayOrigin = worldToGridPos(rayOrigin);
    rayDirection = worldToGridDir(rayDirection);
#else
    vec3 voxelGridLower = minBoundingBox;
    vec3 voxelGridUpper = maxBoundingBox;
#endif
    if (rayBoxIntersectionRayCoords(rayOrigin, rayDirection, voxelGridLower, voxelGridUpper, tNear, tFar)) {
        vec3 entrancePoint = rayOrigin + rayDirection * tNear;
        vec3 exitPoint = rayOrigin + rayDirection * tFar;
        vec3 currentPoint = entrancePoint;

#ifdef CLOSE_ISOSURFACES
        vec3 boxNormal;
        vec3 C_Min = (voxelGridLower - rayOrigin) / rayDirection;
        vec3 C_Max = (voxelGridUpper - rayOrigin) / rayDirection;
        float minX = min(C_Min.x, C_Max.x);
        float minY = min(C_Min.y, C_Max.y);
        float minZ = min(C_Min.z, C_Max.z);
        if (minX > minY && minX > minZ) {
            boxNormal = vec3(-sign(rayDirection.x), 0.0, 0.0);
        } else if (minY > minZ) {
            boxNormal = vec3(0.0, -sign(rayDirection.y), 0.0);
        } else {
            boxNormal = vec3(0.0, 0.0, -sign(rayDirection.z));
        }
#endif

#ifdef SUPPORT_DEPTH_BUFFER
        closestDepth = convertDepthBufferValueToLinearDepth(imageLoad(depthBuffer, imageCoords).x);
        // Convert depth to distance.
        closestDepth = closestDepth / zFactor;
        closestDepthNew = closestDepth;
#endif

#ifdef ANALYTIC_INTERSECTIONS
        if (tNear < 0.0) {
            entrancePoint = rayOrigin;
        }
        vec4 outputColor = traverseVoxelGridAnalytic(rayDirection, entrancePoint, exitPoint);
#else
        bool startsInVolume = tNear < 0.0;
        if (tNear < 0.0) {
            currentPoint = rayOrigin;
        }
        vec4 outputColor = traverseVoxelGridRayMarching(
                currentPoint, rayDirection, entrancePoint, exitPoint, startsInVolume
#ifdef CLOSE_ISOSURFACES
                , boxNormal
#endif
        );
#endif

        vec4 backgroundColor = imageLoad(outputImage, imageCoords);
        blend(backgroundColor, outputColor);

        outputColor = vec4(outputColor.rgb / outputColor.a, outputColor.a);
        imageStore(outputImage, imageCoords, outputColor);
#ifdef SUPPORT_DEPTH_BUFFER
        // Convert depth to distance.
        closestDepthNew = closestDepthNew * zFactor;
        imageStore(depthBuffer, imageCoords, vec4(convertLinearDepthToDepthBufferValue(closestDepthNew)));
#ifdef SUPPORT_NORMAL_BUFFER
        if (closestDepthNew < closestDepth) {
            imageStore(normalBuffer, imageCoords, vec4(surfaceNormal, 1.0));
        }
#endif
#endif
    }
}
