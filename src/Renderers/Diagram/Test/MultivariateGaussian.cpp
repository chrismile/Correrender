/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser
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

#define _USE_MATH_DEFINES
#include <cmath>

#include <Eigen/Core>
#include <Eigen/LU>

#include "MultivariateGaussian.hpp"

struct MultivariateGaussianCache {
    double normFactor{};
    Eigen::VectorXd mean;
    Eigen::MatrixXd invCov;
};

MultivariateGaussian::MultivariateGaussian(int dfx, int dfy, int dfz) : dfx(dfx), dfy(dfy), dfz(dfz) {
    cache = new MultivariateGaussianCache;
}

MultivariateGaussian::~MultivariateGaussian() {
    delete cache;
}

static Eigen::VectorXd normalizedVector(const Eigen::VectorXd& v) {
    double vLen = std::sqrt(v.cwiseProduct(v).sum());
    return v / vLen;
}

static Eigen::VectorXd proj(const Eigen::VectorXd& u, const Eigen::VectorXd& v) {
    return (v.dot(u) / u.dot(u)) * u;
}

void MultivariateGaussian::initRandom(std::mt19937& generator) {
    constexpr double domainOffset = 0.0; // TODO: Max may lie outside domain if this is > 0.
    std::uniform_real_distribution<double> dm(-domainOffset, 1.0 + domainOffset);
    cache->mean = Eigen::VectorXd(6);
    cache->mean << dm(generator), dm(generator), dm(generator), dm(generator), dm(generator), dm(generator);

    // Use Gram-Schmidt to sample orthogonal random vectors.
    Eigen::MatrixXd Q(6, 6);
    std::uniform_real_distribution<double> dv(-1.0, 1.0);
    for (int i = 0; i < 6; i++) {
        Eigen::VectorXd v(6);
        v << dv(generator), dv(generator), dv(generator), dv(generator), dv(generator), dv(generator);
        Q.col(i) = v;
        for (int j = 0; j < i; j++) {
            Q.col(i) -= proj(Q.col(j), v);
        }
        Q.col(i) = normalizedVector(Q.col(i));
    }

    // Compute the covariance matrix with randomly generated eigenvectors.
    Eigen::DiagonalMatrix<double, 6> E;
    std::uniform_real_distribution<double> dev(0.1, 0.5);
    E.diagonal() << dev(generator), dev(generator), dev(generator), dev(generator), dev(generator), dev(generator);
    Eigen::MatrixXd cov = Q * E * Q.transpose();

    // Cache the inverse covariance matrix and the normalization factor for the multivariate Gaussian PDF.
    cache->invCov = cov.inverse();
    cache->normFactor = 1.0 / std::sqrt(std::pow(2.0 * M_PI, 6.0) * cov.determinant());
}

float MultivariateGaussian::eval(int xi, int yi, int zi, int xj, int yj, int zj) {
    Eigen::VectorXd x(6);
    x << double(xi) / double(dfx - 1), double(yi) / double(dfy - 1), double(zi) / double(dfz - 1),
            double(xj) / double(dfx - 1), double(yj) / double(dfy - 1), double(zj) / double(dfz - 1);
    auto diff = x - cache->mean;
    auto value = float(std::exp(-0.5 * diff.dot(cache->invCov * diff)) * cache->normFactor);
    return value;
}

float MultivariateGaussian::eval(double* data) {
    Eigen::VectorXd x(6);
    x << data[0], data[1], data[2], data[3], data[4], data[5];
    auto diff = x - cache->mean;
    auto value = float(std::exp(-0.5 * diff.dot(cache->invCov * diff)) * cache->normFactor);
    return value;
}

std::pair<float, float> MultivariateGaussian::getGlobalMinMax() {
    Eigen::VectorXd cornerPoint(6);

    // Compute maximum.
    //float maxVal = eval(cache->mean.data());
    double posGridLower[6];
    double posGridUpper[6];
    for (int i = 0; i < 6; i++) {
        int df = 0;
        switch(i % 3) {
            case 0:
            case 3:
                df = dfx;
                break;
            case 1:
            case 4:
                df = dfy;
                break;
            case 2:
            case 5:
                df = dfz;
                break;
        }
        double posGrid = cache->mean(i) * double(df - 1);
        posGridLower[i] = std::floor(posGrid) / double(df - 1);
        posGridUpper[i] = std::ceil(posGrid) / double(df - 1);
    }
    float maxVal = std::numeric_limits<float>::lowest();
    for (int cornerIdx = 0; cornerIdx < (1 << 6); cornerIdx++) {
        for (int i = 0; i < 6; i++) {
            cornerPoint(i) = ((cornerIdx >> i) & 0x1) == 0 ? posGridLower[i] : posGridUpper[i];
        }
        maxVal = std::max(maxVal, eval(cornerPoint.data()));
    }

    // Compute minimum.
    float minVal = std::numeric_limits<float>::max();
    for (int cornerIdx = 0; cornerIdx < (1 << 6); cornerIdx++) {
        for (int i = 0; i < 6; i++) {
            cornerPoint(i) = ((cornerIdx >> i) & 0x1) == 0 ? 0.0 : 1.0;
        }
        minVal = std::min(minVal, eval(cornerPoint.data()));
    }

    return std::make_pair(minVal, maxVal);
}
