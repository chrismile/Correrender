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

#ifndef CORRERENDER_BSPLINE_HPP
#define CORRERENDER_BSPLINE_HPP

#include <vector>
#include <glm/vec2.hpp>

/**
 * Evaluates a B-spline curve with the passed control points at parameter x.
 * A knot vector is used such that the curve passes through the start and end point, e.g.,
 * [0, 0, 0, 0.5, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], etc.
 * @param x The parameter value in range [0, 1] along the B-spline curve to evaluate.
 * @param k The order of the B-spline curve. The degree of the curve is k - 1.
 * @param controlPoints The control points.
 * @return The evaluated point location.
 *
 * For more details see:
 * - https://www.gnu.org/software/gsl/doc/html/bspline.html
 * - https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node16.html
 *   https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html
 *   https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node18.html
 * - https://www.cs.cmu.edu/afs/cs/academic/class/15456-f15/Handouts/CAGD-chapter8.pdf
 */
glm::vec2 evaluateBSpline(float x, int k, const std::vector<glm::vec2>& controlPoints);

#endif //CORRERENDER_BSPLINE_HPP
