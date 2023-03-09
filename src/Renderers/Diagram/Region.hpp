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

#ifndef CORRERENDER_REGION_HPP
#define CORRERENDER_REGION_HPP

struct GridRegion {
    GridRegion() = default;
    GridRegion(int xoff, int yoff, int zoff, int xsr, int ysr, int zsr)
            : xoff(xoff), yoff(yoff), zoff(zoff), xsr(xsr), ysr(ysr), zsr(zsr) {
        xmin = xoff;
        ymin = yoff;
        zmin = zoff;
        xmax = xoff + xsr - 1;
        ymax = yoff + ysr - 1;
        zmax = zoff + zsr - 1;
    }
    [[nodiscard]] inline int getNumCells() const {
        return xsr * ysr * zsr;
    }
    inline bool operator==(const GridRegion& rhs) const {
        return
                xoff == rhs.xoff && yoff == rhs.yoff && zoff == rhs.zoff
                && xsr == rhs.xsr && ysr == rhs.ysr && zsr == rhs.zsr;
    }
    int xoff = 0, yoff = 0, zoff = 0; //< Offset.
    int xsr = 0, ysr = 0, zsr = 0; //< Region size.
    int xmin = 0, ymin = 0, zmin = 0, xmax = 0, ymax = 0, zmax = 0;
};

#endif //CORRERENDER_REGION_HPP
