#!/usr/bin/python3

# BSD 2-Clause License
# 
# Copyright (c) 2023, Christoph Neuhauser
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Script for generating a synthetic ensemble data set. How to run:
# - Install all dependencies, e.g., using conda:
# conda install -c numba numba
# conda install -c anaconda numpy
# conda install -c conda-forge netcdf4
# - When launching this program with Python, it will create the file Data/VolumeDataSets/linear_4x4.nc.
# - In Correrender, this file can be opened via "File > Open Dataset".
# - Then, a correlation diagram renderer can be created via "Window > New Renderer... > Diagram Renderer".

from numba import jit
import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
from netCDF4 import Dataset

xs = 128
ys = 128
zs = 32
members = 1000
width = zs
half_width = zs // 2
g = zs // 2

#@jit(nopython=True)
#def peak_fun(x):
#    if x >= 1:
#        return 0
#    return abs(1.0 - x)
@jit(nopython=True)
def peak_fun(x):
    if x >= 1:
        return 0
    return 1.0 - pow(max(0.0, abs(x) * 2.0 - 1.0), 2.0)


#X = np.linspace(0.0, 1.0, 101)
#Y = [peak_fun(x) for x in X]
#plt.plot(X, Y)
#plt.show()


@jit(nopython=True)
def peak_fun_vec(cx, cy, size, scale):
    field = np.empty((zs, ys, xs))
    cz = zs // 2
    for z in range(zs):
        for y in range(ys):
            for x in range(xs):
                dx = abs(x - cx)
                dy = abs(y - cy)
                dz = abs(z - cz)
                #dist = np.sqrt(dx * dx + dy * dy + dz * dz)
                dist = max(dx, max(dy, dz))
                dist /= (size * 0.5)
                field[z, y, x] = scale * peak_fun(dist)
    return field


peaks = [
    (g,     g,     2.0 * g, 1.0),
    (7 * g, 7 * g, 2.0 * g, 1.0),

    (2.5 * g, 0.5 * g, g, 1.0),
    (2.5 * g, 1.5 * g, g, 1.0),

    (5.5 * g, 6.5 * g, g, 1.0),
    (5.5 * g, 7.5 * g, g, 1.0),

    (0.5 * g, 2.5 * g, g, 1.0),
    (1.5 * g, 2.5 * g, g, 1.0),

    (6.5 * g, 5.5 * g, g, 1.0),
    (7.5 * g, 5.5 * g, g, 1.0),
]
lambda_field = np.empty((zs, ys, xs))
for peak in peaks:
    lambda_field += peak_fun_vec(peak[0], peak[1], peak[2], peak[3])


#lambda_field_2d = lambda_field[half_width, :, :]
#plt.imshow(lambda_field_2d, cmap=mpl.colormaps['coolwarm'])
#plt.colorbar()
#plt.show()

is_linear = True
if is_linear:
    s1p = 2.0 * np.linspace(0.0, 1.0, members) - 1.0
    s1n = -s1p
else:
    s1p = np.sin(np.linspace(0.0, 2.0 * np.pi, members))
    s1n = np.cos(np.linspace(0.0, 2.0 * np.pi, members))
data_out = np.zeros((members, zs, ys, xs))
for z in range(zs):
    for y in range(ys):
        for x in range(xs):
            lmbd = lambda_field[z, y, x]
            if lmbd >= 0.0:
                f = 1.0
            else:
                f = -1.0
            lmbd = abs(lmbd)
            s0 = np.random.normal(loc=0.0, scale=1.0, size=members)
            if f >= 0.0:
                s1 = s1p
            else:
                s1 = s1n
            samples = lmbd * s1 + (1.0 - lmbd) * s0
            data_out[:, z, y, x] = samples[:]


#def plot_corr(p0, p1):
#    fig, ax = plt.subplots()
#    data = ax.scatter(data_out[:, p0[0], p0[1], p0[2]], data_out[:, p1[0], p1[1], p1[2]])
#    ax.set_xlim([-3.1, 3.1])
#    ax.set_ylim([-3.1, 3.1])
#    fig.colorbar(data)
#    plt.show()


#plot_corr((zs // 2, half_width, half_width), (zs // 2, half_width, half_width * 3))


outfile = f"../Data/VolumeDataSets/{'linear' if is_linear else 'circle'}_4x4.nc"
ncfile = Dataset(outfile, mode='w', format='NETCDF4_CLASSIC')
mdim = ncfile.createDimension('member', members)
zdim = ncfile.createDimension('lev', zs)
ydim = ncfile.createDimension('lat', ys)
xdim = ncfile.createDimension('lon', xs)
outfield = ncfile.createVariable('data', np.float32, ('member', 'lev', 'lat', 'lon'))
outfield[:, :, :, :] = data_out

ncfile.close()
print('Done.')
