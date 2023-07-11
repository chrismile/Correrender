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

#ifndef CORRERENDER_CORRELATIONMATRIX_HPP
#define CORRERENDER_CORRELATIONMATRIX_HPP

#include <algorithm>
#include <cassert>

class CorrelationMatrix {
public:
    CorrelationMatrix(int rows, int columns) : rows(rows), columns(columns) {}
    virtual ~CorrelationMatrix() { if (data) { delete[] data; data = nullptr; } }
    [[nodiscard]] inline int getNumRows() const { return rows; }
    [[nodiscard]] inline int getNumColumns() const { return columns; }
    [[nodiscard]] virtual bool getIsSymmetric() const=0;
    virtual void set(int i, int j, float value)=0;
    [[nodiscard]] virtual float get(int i, int j) const=0;

protected:
    int rows, columns;
    float* data = nullptr;
};

class FullCorrelationMatrix : public CorrelationMatrix {
public:
    FullCorrelationMatrix(int rows, int columns) : CorrelationMatrix(rows, columns) {
        int numEntries = rows * columns;
        data = new float[numEntries];
        std::fill(data, data + numEntries, std::numeric_limits<float>::quiet_NaN());
    }
    ~FullCorrelationMatrix() override = default;
    [[nodiscard]] bool getIsSymmetric() const override { return false; }
    void set(int i, int j, float value) override {
        //assert(i < rows && j < columns);
        if (i >= rows || j >= columns) {
            throw std::runtime_error("Error: Invalid row or column index in symmetric correlation matrix.");
        }
        data[i + j * rows] = value;
    }
    [[nodiscard]] float get(int i, int j) const override {
        //assert(i < rows && j < columns);
        if (i >= rows || j >= columns) {
            throw std::runtime_error("Error: Invalid row or column index in symmetric correlation matrix.");
        }
        return data[i + j * rows];
    }
};

class SymmetricCorrelationMatrix : public CorrelationMatrix {
public:
    explicit SymmetricCorrelationMatrix(int N) : CorrelationMatrix(N, N) {
        int numEntries = (N * N - N) / 2;
        data = new float[numEntries];
        std::fill(data, data + numEntries, std::numeric_limits<float>::quiet_NaN());
    }
    ~SymmetricCorrelationMatrix() override = default;
    [[nodiscard]] bool getIsSymmetric() const override { return true; }
    void set(int i, int j, float value) override {
        //assert(i > j && i < rows && j < columns);
        if (i >= rows || j >= columns || i == j) {
            throw std::runtime_error("Error: Invalid row or column index in symmetric correlation matrix.");
        }
        if (i < j) {
            int tmp = i;
            i = j;
            j = tmp;
        }
        int linearIdx = i * (i - 1) / 2 + j;
        data[linearIdx] = value;
    }
    [[nodiscard]] float get(int i, int j) const override {
        //assert(i > j && i < rows && j < columns);
        if (i >= rows || j >= columns || i == j) {
            throw std::runtime_error("Error: Invalid row or column index in symmetric correlation matrix.");
        }
        if (i < j) {
            int tmp = i;
            i = j;
            j = tmp;
        }
        int linearIdx = i * (i - 1) / 2 + j;
        return data[linearIdx];
    }
};

#endif //CORRERENDER_CORRELATIONMATRIX_HPP
