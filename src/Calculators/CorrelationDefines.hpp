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

#ifndef CORRERENDER_CORRELATIONDEFINES_HPP
#define CORRERENDER_CORRELATIONDEFINES_HPP

const char* const CORRELATION_MODE_NAMES[2] = {
        "Ensemble", "Time"
};

enum class CorrelationDataMode {
    IMAGE_3D_ARRAY, BUFFER_ARRAY
};
const char* const DATA_MODE_NAMES[] = { "Image 3D Array", "Buffer Array" };

enum class CorrelationMeasureType {
    PEARSON, SPEARMAN, KENDALL, MUTUAL_INFORMATION_BINNED, MUTUAL_INFORMATION_KRASKOV,
    // For more information on MICC see "An informational measure of correlation", Linfoot 1957.
    BINNED_MI_CORRELATION_COEFFICIENT, KMI_CORRELATION_COEFFICIENT
};
const char* const CORRELATION_MEASURE_TYPE_NAMES[] = {
        "Pearson", "Spearman", "Kendall", "Mutual Information (Binned)", "Mutual Information (Kraskov)",
        "Binned MI Correlation Coefficient", "KMI Correlation Coefficient"
};
const char* const CORRELATION_MEASURE_TYPE_SHORT_NAMES[] = {
        "Pearson", "Spearman", "Kendall", "MI (Binned)", "MI (Kraskov)",
        "Binned MI CC", "KMI CC"
};
const char* const CORRELATION_MEASURE_TYPE_IDS[] = {
        "pearson", "spearman", "kendall", "mi_binned", "mi_kraskov",
        "binned_mi_correlation_coefficient", "kmi_correlation_coefficient"
};
inline bool isMeasureBinnedMI(CorrelationMeasureType m) {
    return m == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
            || m == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT;
}
inline bool isMeasureKraskovMI(CorrelationMeasureType m) {
    return m == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            || m == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT;
}
inline bool isMeasureMI(CorrelationMeasureType m) {
    return m == CorrelationMeasureType::MUTUAL_INFORMATION_BINNED
            || m == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT
            || m == CorrelationMeasureType::MUTUAL_INFORMATION_KRASKOV
            || m == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT;
}
inline bool isMeasureCorrelationCoefficientMI(CorrelationMeasureType m) {
    return m == CorrelationMeasureType::BINNED_MI_CORRELATION_COEFFICIENT
           || m == CorrelationMeasureType::KMI_CORRELATION_COEFFICIENT;
}

const char* const FIELD_ACCURACY_NAMES[] = {
        "Float (32-bit)", "Double (64-bit)"
};

enum class NetworkType {
    /*
     * Network based on the paper "Mutual Information Neural Estimation", Belghazi et al. 2018.
     * For more details see: https://arxiv.org/abs/1801.04062
     * This network takes as an input a scalar value and positions. It consists of one encoder and one decoder network.
     */
    MINE,
    /*
     * Scene representation network consisting of an encoder and decoder stage.
     */
    SRN_MINE,
    /*
     * Scene representation network. This network takes as an input positions and outputs the correlation value.
     */
    SRN
};
const char* const NETWORK_TYPE_SHORT_NAMES[] = {
        "MINE", "SRN_MINE", "SRN"
};

enum class DiagramMode {
    CHORD, MATRIX
};
const char* const DIAGRAM_MODE_NAMES[] = {
        "Chord Diagram", "Correlation Matrix"
};
const char* const DIAGRAM_MODE_TYPE_IDS[] = {
        "chords", "matrix"
};

#endif //CORRERENDER_CORRELATIONDEFINES_HPP
