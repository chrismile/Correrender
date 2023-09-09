/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2023, Christoph Neuhauser, Josef Stumpfegger
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

// this header contains structures needed for the bayesion optimization used to 
// find maximum correlations in the 6 dimensional correlation space

#ifndef CORRERENDER_BAYOPT_HPP
#define CORRERENDER_BAYOPT_HPP

constexpr bool EVAL_TIMINGS = false;

#include <variant>

#define USE_NLOPT
#include <limbo/bayes_opt/boptimizer.hpp>
#include "Calculators/MutualInformation.hpp"
#include "Loaders/DataSet.hpp"

namespace std{
    template <typename T> int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }
}

namespace BayOpt{
using namespace limbo;

template<typename T>
class i_range {
public:
    constexpr i_range(T end): _begin(0), _end(end < 0 ? 0: end), _step(1){}; // single element constructor 
    constexpr i_range(T begin, T end, T step = T(1)):
     _begin(begin), _end((end - begin + step - std::sign(step)) / step * step + begin), _step(step){
        assert(step != 0 && "step of 0 is invalid");
        if((begin > end && step > 0) || (begin < end && step < 0))
            _begin = _end;
    };

    class iterator {
        friend class i_range;
    public:
        T operator *() const { return i_; }
        const iterator &operator ++() { i_ += _step; return *this; }
        iterator operator ++(int) { iterator copy(*this); i_ += _step; return copy; }    
        bool operator ==(const iterator &other) const { return i_ == other.i_; }
        bool operator !=(const iterator &other) const { return i_ != other.i_; } 
    protected:
        iterator(T start, T step = 1) : i_ (start), _step(step) { }    
    private:
        T i_, _step;
    };  

    iterator begin() const { return iterator(_begin, _step); }
    iterator end() const { return iterator(_end); }
private:
    T _begin, _end;
    T _step;
};

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    };

// depending on which internal optimizer we use, we need to import different parameters
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif

    // enable / disable the writing of the result files
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, false);
    };

    // no noise
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };

    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
    };

    // we use 10 random samples to initialize the algorithm
    struct init_randomsampling {
        BO_DYN_PARAM(int, samples);
    };

    // we stop after 40 iterations
    struct stop_maxiterations {
        BO_DYN_PARAM(int, iterations);
    };

    // we use the default parameters for acqui_ucb
    struct acqui_ucb : public defaults::acqui_ucb {
    };
};

std::random_device rand_dev;
std::mt19937 gen(rand_dev());
std::uniform_real_distribution<> uniform_dist;

inline int64_t pr(double x){
    double base = std::floor(x);
    return int64_t(base) + int64_t((x - base) > uniform_dist(gen));
}

template<typename Functor>
struct Eval{
    const std::vector<const float *>& d1;
    const std::vector<const float *>& d2;
    const std::array<int, 6>&       bounds_min;
    const std::array<int, 6>&       bounds_max;
    const int                       num_members, xs, ys;
    const Functor                   f;
    mutable std::array<int64_t, 6>  dims;
    mutable std::vector<float>      a,b;
    mutable int                     execution_count{};
    mutable float                   execution_sum{};
    // number of input dimension (x.size())
    BO_PARAM(size_t, dim_in, 6);
    // number of dimensions of the result (res.size())
    BO_PARAM(size_t, dim_out, 1);

    // the function to be optimized
    // limbo is set to bounded mode, meaning that the values in x are always in 0-1
    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        a.resize(num_members); b.resize(num_members);
        auto start = std::chrono::system_clock::now();
        // converting the continuous indices do discrete ones with probabilistic reparametrization and filling the vectors a and b
        for(int i: i_range(6)){
            double v = x[i] * (bounds_max[i] - bounds_min[i]) + bounds_min[i];
            dims[i] = pr(v);
        }

        bool isNan = false;
        int64_t lin_pos_a = IDXS(dims[0], dims[1], dims[2]);
        int64_t lin_pos_b = IDXS(dims[3], dims[4], dims[5]);
        for(int i: i_range(num_members)) {
            a[i] = d1.at(i)[lin_pos_a];
            b[i] = d2.at(i)[lin_pos_b];
            if (std::isnan(a[i]) || std::isnan(b[i])) {
                isNan = true;
                break;
            }
        }

        auto middle = std::chrono::system_clock::now();
        
        float y = isNan ? 0.0f : std::abs(f(a.data(), b.data(), num_members));

        //float y = computeMutualInformationKraskov(a.data(), b.data(), k, num_members, cache);
        auto end = std::chrono::system_clock::now();
        ++execution_count;
        if constexpr(EVAL_TIMINGS){
            execution_sum += std::chrono::duration<double>(end - start).count();
            std::cout << "Mutual information calc time: " << std::chrono::duration<double>(end - start).count() << " s(" << std::chrono::duration<double>(middle - start).count() << "/" << std::chrono::duration<double>(end - middle).count() << "), total time spent: " << execution_sum << " s." << std::endl;
        }
        if(std::isinf(y)) y = 0;
        // we return a 1-dimensional vector
        return tools::make_vector(y);
    }
};

// evaluation functors for the Eval struct
struct PearsonFunctor{
    float operator()(float* a, float* b, int num_members) const {
        return computePearson2<float>(a, b, num_members);
    }
};
struct SpearmanFunctor{
    mutable std::vector<float> referenceRanks;
    mutable std::vector<float> gridPointRanks;
    mutable std::vector<std::pair<float, int>> ordinalRankArrayRef;
    mutable std::vector<std::pair<float, int>> ordinalRankArraySpearman;

    float operator()(float* a, float* b, int num_members) const {
        referenceRanks.resize(num_members);
        gridPointRanks.resize(num_members);
        ordinalRankArrayRef.resize(num_members);
        ordinalRankArraySpearman.resize(num_members);
        computeRanks(a, referenceRanks.data(), ordinalRankArrayRef, num_members);
        computeRanks(b, gridPointRanks.data(), ordinalRankArraySpearman, num_members);
        return computePearson2<float>(referenceRanks.data(), gridPointRanks.data(), num_members);
    }
};
struct KendallFunctor{
    mutable std::vector<std::pair<float, float>> jointArray;
    mutable std::vector<float> ordinalRankArray;
    mutable std::vector<float> y;
    mutable std::vector<float> sortArray;
    mutable std::vector<std::pair<int, int>> stack;
    float operator()(float* a, float* b, int num_members) const {
        jointArray.reserve(num_members);
        ordinalRankArray.reserve(num_members);
        y.reserve(num_members);
        sortArray.reserve(num_members);
        return computeKendall<int32_t>(a, b, num_members, jointArray, ordinalRankArray, y, sortArray, stack);
    }
};
struct MutualBinnedFunctor{
    const float& minFieldVal1, maxFieldVal1;
    const float& minFieldVal2, maxFieldVal2;
    const int&   numBins;
    mutable std::vector<double> histogram0, histogram1, histogram2d;
    float operator()(float* a, float* b, int num_members) const {
        histogram0.reserve(numBins);
        histogram1.reserve(numBins);
        histogram2d.reserve(numBins * numBins);
        for (int c = 0; c < num_members; c++) {
            a[c] = (a[c] - minFieldVal1) / (maxFieldVal1 - minFieldVal1);
            b[c] = (b[c] - minFieldVal2) / (maxFieldVal2 - minFieldVal2);
        }
        return computeMutualInformationBinned<double>(a, b, numBins, num_members, histogram0.data(), histogram1.data(), histogram2d.data());
    }
};
struct MutualFunctor{
    const int& k;
    mutable KraskovEstimatorCache<double> kraskovEstimatorCache;
    float operator()(float* a, float* b, int num_members) const {
        return computeMutualInformationKraskov<double>(a, b, k, num_members, kraskovEstimatorCache);
    }
};

struct MutualBinnedCCFunctor{
    const float& minFieldVal1, maxFieldVal1;
    const float& minFieldVal2, maxFieldVal2;
    const int&   numBins;
    mutable std::vector<double> histogram0, histogram1, histogram2d;
    float operator()(float* a, float* b, int num_members) const {
        histogram0.reserve(numBins);
        histogram1.reserve(numBins);
        histogram2d.reserve(numBins * numBins);
        for (int c = 0; c < num_members; c++) {
            a[c] = (a[c] - minFieldVal1) / (maxFieldVal1 - minFieldVal1);
            b[c] = (b[c] - minFieldVal2) / (maxFieldVal2 - minFieldVal2);
        }
        float mi = computeMutualInformationBinned<double>(a, b, numBins, num_members, histogram0.data(), histogram1.data(), histogram2d.data());
        return std::sqrt(1.0f - std::exp(-2.0f * mi));
    }
};
struct MutualCCFunctor{
    const int& k;
    mutable KraskovEstimatorCache<double> kraskovEstimatorCache;
    float operator()(float* a, float* b, int num_members) const {
        float mi = computeMutualInformationKraskov<double>(a, b, k, num_members, kraskovEstimatorCache);
        return std::sqrt(1.0f - std::exp(-2.0f * mi));
    }
};

template<typename Params, nlopt::algorithm Algo> using o = limbo::opt::NLOptNoGrad<Params, Algo>;
template<typename Params>
using AlgorithmNoGradVariants = std::variant<o<Params, nlopt::LN_COBYLA> ,  o<Params, nlopt::LN_BOBYQA> ,
    o<Params, nlopt::LN_NEWUOA              >, o<Params, nlopt::LN_NEWUOA_BOUND     >,
    o<Params, nlopt::LN_PRAXIS              >, o<Params, nlopt::LN_NELDERMEAD       >,
    o<Params, nlopt::LN_SBPLX               >, o<Params, nlopt::GN_DIRECT           >,
    o<Params, nlopt::GN_DIRECT_L            >, o<Params, nlopt::GN_DIRECT_L_RAND    >,
    o<Params, nlopt::GN_DIRECT_NOSCAL       >, o<Params, nlopt::GN_DIRECT_L_NOSCAL  >,
    o<Params, nlopt::GN_DIRECT_L_RAND_NOSCAL>, o<Params, nlopt::GN_ORIG_DIRECT      >,
    o<Params, nlopt::GN_ORIG_DIRECT_L       >, o<Params, nlopt::GN_CRS2_LM          >,
    o<Params, nlopt::LN_AUGLAG              >, o<Params, nlopt::LN_AUGLAG_EQ        >,
    o<Params, nlopt::GN_ISRES               >, o<Params, nlopt::GN_ESCH             >>;
template<typename Params>
inline AlgorithmNoGradVariants<Params> getOptimizerAsVariant(nlopt::algorithm a){
    switch(a){
    case nlopt::LN_COBYLA: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_COBYLA>{}};
    case nlopt::LN_BOBYQA: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_BOBYQA>{}};
    case nlopt::LN_NEWUOA: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_NEWUOA>{}};
    case nlopt::LN_SBPLX: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_SBPLX>{}};
    case nlopt::LN_NEWUOA_BOUND: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_NEWUOA_BOUND>{}};
    case nlopt::LN_PRAXIS: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_PRAXIS>{}};
    case nlopt::LN_NELDERMEAD: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_NELDERMEAD>{}};
    case nlopt::GN_DIRECT: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT>{}};
    case nlopt::GN_DIRECT_L: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L>{}};
    case nlopt::GN_DIRECT_L_RAND: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND>{}};
    case nlopt::GN_DIRECT_NOSCAL: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_NOSCAL>{}};
    case nlopt::GN_DIRECT_L_NOSCAL: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_NOSCAL>{}};
    case nlopt::GN_DIRECT_L_RAND_NOSCAL: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND_NOSCAL>{}};
    case nlopt::GN_ORIG_DIRECT: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_ORIG_DIRECT>{}};
    case nlopt::GN_ORIG_DIRECT_L: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_ORIG_DIRECT_L>{}};
    case nlopt::GN_CRS2_LM: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_CRS2_LM>{}};
    case nlopt::LN_AUGLAG: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_AUGLAG>{}};
    case nlopt::LN_AUGLAG_EQ: return {limbo::opt::NLOptNoGrad<Params, nlopt::LN_AUGLAG_EQ>{}};
    case nlopt::GN_ISRES: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_ISRES>{}};
    case nlopt::GN_ESCH: return {limbo::opt::NLOptNoGrad<Params, nlopt::GN_ESCH>{}};
    default: exit(-1); return {};
    };
    return {};
}
};

#endif
