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

#ifndef CORRERENDER_NLOPTDEFINES_HPP
#define CORRERENDER_NLOPTDEFINES_HPP

#include <nlopt.hpp>

const nlopt::algorithm NLoptAlgorithmsNoGrad[]{
    nlopt::LN_COBYLA ,  nlopt::LN_BOBYQA ,
    nlopt::LN_NEWUOA ,  nlopt::LN_NEWUOA_BOUND ,
    nlopt::LN_PRAXIS ,  nlopt::LN_NELDERMEAD ,
    nlopt::LN_SBPLX , nlopt::GN_DIRECT ,
    nlopt::GN_DIRECT_L , nlopt::GN_DIRECT_L_RAND ,
    nlopt::GN_DIRECT_NOSCAL , nlopt::GN_DIRECT_L_NOSCAL ,
    nlopt::GN_DIRECT_L_RAND_NOSCAL , nlopt::GN_ORIG_DIRECT ,
    nlopt::GN_ORIG_DIRECT_L , nlopt::GN_CRS2_LM ,
    nlopt::LN_AUGLAG , nlopt::LN_AUGLAG_EQ ,
    nlopt::GN_ISRES , nlopt::GN_ESCH
};

const char* const NLOPT_ALGORITHM_NAMES_NOGRAD[] = {
    "LN_COBYLA" ,  "LN_BOBYQA" ,
    "LN_NEWUOA" ,  "LN_NEWUOA_BOUND" , // Seems to hang the program. One thread worker doesn't terminate.
    "LN_PRAXIS" ,  "LN_NELDERMEAD" ,
    "LN_SBPLX" , "GN_DIRECT" ,
    "GN_DIRECT_L" , "GN_DIRECT_L_RAND" ,
    "GN_DIRECT_NOSCAL" , "GN_DIRECT_L_NOSCAL" ,
    "GN_DIRECT_L_RAND_NOSCAL" , "GN_ORIG_DIRECT" ,
    "GN_ORIG_DIRECT_L" , "GN_CRS2_LM" ,
    "LN_AUGLAG" , "LN_AUGLAG_EQ" ,
    "GN_ISRES" , "GN_ESCH"
};


#endif
