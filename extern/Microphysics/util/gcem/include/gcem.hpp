/*################################################################################
  ##
  ##   Copyright (C) 2016-2020 Keith O'Hara
  ##
  ##   This file is part of the GCE-Math C++ library.
  ##
  ##   Licensed under the Apache License, Version 2.0 (the "License");
  ##   you may not use this file except in compliance with the License.
  ##   You may obtain a copy of the License at
  ##
  ##       http://www.apache.org/licenses/LICENSE-2.0
  ##
  ##   Unless required by applicable law or agreed to in writing, software
  ##   distributed under the License is distributed on an "AS IS" BASIS,
  ##   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ##   See the License for the specific language governing permissions and
  ##   limitations under the License.
  ##
  ################################################################################*/

#ifndef _gcem_HPP
#define _gcem_HPP

#include "gcem_incl/gcem_options.hpp"

// Modified for Quokka: pow, sqrt, exp, and their transitive helpers only.
// See ../../../README.quokka.md for provenance.
namespace gcem
{

    #include "gcem_incl/is_inf.hpp"
    #include "gcem_incl/is_nan.hpp"
    #include "gcem_incl/is_finite.hpp"

    #include "gcem_incl/abs.hpp"
    #include "gcem_incl/floor.hpp"
    #include "gcem_incl/is_odd.hpp"
    #include "gcem_incl/sqrt.hpp"
    #include "gcem_incl/sgn.hpp"

    #include "gcem_incl/find_exponent.hpp"
    #include "gcem_incl/find_fraction.hpp"
    #include "gcem_incl/find_whole.hpp"
    #include "gcem_incl/mantissa.hpp"

    #include "gcem_incl/pow_integral.hpp"
    #include "gcem_incl/exp.hpp"
    #include "gcem_incl/log.hpp"
    #include "gcem_incl/pow.hpp"
}

#endif
