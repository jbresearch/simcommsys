/*!
 * \file
 * 
 * Copyright (c) 2010 Johann A. Briffa
 * 
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * \section svn Version Control
 * - $Id$
 */

#include "codec_softout.h"
#include "hard_decision.h"

namespace libcomm {

template <class dbl>
void codec_softout<libbase::vector, dbl>::init_decoder(const array1vd_t& ptable)
   {
   array1vd_t temp;
   temp = ptable;
   this->setreceiver(temp);
   this->resetpriors();
   }

template <class dbl>
void codec_softout<libbase::vector, dbl>::init_decoder(
      const array1vd_t& ptable, const array1vd_t& app)
   {
   setreceiver(ptable);
   setpriors(app);
   }

template <class dbl>
void codec_softout<libbase::vector, dbl>::decode(array1i_t& decoded)
   {
   array1vd_t ri;
   softdecode(ri);
   hard_decision<libbase::vector, dbl> functor;
   functor(ri, decoded);
   }

} // end namespace

#include "mpreal.h"
#include "mpgnu.h"
#include "logreal.h"
#include "logrealfast.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::vector;

using libbase::mpreal;
using libbase::mpgnu;
using libbase::logreal;
using libbase::logrealfast;

#define REAL_TYPE_SEQ \
   (float)(double) \
   (mpreal)(mpgnu) \
   (logreal)(logrealfast)

#define INSTANTIATE(r, x, type) \
      template class codec_softout<vector, type>;

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, REAL_TYPE_SEQ)

} // end namespace
