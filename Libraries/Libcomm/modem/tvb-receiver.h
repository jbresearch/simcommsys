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
 */

#ifndef __tvb_receiver_h
#define __tvb_receiver_h

#include "config.h"
#include "channel_insdel.h"
#include <memory>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show settings when initializing the tvb computer
// NOTE: since this is a header, it may be included in other classes as well;
//       to avoid problems, the debug level is reset at the end of this file.
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

/*!
 * \brief   Time-Varying Block Code support.
 * \author  Johann Briffa
 *
 * \tparam sig Channel symbol type
 * \tparam real Floating-point type for returning results
 * \tparam real2 Floating-point type for internal computation
 */

template <class sig, class real, class real2>
class tvb_receiver {
public:
   /*! \name Type definitions */
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<array1s_t> array1vs_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<real2> array1r2_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   /*! \name User-defined parameters */
   mutable array2vs_t encoding_table; //!< Local copy of per-frame encoding table
   std::auto_ptr<typename channel_insdel<sig, real2>::metric_computer> computer; //!< Channel object for computing receiver metric
   // @}
public:
   /*! \name User initialization (can be adapted for needs of user class) */
   /*! \brief Set up channel receiver
    * Only needs to be done before the first frame.
    */
   void init(const typename libcomm::channel_insdel<sig, real2>::metric_computer& computer)
      {
      this->computer.reset(
            dynamic_cast<typename channel_insdel<sig, real2>::metric_computer*>(computer.clone()));
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "T = " << computer.T << std::endl;
      std::cerr << "mT_min = " << computer.mT_min << std::endl;
      std::cerr << "mT_max = " << computer.mT_max << std::endl;
      std::cerr << "m1_min = " << computer.m1_min << std::endl;
      std::cerr << "m1_max = " << computer.m1_max << std::endl;
      std::cerr << "Rval = " << computer.Rval << std::endl;
      std::cerr << "Rtable = " << libbase::matrix<real2>(computer.Rtable) << std::endl;
#endif
      }
   /*! \brief Set up encoding table
    * Needs to be done before every frame.
    */
   void init(const array2vs_t& encoding_table) const
      {
      this->encoding_table = encoding_table;
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "encoding_table = " << this->encoding_table << std::endl;
      std::cerr << "sizeof(encoding_table) = " << sizeof(this->encoding_table) << std::endl;
#endif
      }
   // @}
   /*! \name Interface with fba2 algorithm (cannot be changed) */
   //! Batch receiver interface
   void R(int d, int i, const array1s_t& r, const array1vd_t& app,
         array1r_t& ptable) const
      {
      // 'tx' is the vector of transmitted symbols that we're considering
      const array1s_t& tx = encoding_table(i, d);
      // set up space for results
      static array1r2_t ptable_r;
      ptable_r.init(ptable.size());
      // call batch receiver method
      computer->receive(tx, r, ptable_r);
      // apply priors at codeword level if applicable
      if (app.size() > 0)
         ptable_r *= real2(app(i)(d));
      // convert results
      ptable = ptable_r;
      }
   // @}
};

// Reset debug level, to avoid affecting other files
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG
#endif

} // end namespace

#endif
