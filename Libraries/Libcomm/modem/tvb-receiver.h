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

#ifndef __tvb_receiver_h
#define __tvb_receiver_h

#include "config.h"
#include "channel/qids.h"

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
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
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
   int n; //!< Number of bits per codeword
   bool splitpriors; //!< Flag indicating channel-symbol-level priors
   mutable array2vs_t encoding_table; //!< Local copy of per-frame encoding table
   typename qids<sig, real2>::metric_computer computer; //!< Channel object for computing receiver metric
   // @}
public:
   /*! \name User initialization (can be adapted for needs of user class) */
   /*! \brief Set up code size and channel receiver
    * Only needs to be done before the first frame.
    */
   void init(const int n, const bool splitpriors, const libcomm::qids<sig, real2>& chan)
      {
      this->n = n;
      this->splitpriors = splitpriors;
      computer = chan.get_computer();
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "n = " << this->n << std::endl;
      std::cerr << "N = " << computer.N << std::endl;
      std::cerr << "I = " << computer.I << std::endl;
      std::cerr << "xmax = " << computer.xmax << std::endl;
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
   //! Determine priors for each transmitted symbol
   void compute_priors(const int i, const array1s_t& tx, const array1vd_t& app,
         array1r2_t& tx_app) const
      {
      // short-circuit when no priors were submitted
      if (app.size() == 0)
         {
         tx_app.init(0);
         return;
         }
      // convert codeword priors to symbol priors
      tx_app.init(n);
      const int q = app(i).size();
      for (int j = 0; j < n; j++)
         {
         real2 p = 0;
         for (int d = 0; d < q; d++)
            {
            if(tx(j) == encoding_table(i,d)(j))
               p += real2(app(i)(d));
            }
         tx_app(j) = p;
         }
      }
   // @}
   /*! \name Interface with fba2 algorithm (cannot be changed) */
   //! Receiver interface
   real R(int d, int i, const array1s_t& r, const array1vd_t& app) const
      {
      // 'tx' is the vector of transmitted symbols that we're considering
      const array1s_t& tx = encoding_table(i, d);
      if(splitpriors)
         {
         // determine priors for each transmitted symbol
         static array1r2_t tx_app;
         compute_priors(i, tx, app, tx_app);
         // compute the conditional probability
         return real(computer.receive(tx, r, tx_app));
         }
      else
         {
         // empty channel-symbol-level priors
         array1r2_t tx_app;
         // compute the conditional probability
         real result = real(computer.receive(tx, r, tx_app));
         // apply priors at codeword level if applicable
         if (app.size() > 0)
            result *= real(app(i)(d));
         return result;
         }
      }
   //! Batch receiver interface
   void R(int d, int i, const array1s_t& r, const array1vd_t& app,
         array1r_t& ptable) const
      {
      // 'tx' is the vector of transmitted symbols that we're considering
      const array1s_t& tx = encoding_table(i, d);
      // set up space for results
      static array1r2_t ptable_r;
      ptable_r.init(ptable.size());
      if(splitpriors)
         {
         // determine priors for each transmitted symbol
         static array1r2_t tx_app;
         compute_priors(i, tx, app, tx_app);
         // call batch receiver method and convert results
         computer.receive(tx, r, tx_app, ptable_r);
         ptable = ptable_r;
         }
      else
         {
         // empty channel-symbol-level priors
         array1r2_t tx_app;
         // call batch receiver method
         computer.receive(tx, r, tx_app, ptable_r);
         // apply priors at codeword level if applicable
         if (app.size() > 0)
            ptable_r *= real2(app(i)(d));
         // convert results
         ptable = ptable_r;
         }
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
