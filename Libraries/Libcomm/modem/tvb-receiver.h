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
 * \todo Move tvb::encode() here (and access this from tvb when needed).
 *
 * \todo Change init() and internal representation so that all possible
 * encodings are done at init, rather than repeating every time.
 */

template <class real, class sig>
class tvb_receiver {
public:
   /*! \name Type definitions */
   typedef float qids_real;
   typedef libbase::vector<sig> array1s_t;
   typedef libbase::vector<array1s_t> array1vs_t;
   typedef libbase::matrix<array1s_t> array2vs_t;
   typedef libbase::vector<real> array1r_t;
   // @}
private:
   /*! \name User-defined parameters */
   int n; //!< Number of bits per codeword
   mutable array2vs_t encoding_table; //!< Local copy of per-frame encoding table
   typename qids<sig>::metric_computer computer; //!< Channel object for computing receiver metric
   // @}
public:
   /*! \name User initialization (can be adapted for needs of user class) */
   /*! \brief Set up code size and channel receiver
    * Only needs to be done before the first frame.
    */
   void init(const int n, const libcomm::qids<sig>& chan)
      {
      this->n = n;
      computer = chan.get_computer();
#if DEBUG>=2
      std::cerr << "Initialize tvb computer..." << std::endl;
      std::cerr << "n = " << this->n << std::endl;
      std::cerr << "N = " << computer.N << std::endl;
      std::cerr << "I = " << computer.I << std::endl;
      std::cerr << "xmax = " << computer.xmax << std::endl;
      std::cerr << "Rval = " << computer.Rval << std::endl;
      std::cerr << "Rtable = " << libbase::matrix<qids_real>(
            computer.Rtable) << std::endl;
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
   //! Receiver interface
   real R(int d, int i, const array1s_t& r) const
      {
      // 'tx' is the vector of transmitted symbols that we're considering
      const array1s_t& tx = encoding_table(i, d);
      // compute the conditional probability
      return computer.receive(tx, r);
      }
   //! Batch receiver interface
   void R(int d, int i, const array1s_t& r, array1r_t& ptable) const
      {
      const array1s_t& tx = encoding_table(i, d);
      // set up space for results
      static libbase::vector<qids_real> ptable_r;
      ptable_r.init(ptable.size());
      // call batch receiver method and convert results
      computer.receive(tx, r, ptable_r);
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
