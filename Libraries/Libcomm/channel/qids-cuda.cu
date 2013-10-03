/*!
 * \file
 * \brief   Parallel code for BSID channel.
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

#include "qids.h"

namespace cuda {

// CUDA kernels

template <class G, class real>
__global__
void receive_trellis_kernel(const typename libcomm::qids<G,real>::metric_computer object,
      const cuda::vector_reference<G> tx, const cuda::vector_reference<G> rx,
      cuda::vector_reference<real> ptable)
   {
   object.receive_trellis(tx, rx, ptable);
   }

template <class G, class real>
__global__
void receive_lattice_kernel(const typename libcomm::qids<G,real>::metric_computer object,
      const cuda::vector_reference<G> tx, const cuda::vector_reference<G> rx,
      cuda::vector_reference<real> ptable)
   {
   object.receive_lattice(tx, rx, ptable);
   }

template <class G, class real>
__global__
void receive_lattice_corridor_kernel(const typename libcomm::qids<G,real>::metric_computer object,
      const cuda::vector_reference<G> tx, const cuda::vector_reference<G> rx,
      cuda::vector_reference<real> ptable)
   {
   object.receive_lattice_corridor(tx, rx, ptable);
   }

} // end namespace

namespace libcomm {

template <class G, class real>
void qids<G, real>::metric_computer::receive_trellis(const array1g_t& tx,
      const array1g_t& rx, array1r_t& ptable) const
   {
   // allocate space on device for result, and initialize
   cuda::vector<real> dev_ptable;
   dev_ptable.init(mT_max - mT_min + 1);
   // allocate space on device for tx and rx vectors, and copy over
   cuda::vector<G> dev_tx;
   cuda::vector<G> dev_rx;
   dev_tx = tx;
   dev_rx = rx;
   // call the kernel with a copy of this object
   cuda::receive_trellis_kernel<G,real> <<<1,1>>>(*this, dev_tx, dev_rx, dev_ptable);
   // return the result
   ptable = array1r_t(dev_ptable);
   }

template <class G, class real>
void qids<G, real>::metric_computer::receive_lattice(const array1g_t& tx,
      const array1g_t& rx, array1r_t& ptable) const
   {
   // allocate space on device for result, and initialize
   cuda::vector<real> dev_ptable;
   dev_ptable.init(mT_max - mT_min + 1);
   // allocate space on device for tx and rx vectors, and copy over
   cuda::vector<G> dev_tx;
   cuda::vector<G> dev_rx;
   dev_tx = tx;
   dev_rx = rx;
   // call the kernel with a copy of this object
   cuda::receive_lattice_kernel<G,real> <<<1,1>>>(*this, dev_tx, dev_rx, dev_ptable);
   // return the result
   ptable = array1r_t(dev_ptable);
   }

template <class G, class real>
void qids<G, real>::metric_computer::receive_lattice_corridor(const array1g_t& tx,
      const array1g_t& rx, array1r_t& ptable) const
   {
   // allocate space on device for result, and initialize
   cuda::vector<real> dev_ptable;
   dev_ptable.init(mT_max - mT_min + 1);
   // allocate space on device for tx and rx vectors, and copy over
   cuda::vector<G> dev_tx;
   cuda::vector<G> dev_rx;
   dev_tx = tx;
   dev_rx = rx;
   // call the kernel with a copy of this object
   cuda::receive_lattice_corridor_kernel<G,real> <<<1,1>>>(*this, dev_tx, dev_rx, dev_ptable);
   // return the result
   ptable = array1r_t(dev_ptable);
   }

} // end namespace

#include "gf.h"

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/enum.hpp>

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

#define SYMBOL_TYPE_SEQ \
   (bool) \
   GF_TYPE_SEQ
#define REAL_TYPE_SEQ \
   (float)(double)

namespace cuda {

#define INSTANTIATE_FUNC(r, args) \
      template __global__ void receive_trellis_kernel( \
            const typename libcomm::qids<BOOST_PP_SEQ_ENUM(args)>::metric_computer object, \
            const cuda::vector_reference<BOOST_PP_SEQ_ELEM(0,args)> tx, \
            const cuda::vector_reference<BOOST_PP_SEQ_ELEM(0,args)> rx, \
            cuda::vector_reference<BOOST_PP_SEQ_ELEM(1,args)> ptable); \
      template __global__ void receive_lattice_kernel( \
            const typename libcomm::qids<BOOST_PP_SEQ_ENUM(args)>::metric_computer object, \
            const cuda::vector_reference<BOOST_PP_SEQ_ELEM(0,args)> tx, \
            const cuda::vector_reference<BOOST_PP_SEQ_ELEM(0,args)> rx, \
            cuda::vector_reference<BOOST_PP_SEQ_ELEM(1,args)> ptable); \
      template __global__ void receive_lattice_corridor_kernel( \
            const typename libcomm::qids<BOOST_PP_SEQ_ENUM(args)>::metric_computer object, \
            const cuda::vector_reference<BOOST_PP_SEQ_ELEM(0,args)> tx, \
            const cuda::vector_reference<BOOST_PP_SEQ_ELEM(0,args)> rx, \
            cuda::vector_reference<BOOST_PP_SEQ_ELEM(1,args)> ptable);

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_FUNC, (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace

namespace libcomm {

#define INSTANTIATE_CLASS(r, args) \
      template class qids<BOOST_PP_SEQ_ENUM(args)>;

BOOST_PP_SEQ_FOR_EACH_PRODUCT(INSTANTIATE_CLASS, (SYMBOL_TYPE_SEQ)(REAL_TYPE_SEQ))

} // end namespace
