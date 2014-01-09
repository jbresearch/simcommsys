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

#include "exit_computer.h"

#include "modem/informed_modulator.h"
#include "codec/codec_softout.h"
#include "vector_itfunc.h"
#include "fsm.h"
#include "itfunc.h"
#include "secant.h"
#include "timer.h"
#include <iostream>
#include <sstream>

namespace libcomm {

// *** Templated Common Base ***

// Internal functions

/*!
 * \brief Create source sequence to be encoded
 * \return Source sequence of the required length
 *
 * The source sequence consists of uniformly random symbols followed by a
 * tail sequence if required by the given codec.
 */
template <class S>
libbase::vector<int> exit_computer<S>::createsource()
   {
   const int tau = sys->input_block_size();
   array1i_t source(tau);
   for (int t = 0; t < tau; t++)
      source(t) = src.ival(sys->num_inputs());
   return source;
   }

/*!
 * \brief Create table of Gaussian-distributed priors
 * \param[out] priors Table of Gaussian-distributed priors for given input
 * \param[in] tx Vector of transmitted symbols
 */
template <class S>
libbase::vector<libbase::vector<double> > exit_computer<S>::createpriors(
      const array1i_t& tx)
   {
   // determine sizes
   const int N = sys->getmodem()->input_block_size();
   const int q = sys->getmodem()->num_symbols();
   const int k = int(log2(q));
   assert(tx.size() == N);
   assert(q == (1<<k));
   // allocate space for results
   array1vd_t priors;
   libbase::allocate(priors, N, q);
   // allocate space for temporary binary LLRs
   array1d_t llr(k);
   // determine random priors
   for (int i = 0; i < N; i++)
      {
      const int cw = tx(i);
      assert(cw >= 0 && cw < q);
      // generate random LLRs (for given codeword)
      // Note: i. LLR is interpreted as ln(Pr(0)/Pr(1))
      //       ii. vector is given lsb first
      for (int j = 0; j < k; j++)
         {
         llr(j) = src.gval(sigma);
         if ((cw & (1 << j)) == 0)
            llr(j) += sigma / 2;
         else
            llr(j) -= sigma / 2;
         }
      // determine non-binary priors from binary ones
      for (int d = 0; d < q; d++)
         {
         double p = 1;
         for (int j = 0; j < k; j++)
            {
            const double lr = exp(llr(j)); // = p0/p1 = p0/(1-p0) = (1-p1)/p1
            if ((d & (1 << j)) == 0)
               p *= lr / (1 + lr); // = p0
            else
               p *= 1 / (1 + lr); // = p1
            }
         priors(i)(d) = p;
         }
      }
   return priors;
   }

/*!
 * \brief Determine the mutual information between x and p
 * \param x The known transmitted sequence
 * \param p The probability table at the receiving end p(y)
 *
 * I(X;Y) = H(Y) - H(Y|X)
 * where
 * H(Y) = ∑ -p(y) . log₂ p(y)
 * H(Y|X) = ∑ p(x) ∑ -p(y|x) . log₂ p(y|x)
 * for known X, p(x)=1 only at given x, so that
 * H(Y|X) = ∑ -p(y|x) . log₂ p(y|x) (for given x)
 */
template <class S>
double exit_computer<S>::compute_mutual_information(const array1i_t& x, const array1vd_t& p)
   {
   // determine sizes
   const int N = p.size();
   assert(N > 0);
   assert(x.size() == N);
   // compute conditional entropy
   double H = 0;
   for (int i = 0; i < N; i++)
      {
      const int d = x(i);
      H += -p(i)(d) * log2(p(i)(d));
      }
   H /= N;
   return libbase::compute_entropy(p) - H;
   }

// Experiment handling

/*!
 * \brief Determine mutual information at input and output of inner and outer decoders
 * \param[out] result   Vector containing the set of results to be updated
 *
 * Results are organized as ...
 */
template <class S>
void exit_computer<S>::sample(array1d_t& result)
   {
   // Initialise result vector
   result.init(count());
   result = 0;
   // Create source stream
   const array1i_t source = createsource();
   // Encode
   array1i_t encoded;
   sys->getcodec()->encode(source, encoded);
   // Map
   array1i_t mapped;
   sys->getmapper()->transform(encoded, mapped);
   // Modulate
   array1s_t transmitted;
   sys->getmodem()->modulate(sys->getmodem()->num_symbols(), mapped,
         transmitted);
   // Transmit
   const array1s_t received = sys->transmit(transmitted);
   // Create random priors
   array1vd_t priors = createpriors(mapped);
   // Demodulate
   array1vd_t ptable_mapped;
   informed_modulator<S>& m =
         dynamic_cast<informed_modulator<S>&>(*sys->getmodem());
   m.demodulate(*sys->getrxchan(), received, priors, ptable_mapped);
   // Compute extrinsic information
   libbase::compute_extrinsic(ptable_mapped, ptable_mapped, priors);
   // Inverse Map
   array1vd_t ptable_encoded;
   sys->getmapper()->inverse(ptable_mapped, ptable_encoded);
   // Translate
   sys->getcodec()->init_decoder(ptable_encoded);
   // Perform soft-output decoding for as many iterations as required
   codec_softout<libbase::vector>& c = dynamic_cast<codec_softout<
         libbase::vector>&>(*sys->getcodec());
   array1vd_t ri;
   array1vd_t ro;
   for (int i = 0; i < sys->num_iter(); i++)
      c.softdecode(ri, ro);
   // Compute extrinsic information
   libbase::compute_extrinsic(ro, ro, ptable_encoded);

   // compute results
   result(0) = compute_mutual_information(mapped, priors);
   result(1) = compute_mutual_information(mapped, ptable_mapped);
   result(2) = compute_mutual_information(encoded, ptable_encoded);
   result(3) = compute_mutual_information(encoded, ro);
   }

// Description & Serialization

template <class S>
std::string exit_computer<S>::description() const
   {
   std::ostringstream sout;
   sout << "EXIT Chart Computer for ";
   sout << sys->description();
   return sout.str();
   }

// object serialization - saving

template <class S>
std::ostream& exit_computer<S>::serialize(std::ostream& sout) const
   {
   // format version
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   // system parameter
   const double p = sys->gettxchan()->get_parameter();
   assert(p == sys->getrxchan()->get_parameter());
   sout << "# System parameter" << std::endl;
   sout << p << std::endl;
   // underlying system
   sout << sys;
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version
 */

template <class S>
std::istream& exit_computer<S>::serialize(std::istream& sin)
   {
   free();
   assertalways(sin.good());
   // get format version
   int version;
   sin >> libbase::eatcomments >> version >> libbase::verify;
   // get system parameter
   double p;
   sin >> libbase::eatcomments >> p >> libbase::verify;
   // underlying system
   sin >> libbase::eatcomments >> sys >> libbase::verify;
   // setup
   sys->gettxchan()->set_parameter(p);
   sys->getrxchan()->set_parameter(p);
   return sin;
   }

} // end namespace

#include "gf.h"

namespace libcomm {

// Explicit Realizations
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/stringize.hpp>

using libbase::serializer;

#define USING_GF(r, x, type) \
      using libbase::type;

BOOST_PP_SEQ_FOR_EACH(USING_GF, x, GF_TYPE_SEQ)

// *** General Communication System ***

#define SYMBOL_TYPE_SEQ \
   (sigspace)(bool) \
   GF_TYPE_SEQ

/* Serialization string: exit_computer<type>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 */
#define INSTANTIATE(r, x, type) \
      template class exit_computer<type>; \
      template <> \
      const serializer exit_computer<type>::shelper( \
            "experiment", \
            "exit_computer<" BOOST_PP_STRINGIZE(type) ">", \
            exit_computer<type>::create);

BOOST_PP_SEQ_FOR_EACH(INSTANTIATE, x, SYMBOL_TYPE_SEQ)

} // end namespace
