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

#include "tvb.h"
#include "sparse.h"
#include "timer.h"
#include "pacifier.h"
#include "vectorutils.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show codebooks on frame advance
// 3 - Show input and output of encoding process
// 9 - Show prior and posterior sof/eof probabilities when decoding
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

template <class real, class sig, bool norm>
libbase::vector<libbase::vector<sig> > tvb<real, sig, norm>::select_codebook(
      const int i) const
   {
   // Inherit sizes
   const int q = num_symbols();
   // Initialize space for result
   array1vs_t codebook;
   libbase::allocate(codebook, q, n);
   // Select codebook
   int cb_index = 0;
   switch (codebook_type)
      {
      case codebook_sparse:
         assert(num_codebooks() == 1);
         cb_index = 0;
         break;

      case codebook_random:
         for (int d = 0; d < q; d++)
            for (bool ready = false; !ready;)
               {
               for (int s = 0; s < n; s++)
                  codebook(d)(s) = sig(this->r.ival(sig::elements()));
               // this entry should be distinct from earlier entries
               ready = true;
               for (int dd = 0; dd < d; dd++)
                  if (codebook(dd).isequalto(codebook(d)))
                     {
                     ready = false;
                     break;
                     }
               }
         return codebook;

      case codebook_user_sequential:
         assert(num_codebooks() >= 1);
         cb_index = i % num_codebooks();
         break;

      case codebook_user_random:
         assert(num_codebooks() >= 1);
         cb_index = r.ival(num_codebooks());
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   // Copy chosen codebook and return
   for (int d = 0; d < num_symbols(); d++)
      codebook(d) = codebook_tables(cb_index, d);
   return codebook;
   }

template <class real, class sig, bool norm>
libbase::vector<sig> tvb<real, sig, norm>::select_marker(const int i) const
   {
   // Initialize space for result
   array1s_t marker;
   marker.init(n);
   // Select marker vector
   int mv_index = 0;
   switch (marker_type)
      {
      case marker_zero:
         marker = 0;
         return marker;

      case marker_random:
         for (int s = 0; s < n; s++)
            marker(s) = sig(this->r.ival(sig::elements()));
         return marker;

      case marker_user_sequential:
         assert(marker_vectors.size() >= 1);
         mv_index = i % marker_vectors.size();
         break;

      case marker_user_random:
         assert(marker_vectors.size() >= 1);
         mv_index = r.ival(marker_vectors.size());
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   // Copy chosen marker vector and return
   marker = marker_vectors(mv_index);
   return marker;
   }

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::advance() const
   {
   // Inherit sizes
   const int q = num_symbols();
   const int N = this->input_block_size();
   // Advance only for non-zero block sizes
   if (N > 0)
      {
      // Initialize space for encoding table
      libbase::allocate(encoding_table, N, q, n);
      // Go through each symbol index
      for (int i = 0; i < N; i++)
         {
         // Select codebook and marker vector
         const array1vs_t codebook = select_codebook(i);
         const array1s_t marker = select_marker(i);
#if DEBUG>=2
         std::cerr << "Codebook for i = " << i << std::endl;
         showcodebook(std::cerr, codebook);
         std::cerr << "Marker for i = " << i << std::endl;
         std::cerr << "\t";
         marker.serialize(std::cerr, ' ');
#endif
         // Encode each possible input symbol
         for (int d = 0; d < q; d++)
            encoding_table(i, d) = marker + codebook(d);
         }
      }
   // initialize our embedded metric computer
   fba.get_receiver().init(encoding_table);
   }

// encoding and decoding functions

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::domodulate(const int N, const array1i_t& encoded,
      array1s_t& tx)
   {
   // TODO: when N is removed from the interface, rename 'tau' to 'N'
   // Inherit sizes
   const int q = num_symbols();
   const int tau = this->input_block_size();
   // Check validity
   assertalways(tau == encoded.size());
   // Each 'encoded' symbol must be representable by a single codeword
   assertalways(N == q);
   // Initialise result vector
   tx.init(n * tau);
   assertalways(encoding_table.size().rows() == tau);
   assertalways(encoding_table.size().cols() == q);
   // Encode source stream
   for (int i = 0; i < tau; i++)
      tx.segment(i * n, n) = encoding_table(i, encoded(i));
   }

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, array1vd_t& ptable)
   {
   const array1vd_t app; // empty APP table
   dodemodulate(chan, rx, app, ptable);
   }

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const array1vd_t& app, array1vd_t& ptable)
   {
   // Initialize for known-start
   init(chan);
   // Shorthand for transmitted and received frame sizes
   const int tau = this->output_block_size();
   const int rho = rx.size();
   // Algorithm parameters
   const int xmax = fba.get_xmax();
   // Check that rx size is within valid range
   assertalways(xmax >= abs(rho - tau));
   // Set up start-of-frame drift pdf (drift = 0)
   array1d_t sof_prior;
   sof_prior.init(2 * xmax + 1);
   sof_prior = 0;
   sof_prior(xmax + 0) = 1;
   // Set up end-of-frame drift pdf (drift = rho-tau)
   array1d_t eof_prior;
   eof_prior.init(2 * xmax + 1);
   eof_prior = 0;
   eof_prior(xmax + rho - tau) = 1;
   // Offset rx by xmax and pad to a total size of tau+2*xmax
   array1s_t r;
   r.init(tau + 2 * xmax);
   r.segment(xmax, rho) = rx;
   // Delegate
   array1d_t sof_post;
   array1d_t eof_post;
   demodulate_wrapper(chan, r, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, libbase::size_type<libbase::vector>(xmax));
   }

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::dodemodulate(const channel<sig>& chan,
      const array1s_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const libbase::size_type<
            libbase::vector> offset)
   {
   // Initialize for known-start
   init(chan, sof_prior, offset);
   // TODO: validate priors have required size?
#ifndef NDEBUG
   std::cerr << "DEBUG (tvb): offset = " << offset << ", xmax = "
         << fba.get_xmax() << "." << std::endl;
#endif
   assert(offset == fba.get_xmax());
   // Delegate
   demodulate_wrapper(chan, rx, sof_prior, eof_prior, app, ptable, sof_post,
         eof_post, offset);
   }

/*!
 * \brief Wrapper for calling demodulation algorithm
 *
 * This method assumes that the init() method has already been called with
 * the appropriate parameters.
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::demodulate_wrapper(const channel<sig>& chan,
      const array1s_t& rx, const array1d_t& sof_prior,
      const array1d_t& eof_prior, const array1vd_t& app, array1vd_t& ptable,
      array1d_t& sof_post, array1d_t& eof_post, const int offset)
   {
   // Call FBA and normalize results
#if DEBUG>=9
   using libbase::index_of_max;
   std::cerr << "sof_prior = " << sof_prior << std::endl;
   std::cerr << "max at " << index_of_max(sof_prior) - offset << std::endl;
   std::cerr << "eof_prior = " << eof_prior << std::endl;
   std::cerr << "max at " << index_of_max(eof_prior) - offset << std::endl;
#endif
   array1vr_t ptable_r;
   array1r_t sof_post_r;
   array1r_t eof_post_r;
   fba.decode(*this, rx, sof_prior, eof_prior, app, ptable_r, sof_post_r,
         eof_post_r, offset);
   normalize_results(ptable_r, ptable);
   normalize(sof_post_r, sof_post);
   normalize(eof_post_r, eof_post);
#if DEBUG>=9
   std::cerr << "sof_post = " << sof_post << std::endl;
   std::cerr << "max at " << index_of_max(sof_post) - offset << std::endl;
   std::cerr << "eof_post = " << eof_post << std::endl;
   std::cerr << "max at " << index_of_max(eof_post) - offset << std::endl;
#endif
   }

/*!
 * \brief Normalize probability table
 *
 * The input probability table is normalized such that the largest value is
 * equal to 1; result is converted to double.
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::normalize(const array1r_t& in, array1d_t& out)
   {
   const int N = in.size();
   assert(N > 0);
   // check for numerical underflow
   real scale = in.max();
   assert(scale != real(0));
   scale = real(1) / scale;
   // allocate result space
   out.init(N);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      out(i) = in(i) * scale;
   }

/*!
 * \brief Normalize probability table
 *
 * The input probability table is normalized such that the largest value is
 * equal to 1; result is converted to double.
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::normalize_results(const array1vr_t& in,
      array1vd_t& out)
   {
   const int N = in.size();
   assert(N > 0);
   const int q = in(0).size();
   // check for numerical underflow
   real scale = 0;
   for (int i = 0; i < N; i++)
      scale = std::max(scale, in(i).max());
   assert(scale != real(0));
   scale = real(1) / scale;
   // allocate result space
   libbase::allocate(out, N, q);
   // normalize and copy results
   for (int i = 0; i < N; i++)
      for (int d = 0; d < q; d++)
         out(i)(d) = in(i)(d) * scale;
   }

// Setup procedure

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::init(const channel<sig>& chan,
      const array1d_t& sof_pdf, const int offset)
   {
   // Inherit block size from last modulation step
   const int q = num_symbols();
   const int N = this->input_block_size();
   const int tau = N * n;
   assert(N > 0);
   // Copy channel for access within R()
   mychan = dynamic_cast<const qids<sig>&> (chan);
   // Set channel block size to q-ary symbol size
   mychan.set_blocksize(n);
   // Determine required FBA parameter values
   const int I = mychan.compute_I(tau);
   // No need to recompute xmax if we are given a prior PDF
   const int xmax = sof_pdf.size() > 0 ? offset : mychan.compute_xmax(tau,
         sof_pdf, offset);
   const int dxmax = mychan.compute_xmax(n);
   checkforchanges(I, xmax);
   // Initialize forward-backward algorithm
   fba.init(N, n, q, I, xmax, dxmax, th_inner, th_outer);
   // initialize our embedded metric computer with unchanging elements
   fba.get_receiver().init(n, mychan);
   }

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::init()
   {
   // Build codebook if necessary
   switch (codebook_type)
      {
      case codebook_sparse:
         {
         // sparse codebooks defined only for GF(2)
         assertalways(sig::elements() == 2);
         const int q = num_symbols();
         libbase::sparse mycodebook(q, n);
         codebook_tables.init(1, q);
         for (int i = 0; i < q; i++)
            {
            const libbase::bitfield codeword(mycodebook(i), n);
            codebook_tables(0, i) = codeword.asvector();
            }
         }
         break;

      case codebook_random:
         // gets generated per symbol index on advance()
         codebook_tables.init(0, 0);
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         // nothing to do
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
#ifndef NDEBUG
   // If applicable, display codebook
   if (n > 2 && codebook_type != codebook_random)
      showcodebooks(libbase::trace);
#endif
   // If necessary, validate codebook
   if (codebook_type != codebook_random)
      validatecodebook();
   // Seed the generator and clear the per-frame encoding table
   r.seed(0);
   encoding_table.init(0, 0);
   // Check that everything makes sense
   test_invariant();
   }

/*!
 * \brief Check that all entries in table have correct length
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::validate_sequence_length(const array1vs_t& table) const
   {
   assertalways(table.size() > 0);
   for (int i = 0; i < table.size(); i++)
      assertalways(table(i).size() == n);
   }

/*!
 * \brief Set up marker sequence for the current frame as given
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::copymarker(const array1vs_t& marker_s)
   {
   validate_sequence_length(marker_s);
   encoding_table = marker_s;
   }

/*!
 * \brief Set up codebook with the given codewords
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::copycodebook(const int i,
      const array1vs_t& codebook_s)
   {
   assertalways(codebook_s.size() == num_symbols());
   validate_sequence_length(codebook_s);
   // insert into matrix
   assertalways(i >= 0 && i < num_codebooks());
   assertalways(codebook_tables.size().cols() == num_symbols());
   codebook_tables.insertrow(codebook_s, i);
   }

/*!
 * \brief Display given codebook on given stream
 */

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::showcodebook(std::ostream& sout,
      const array1vs_t& codebook) const
   {
   assert(codebook.size() == num_symbols());
   for (int d = 0; d < num_symbols(); d++)
      {
      sout << d << "\t";
      codebook(d).serialize(sout, ' ');
      }
   }

/*!
 * \brief Display codebook on given stream
 */

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::showcodebooks(std::ostream& sout) const
   {
   assert(num_codebooks() >= 1);
   assert(codebook_tables.size().cols() == num_symbols());
   for (int i = 0; i < num_codebooks(); i++)
      {
      sout << "Codebook " << i << ":" << std::endl;
      showcodebook(sout, codebook_tables.extractrow(i));
      }
   }

/*!
 * \brief Confirm that codebook is valid
 * Checks that all codebook entries are within range and that there are no
 * duplicate entries.
 */

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::validatecodebook() const
   {
   assertalways(num_codebooks() >= 1);
   assertalways(codebook_tables.size().cols() == num_symbols());
   for (int i = 0; i < num_codebooks(); i++)
      for (int d = 0; d < num_symbols(); d++)
         {
         // all entries should be distinct for each codebook index
         for (int dd = 0; dd < d; dd++)
            assertalways(codebook_tables(i, dd).isnotequalto(codebook_tables(i, d)));
         //assertalways(codebook(i, dd) != codebook(i, d));
         }
   }

//! Inform user if I or xmax have changed

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::checkforchanges(int I, int xmax) const
   {
   static int last_I = 0;
   static int last_xmax = 0;
   if (last_I != I || last_xmax != xmax)
      {
      std::cerr << "TVB: I = " << I << ", xmax = " << xmax << std::endl;
      last_I = I;
      last_xmax = xmax;
      }
   }

// Marker-specific setup functions

/*!
 * \brief Overrides the internally-generated marker sequence with given one
 *
 * The intent of this method is to allow users to apply the tvb decoder
 * in derived algorithms, such as the 2D extension.
 *
 * \todo merge with copymarker()
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::set_marker(const array1vs_t& marker_s)
   {
   copymarker(marker_s);
   }

/*!
 * \brief Overrides the codebook with given one
 *
 * The intent of this method is to allow users to apply the tvb decoder
 * in derived algorithms, such as the 2D extension.
 */
template <class real, class sig, bool norm>
void tvb<real, sig, norm>::set_codebook(const array1vs_t& codebook_s)
   {
   // allocate memory and copy read codebook
   codebook_tables.init(1, num_symbols());
   copycodebook(0, codebook_s);
   }

template <class real, class sig, bool norm>
void tvb<real, sig, norm>::set_thresholds(const real th_inner,
      const real th_outer)
   {
   This::th_inner = th_inner;
   This::th_outer = th_outer;
   test_invariant();
   }

// description output

template <class real, class sig, bool norm>
std::string tvb<real, sig, norm>::description() const
   {
   std::ostringstream sout;
   const int q = num_symbols();
   sout << "Time-Varying Block Code (" << n << "," << q << ", ";
   switch (codebook_type)
      {
      case codebook_sparse:
         sout << "sparse codebook";
         break;

      case codebook_random:
         sout << "random codebooks";
         break;

      case codebook_user_sequential:
         sout << codebook_name << " codebook ["
               << codebook_tables.size().rows() << ", sequential]";
         break;

      case codebook_user_random:
         sout << codebook_name << " codebook ["
               << codebook_tables.size().rows() << ", random]";
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   sout << ", thresholds " << th_inner << "/" << th_outer;
   if (norm)
      sout << ", normalized";
   switch (marker_type)
      {
      case marker_zero:
         sout << ", no marker";
         break;

      case marker_random:
         sout << ", random marker";
         break;

      case marker_user_sequential:
         sout << ", AMVs [" << marker_vectors.size() << ", sequential]";
         break;

      case marker_user_random:
         sout << ", AMVs [" << marker_vectors.size() << ", random]";
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   sout << "), " << fba.description();
   return sout.str();
   }

// object serialization - saving

template <class real, class sig, bool norm>
std::ostream& tvb<real, sig, norm>::serialize(std::ostream& sout) const
   {
   sout << "# Version" << std::endl;
   sout << 1 << std::endl;
   sout << "#: Inner threshold" << std::endl;
   sout << th_inner << std::endl;
   sout << "#: Outer threshold" << std::endl;
   sout << th_outer << std::endl;
   sout << "# n" << std::endl;
   sout << n << std::endl;
   sout << "# k" << std::endl;
   sout << k << std::endl;
   sout << "# codebook type (0=sparse, 1=random, 2=user[seq], 3=user[ran])"
         << std::endl;
   sout << codebook_type << std::endl;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         sout << "#: codebook name" << std::endl;
         sout << codebook_name << std::endl;
         sout << "#: codebook count" << std::endl;
         sout << num_codebooks() << std::endl;
         assert(num_codebooks() >= 1);
         assert(codebook_tables.size().cols() == num_symbols());
         for (int i = 0; i < num_codebooks(); i++)
            {
            sout << "#: codebook entries (table " << i << ")" << std::endl;
            for (int d = 0; d < num_symbols(); d++)
               {
               codebook_tables(i, d).serialize(sout, ' ');
               //sout << std::endl;
               }
            }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   sout << "# marker type (0=zero, 1=random, 2=user[seq], 3=user[ran])"
         << std::endl;
   sout << marker_type << std::endl;
   switch (marker_type)
      {
      case marker_zero:
      case marker_random:
         break;

      case marker_user_sequential:
      case marker_user_random:
         sout << "#: marker vectors" << std::endl;
         sout << marker_vectors.size() << std::endl;
         for (int i = 0; i < marker_vectors.size(); i++)
            {
            marker_vectors(i).serialize(sout, ' ');
            //sout << std::endl;
            }
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   return sout;
   }

// object serialization - loading

/*!
 * \version 1 Initial version - based on dminner v.2, except:
 *      - no user-threshold flag (specification is mandatory)
 *      - codebook and marker type field has incompatible values
 *      - codebook stored as vectors (so lsb is on left rather than right)
 */

template <class real, class sig, bool norm>
std::istream& tvb<real, sig, norm>::serialize(std::istream& sin)
   {
   std::streampos start = sin.tellg();
   // get format version
   int version;
   sin >> libbase::eatcomments >> version;
   // read thresholds
   sin >> libbase::eatcomments >> th_inner;
   sin >> libbase::eatcomments >> th_outer;
   sin >> libbase::eatcomments >> n;
   sin >> libbase::eatcomments >> k;
   int temp;
   sin >> libbase::eatcomments >> temp;
   codebook_type = (codebook_t) temp;
   switch (codebook_type)
      {
      case codebook_sparse:
      case codebook_random:
         // gets generated automatically
         break;

      case codebook_user_sequential:
      case codebook_user_random:
         sin >> libbase::eatcomments >> codebook_name;
         // read codebook count
         sin >> libbase::eatcomments >> temp;
         // allocate memory
         codebook_tables.init(temp, num_symbols());
         for (int i = 0; i < num_codebooks(); i++)
            {
            // read codebook from stream
            array1vs_t codebook_s;
            libbase::allocate(codebook_s, num_symbols(), n);
            sin >> libbase::eatcomments;
            for (int d = 0; d < num_symbols(); d++)
               codebook_s(d).serialize(sin);
            // copy read codebook
            copycodebook(i, codebook_s);
            }
         break;

      default:
         failwith("Unknown codebook type");
         break;
      }
   // read marker sequence type
   sin >> libbase::eatcomments >> temp;
   marker_type = (marker_t) temp;
   switch (marker_type)
      {
      case marker_zero:
      case marker_random:
         // gets generated automatically
         break;

      case marker_user_sequential:
      case marker_user_random:
         // read count of modification vectors
         sin >> libbase::eatcomments >> temp;
         // read modification vectors from stream
         libbase::allocate(marker_vectors, temp, n);
         sin >> libbase::eatcomments;
         for (int i = 0; i < temp; i++)
            marker_vectors(i).serialize(sin);
         // validate list of modification vectors
         validate_sequence_length(marker_vectors);
         break;

      default:
         failwith("Unknown marker sequence type");
         break;
      }
   init();
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"
#include "gf.h"

namespace libcomm {

using libbase::serializer;
using libbase::logrealfast;
using libbase::gf;

//#ifndef USE_CUDA
#if 1
template class tvb<logrealfast, gf<1, 0x3> , false> ;
template <>
const serializer tvb<logrealfast, gf<1, 0x3> , false>::shelper = serializer(
      "blockmodem", "tvb<gf<1,0x3>,logrealfast>", tvb<logrealfast, gf<1, 0x3> ,
            false>::create);
#endif

template class tvb<float, gf<1, 0x3> , true> ;
template <>
const serializer tvb<float, gf<1, 0x3> , true>::shelper = serializer(
      "blockmodem", "tvb<gf<1,0x3>,float>",
      tvb<float, gf<1, 0x3> , true>::create);

template class tvb<double, gf<1, 0x3> , true> ;
template <>
const serializer tvb<double, gf<1, 0x3> , true>::shelper("blockmodem",
      "tvb<gf<1,0x3>,double>", tvb<double, gf<1, 0x3> , true>::create);

template <>
const serializer tvb<double, gf<2, 0x7> , true>::shelper("blockmodem",
      "tvb<gf<2,0x7>,double>", tvb<double, gf<2, 0x7> , true>::create);

template <>
const serializer tvb<double, gf<3, 0xB> , true>::shelper("blockmodem",
      "tvb<gf<3,0xB>,double>", tvb<double, gf<3, 0xB> , true>::create);

template <>
const serializer tvb<double, gf<4, 0x13> , true>::shelper("blockmodem",
      "tvb<gf<4,0x13>,double>", tvb<double, gf<4, 0x13> , true>::create);

template <>
const serializer tvb<double, gf<5, 0x25> , true>::shelper("blockmodem",
      "tvb<gf<5,0x25>,double>", tvb<double, gf<5, 0x25> , true>::create);

template <>
const serializer tvb<double, gf<6, 0x43> , true>::shelper("blockmodem",
      "tvb<gf<6,0x43>,double>", tvb<double, gf<6, 0x43> , true>::create);

template <>
const serializer tvb<double, gf<7, 0x89> , true>::shelper("blockmodem",
      "tvb<gf<7,0x89>,double>", tvb<double, gf<7, 0x89> , true>::create);

template <>
const serializer tvb<double, gf<8, 0x11D> , true>::shelper("blockmodem",
      "tvb<gf<8,0x11D>,double>", tvb<double, gf<8, 0x11D> , true>::create);

template <>
const serializer tvb<double, gf<9, 0x211> , true>::shelper("blockmodem",
      "tvb<gf<9,0x211>,double>", tvb<double, gf<9, 0x211> , true>::create);

template <>
const serializer tvb<double, gf<10, 0x409> , true>::shelper("blockmodem",
      "tvb<gf<10,0x409>,double>", tvb<double, gf<10, 0x409> , true>::create);

} // end namespace
