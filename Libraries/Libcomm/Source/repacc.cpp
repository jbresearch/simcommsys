/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "repacc.h"
#include <sstream>
#include <iomanip>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Show intermediate decoding output
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// initialization / de-allocation

template <class real, class dbl>
void repacc<real, dbl>::init()
   {
   // check presence of components
   assertalways(inter);
   assertalways(acc);
   // check repeat code
   assertalways(This::input_block_size() > 0);
   assertalways(rep.num_inputs() == This::num_inputs());
   // initialize BCJR subsystem for accumulator
   BCJR::init(*acc, This::output_block_size());
   // check interleaver size
   assertalways(inter->size() == This::output_block_size());
   assertalways(iter > 0);

   initialised = false;
   }

template <class real, class dbl>
void repacc<real, dbl>::free()
   {
   if (acc != NULL)
      delete acc;
   if (inter != NULL)
      delete inter;
   }

template <class real, class dbl>
void repacc<real, dbl>::reset()
   {
   if (endatzero)
      {
      BCJR::setstart(0);
      BCJR::setend(0);
      }
   else
      {
      BCJR::setstart(0);
      BCJR::setend();
      }
   }

// memory allocator (for internal use only)

template <class real, class dbl>
void repacc<real, dbl>::allocate()
   {
   rp.init(This::input_block_size());
   for (int i = 0; i < This::input_block_size(); i++)
      rp(i).init(This::num_inputs());
   //rp.init(This::input_block_size(), This::num_inputs());
   ra.init(This::output_block_size(), acc->num_inputs());
   R.init(This::output_block_size(), acc->num_outputs());

   // determine memory occupied and tell user
   std::ios::fmtflags flags = std::cerr.flags();
   std::cerr << "RepAcc Memory Usage: " << std::fixed << std::setprecision(1);
   std::cerr << (rp.size() + ra.size() + R.size()) * sizeof(dbl) / double(1
         << 20) << "MB\n";
   std::cerr.setf(flags);
   // flag the state of the arrays
   initialised = true;
   }

// constructor / destructor

template <class real, class dbl>
repacc<real, dbl>::repacc() :
   inter(NULL), acc(NULL)
   {
   }

// internal codec functions

template <class real, class dbl>
void repacc<real, dbl>::resetpriors()
   {
   // Should be called after setreceivers()
   assertalways(initialised);
   // Initialise intrinsic source statistics (natural)
   rp = 1.0;
   }

template <class real, class dbl>
void repacc<real, dbl>::setpriors(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_inputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::input_block_size());
   // Take into account intrinsic source statistics
   for (int t = 0; t < This::input_block_size(); t++)
      for (int i = 0; i < This::num_inputs(); i++)
         rp(t)(i) *= ptable(t)(i);
   }

/*! \copydoc codec_softout::setreceiver()

 Sets: ra, R

 \note The BCJR normalization method is used to normalize the channel-derived
 (intrinsic) probabilities 'r' and 'R'; in view of this, the a-priori
 probabilities are now created normalized.
 */
template <class real, class dbl>
void repacc<real, dbl>::setreceiver(const array1vd_t& ptable)
   {
   // Encoder symbol space must be the same as modulation symbol space
   assertalways(ptable.size() > 0);
   assertalways(ptable(0).size() == This::num_outputs());
   // Confirm input sequence to be of the correct length
   assertalways(ptable.size() == This::output_block_size());

   // initialise memory if necessary
   if (!initialised)
      allocate();

   // Initialise extrinsic accumulator-input statistics (natural)
   ra = 1.0;
   // Determine intrinsic accumulator-output statistics (interleaved)
   // from the channel
   R = 0.0;
   for (int i = 0; i < This::output_block_size(); i++)
      for (int x = 0; x < This::num_outputs(); x++)
         for (int j = 0; j < This::num_inputs(); j++)
            R(i, x * This::num_inputs() + j) = dbl(ptable(i)(x));
   BCJR::normalize(R);

   // Reset start- and end-state probabilities
   reset();
   }

// encoding and decoding functions

template <class real, class dbl>
void repacc<real, dbl>::seedfrom(libbase::random& r)
   {
   inter->seedfrom(r);
   }

template <class real, class dbl>
void repacc<real, dbl>::encode(const array1i_t& source, array1i_t& encoded)
   {
   assert(source.size() == This::input_block_size());
   // Compute repeater output
   array1i_t rep0;
   rep.encode(source, rep0);
   // Copy and add any necessary tail
   array1i_t rep1(This::output_block_size());
   rep1.copyfrom(rep0);
   for (int i = rep0.size(); i < rep1.size(); i++)
      rep1(i) = fsm::tail;
   // Create interleaved sequence
   array1i_t rep2;
   inter->advance();
   inter->transform(rep1, rep2);

   // Initialise result vector
   encoded.init(This::output_block_size());
   // Reset the encoder to zero state
   acc->reset(0);
   // Encode sequence
   for (int i = 0; i < This::output_block_size(); i++)
      encoded(i) = acc->step(rep2(i)) / This::num_inputs();
   // check that encoder finishes correctly
   if (endatzero)
      assertalways(acc->state() == 0);
   }

/*! \copydoc codec_softout::softdecode()

 \note Implements soft-decision decoding according to Alexandre's
 interpretation:
 - when computing final output at repetition code, use only extrinsic
 information from accumulator
 - when computing extrinsic output at rep code, factor out the input
 information at that position
 */
template <class real, class dbl>
void repacc<real, dbl>::softdecode(array1vd_t& ri)
   {
   // decode accumulator

   // Temporary variables to hold posterior probabilities and
   // interleaved versions of ra/ri
   array2d_t rif, rai, rii;
   inter->transform(ra, rai);
   BCJR::fdecode(R, rai, rii);
   inter->inverse(rii, rif);
   // compute extrinsic information
   rif.mask(ra > 0).divideby(ra);
   ra = rif;
   BCJR::normalize(ra);

   // allocate space for interim results
   const int Nr = rep.output_block_size();
   const int q = rep.num_outputs();
   assertalways(ra.size().rows() >= Nr);
   assertalways(ra.size().cols() == q);
   array1vd_t ravd;
   ravd.init(Nr);
   for (int i = 0; i < Nr; i++)
      ravd(i).init(q);
   // convert interim results
   for (int i = 0; i < Nr; i++)
      for (int x = 0; x < q; x++)
         ravd(i)(x) = ra(i, x);

#if DEBUG>=2
   array1i_t dec;
   This::hard_decision(ravd,dec);
   libbase::trace << "DEBUG (repacc): ravd = ";
   dec.serialize(libbase::trace, ' ');
#endif

   // decode repetition code (based on extrinsic information only)
   array1vd_t ro;
   rep.init_decoder(ravd, rp);
   rep.softdecode(ri, ro);

#if DEBUG>=2
   This::hard_decision(ro,dec);
   libbase::trace << "DEBUG (repacc): ro = ";
   dec.serialize(libbase::trace, ' ');
   This::hard_decision(ri,dec);
   libbase::trace << "DEBUG (repacc): ri = ";
   dec.serialize(libbase::trace, ' ');
#endif

   // compute extrinsic information
   // TODO: figure out how to deal with tail
   for (int i = 0; i < Nr; i++)
      for (int x = 0; x < q; x++)
         if (ra(i, x) > dbl(0))
            ra(i, x) = ro(i)(x) / ra(i, x);
         else
            ra(i, x) = ro(i)(x);

   // normalize results
   BCJR::normalize(ra);
   }

template <class real, class dbl>
void repacc<real, dbl>::softdecode(array1vd_t& ri, array1vd_t& ro)
   {
   failwith("Not yet implemented");
   }

// description output

template <class real, class dbl>
std::string repacc<real, dbl>::description() const
   {
   std::ostringstream sout;
   sout << "Repeat-Accumulate Code - ";
   sout << rep.description() << ", ";
   sout << acc->description() << ", ";
   sout << inter->description() << ", ";
   sout << iter << " iterations, ";
   sout << (endatzero ? "Terminated" : "Unterminated");
   return sout.str();
   }

// object serialization - saving

template <class real, class dbl>
std::ostream& repacc<real, dbl>::serialize(std::ostream& sout) const
   {
   // format version
   sout << 2 << '\n';
   rep.serialize(sout);
   sout << acc;
   sout << inter;
   sout << iter << '\n';
   sout << int(endatzero) << '\n';
   return sout;
   }

// object serialization - loading

template <class real, class dbl>
std::istream& repacc<real, dbl>::serialize(std::istream& sin)
   {
   assertalways(sin.good());
   free();
   // get format version
   int version;
   sin >> version;
   assertalways(version >= 2);
   // get second-version items
   rep.serialize(sin);
   sin >> acc;
   sin >> inter;
   sin >> iter;
   sin >> endatzero;
   init();
   assertalways(sin.good());
   return sin;
   }

} // end namespace

// Explicit Realizations

#include "logrealfast.h"

namespace libcomm {

using libbase::logrealfast;
using libbase::serializer;

template class repacc<float, float> ;
template <>
const serializer repacc<float, float>::shelper = serializer("codec",
      "repacc<float>", repacc<float, float>::create);

template class repacc<double> ;
template <>
const serializer repacc<double>::shelper = serializer("codec",
      "repacc<double>", repacc<double>::create);

template class repacc<logrealfast> ;
template <>
const serializer repacc<logrealfast>::shelper = serializer("codec",
      "repacc<logrealfast>", repacc<logrealfast>::create);

template class repacc<logrealfast, logrealfast> ;
template <>
const serializer repacc<logrealfast, logrealfast>::shelper = serializer(
      "codec", "repacc<logrealfast,logrealfast>", repacc<logrealfast,
            logrealfast>::create);

} // end namespace
