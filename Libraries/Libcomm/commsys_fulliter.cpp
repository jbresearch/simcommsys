/*!
 \file

 \section svn Version Control
 - $Revision$
 - $Date$
 - $Author$
 */

#include "commsys_fulliter.h"

#include "modem/informed_modulator.h"
#include "codec/codec_softout.h"
#include "gf.h"
#include <sstream>

namespace libcomm {

// Determine debug level:
// 1 - Normal debug output only
// 2 - Log calls to receive_path and decode
// 3 - Show part of soft information being passed around
#ifndef NDEBUG
#  undef DEBUG
#  define DEBUG 1
#endif

// Helper functions

/*! \brief Compute extrinsic information
 *
 * \param[out] re extrinsic information
 * \param[in] ro 'full' posterior information
 * \param[in] ri (extrinsic) prior information
 *
 * Computes extrinsic information as re = ro/ri, except cases where ri=0, where
 * re=ro.
 *
 * \note re may point to the same memory as ro/ri, so care must be taken.
 */
template <class S, template <class > class C>
void commsys_fulliter<S, C>::compute_extrinsic(C<array1d_t>& re, const C<
      array1d_t>& ro, const C<array1d_t>& ri)
   {
   // Handle the case where the prior information is empty
   if (ri.size() == 0)
      re = ro;
   else
      {
      // Determine size
      const int tau = ro.size();
      const int N = ro(0).size();
      // Check for validity
      assert(ri.size() == tau);
      assert(ri(0).size() == N);
      // Allocate space for re (if necessary)
      re.init(tau);
      for (int i = 0; i < tau; i++)
         re(i).init(N);
      // Perform computation
      for (int i = 0; i < tau; i++)
         for (int x = 0; x < N; x++)
            if (ri(i)(x) > 0)
               re(i)(x) = ro(i)(x) / ri(i)(x);
            else
               re(i)(x) = ro(i)(x);
      }
   }

// Communication System Interface

template <class S, template <class > class C>
void commsys_fulliter<S, C>::receive_path(const C<S>& received)
   {
#if DEBUG>=2
   libbase::trace << "DEBUG (fulliter): Starting receive path.\n";
#endif
   // Store received vector
   last_received = received;
   // Reset modem
   ptable_mapped.init(0);
   cur_mdm_iter = 0;
   // Reset decoder
   cur_cdc_iter = 0;
   }

template <class S, template <class > class C>
void commsys_fulliter<S, C>::decode(C<int>& decoded)
   {
#if DEBUG>=2
   libbase::trace << "DEBUG (fulliter): Starting decode cycle " << cur_mdm_iter
   << "/" << cur_cdc_iter << ".\n";
#endif
   // If this is the first decode cycle, we need to do the receive-path first
   if (cur_cdc_iter == 0)
      {
      // Demodulate
      C<array1d_t> ptable_full;
      informed_modulator<S>& m =
            dynamic_cast<informed_modulator<S>&> (*this->mdm);
      m.demodulate(*this->chan, last_received, ptable_mapped, ptable_full);
#if DEBUG>=3
      libbase::trace << "DEBUG (fulliter): modem soft-output = \n";
      libbase::trace << ptable_mapped.extract(0,5);
#endif
      // Compute extrinsic information for passing to codec
      compute_extrinsic(ptable_mapped, ptable_full, ptable_mapped);
      // Inverse Map
      C<array1d_t> ptable_encoded;
      this->map->inverse(ptable_mapped, ptable_encoded);
      // Translate
      this->cdc->init_decoder(ptable_encoded);
      }
   // Just do a plain decoder iteration if this is not the last one in the cycle
   if (++cur_cdc_iter < this->cdc->num_iter())
      this->cdc->decode(decoded);
   // Otherwise, do a soft-output iteration
   else
      {
      // Perform soft-output decoding
      codec_softout<C>& c = dynamic_cast<codec_softout<C>&> (*this->cdc);
      C<array1d_t> ri;
      C<array1d_t> ro;
      c.softdecode(ri, ro);
      // Compute hard-decision for results gatherer
      codec_softout<C>::hard_decision(ri, decoded);
      // TODO: Pass posterior information through mapper
      // Compute extrinsic information for next demodulation cycle
      compute_extrinsic(ptable_mapped, ro, ptable_mapped);
#if DEBUG>=3
      libbase::trace << "DEBUG (fulliter): codec soft-output = \n";
      libbase::trace << ptable_mapped.extract(0,5);
#endif
      // Reset decoder iteration count
      cur_cdc_iter = 0;
      // Update modem iteration count
      cur_mdm_iter++;
      // If this was not the last iteration, mark components as clean
      if (cur_mdm_iter < iter)
         {
         this->mdm->mark_as_clean();
         this->map->mark_as_clean();
         }
      }
   }

// Description & Serialization

template <class S, template <class > class C>
std::string commsys_fulliter<S, C>::description() const
   {
   std::ostringstream sout;
   sout << "Full-System Iterative ";
   sout << Base::description() << ", ";
   sout << iter << " iterations";
   return sout.str();
   }

template <class S, template <class > class C>
std::ostream& commsys_fulliter<S, C>::serialize(std::ostream& sout) const
   {
   sout << iter;
   Base::serialize(sout);
   return sout;
   }

template <class S, template <class > class C>
std::istream& commsys_fulliter<S, C>::serialize(std::istream& sin)
   {
   sin >> libbase::eatcomments >> iter;
   Base::serialize(sin);
   return sin;
   }

// Explicit Realizations

template class commsys_fulliter<sigspace> ;
template <>
const libbase::serializer commsys_fulliter<sigspace>::shelper("commsys",
      "commsys_fulliter<sigspace>", commsys_fulliter<sigspace>::create);

template class commsys_fulliter<bool> ;
template <>
const libbase::serializer commsys_fulliter<bool>::shelper("commsys",
      "commsys_fulliter<bool>", commsys_fulliter<bool>::create);

template class commsys_fulliter<libbase::gf<1, 0x3> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<1, 0x3> >::shelper(
      "commsys", "commsys_fulliter<gf<1,0x3>>", commsys_fulliter<libbase::gf<1,
            0x3> >::create);

template class commsys_fulliter<libbase::gf<2, 0x7> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<2, 0x7> >::shelper(
      "commsys", "commsys_fulliter<gf<2,0x7>>", commsys_fulliter<libbase::gf<2,
            0x7> >::create);

template class commsys_fulliter<libbase::gf<3, 0xB> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<3, 0xB> >::shelper(
      "commsys", "commsys_fulliter<gf<3,0xB>>", commsys_fulliter<libbase::gf<3,
            0xB> >::create);

template class commsys_fulliter<libbase::gf<4, 0x13> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<4, 0x13> >::shelper(
      "commsys", "commsys_fulliter<gf<4,0x13>>", commsys_fulliter<libbase::gf<
            4, 0x13> >::create);

template class commsys_fulliter<libbase::gf<5, 0x25> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<5, 0x25> >::shelper(
      "commsys", "commsys_fulliter<gf<5,0x25>>", commsys_fulliter<libbase::gf<
            5, 0x25> >::create);

template class commsys_fulliter<libbase::gf<6, 0x43> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<6, 0x43> >::shelper(
      "commsys", "commsys_fulliter<gf<6,0x43>>", commsys_fulliter<libbase::gf<
            6, 0x43> >::create);

template class commsys_fulliter<libbase::gf<7, 0x89> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<7, 0x89> >::shelper(
      "commsys", "commsys_fulliter<gf<7,0x89>>", commsys_fulliter<libbase::gf<
            7, 0x89> >::create);

template class commsys_fulliter<libbase::gf<8, 0x11D> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<8, 0x11D> >::shelper(
      "commsys", "commsys_fulliter<gf<8,0x11D>>", commsys_fulliter<libbase::gf<
            8, 0x11D> >::create);

template class commsys_fulliter<libbase::gf<9, 0x211> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<9, 0x211> >::shelper(
      "commsys", "commsys_fulliter<gf<9,0x211>>", commsys_fulliter<libbase::gf<
            9, 0x211> >::create);

template class commsys_fulliter<libbase::gf<10, 0x409> > ;
template <>
const libbase::serializer commsys_fulliter<libbase::gf<10, 0x409> >::shelper(
      "commsys", "commsys_fulliter<gf<10,0x409>>", commsys_fulliter<
            libbase::gf<10, 0x409> >::create);

} // end namespace
