#ifndef __dminner2_h
#define __dminner2_h

#include "config.h"

#include "dminner.h"
#ifdef USE_CUDA
#  include "algorithm/fba2-cuda.h"
#  include "dminner2-receiver.h"
#else
#  include "algorithm/fba2.h"
#endif

namespace libcomm {

/*!
 * \brief   Davey-MacKay Inner Code, with symbol-level decoding.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements a novel (and more accurate) decoding algorithm for the inner
 * codes described by Davey and MacKay in "Reliable Communication over Channels
 * with Insertions, Deletions, and Substitutions", Trans. IT, Feb 2001.
 */

template <class real, bool norm>
class dminner2 : public dminner<real, norm>
#ifndef USE_CUDA
      , private fba2<real, bool, norm>
#endif
{
public:
   /*! \name Type definitions */
   typedef libbase::vector<bool> array1b_t;
   typedef libbase::vector<double> array1d_t;
   typedef libbase::vector<real> array1r_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   typedef libbase::vector<array1r_t> array1vr_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef informed_modulator<bool> Interface;
   typedef dminner2<real, norm> This;
   typedef dminner<real, norm> Base;
#ifndef USE_CUDA
   typedef fba2<real, bool, norm> FBA;
#endif
private:
#ifdef USE_CUDA
   cuda::fba2<real, bool, norm> fba;
#else
   // Implementations of channel-specific metrics for fba2
   real R(int d, int i, const array1b_t& r) const;
#endif
   // Setup procedure
   void init(const channel<bool>& chan, const int rho);
protected:
   // Interface with derived classes
   void advance() const;
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         array1vd_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx,
         const array1vd_t& app, array1vd_t& ptable);
public:
   /*! \name Constructors / Destructors */
   explicit dminner2(const int n = 2, const int k = 1) :
      dminner<real, norm> (n, k)
      {
      }
   dminner2(const int n, const int k, const double th_inner,
         const double th_outer) :
      dminner<real, norm> (n, k, th_inner, th_outer)
      {
      }
   // @}

   // Block modem operations
   // (necessary because inheriting methods from templated base)
   using Interface::modulate;
   using Interface::demodulate;

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dminner2)
};

} // end namespace

#endif
