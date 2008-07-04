#ifndef __dminner2_h
#define __dminner2_h

#include "config.h"

#include "dminner.h"
#include "fba2.h"

namespace libcomm {

/*!
   \brief   Davey's Watermark Code.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   Implements a novel (and more accurate) decoding algorithm for the inner
   codes described by Davey in "Reliable Communication over Channels with
   Insertions, Deletions, and Substitutions", Trans. IT, Feb 2001.
*/

template <class real, bool normalize>
class dminner2 : public dminner<real,normalize>, private fba2<real,bool,normalize> {
private:
   // Implementations of channel-specific metrics for fba2
   real Q(int d, int i, const libbase::vector<bool>& r) const;
public:
   /*! \name Constructors / Destructors */
   dminner2(const int n=2, const int k=1)
      : dminner<real,normalize>(n,k) {};
   dminner2(const int n, const int k, const double th_inner, const double th_outer)
      : dminner<real,normalize>(n,k,th_inner,th_outer) {};
   // @}

   // Vector modem operations
   void demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable);

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dminner2)
};

}; // end namespace

#endif
