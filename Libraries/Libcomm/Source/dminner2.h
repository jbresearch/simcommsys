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
   /*! \name Internally-used types */
   typedef libbase::vector<bool>    array1b_t;
   typedef libbase::vector< libbase::vector<double> >  array2d_t;
   typedef libbase::vector< libbase::vector<real> >    array2r_t;
   // @}
private:
   // Implementations of channel-specific metrics for fba2
   real R(int d, int i, const array1b_t& r) const;
   // Setup procedure
   void init(const channel<bool>& chan);
protected:
   // Interface with derived classes
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx, array2d_t& ptable);
   void dodemodulate(const channel<bool>& chan, const array1b_t& rx, const array2d_t& app, array2d_t& ptable);
public:
   /*! \name Constructors / Destructors */
   dminner2(const int n=2, const int k=1)
      : dminner<real,normalize>(n,k) {};
   dminner2(const int n, const int k, const double th_inner, const double th_outer)
      : dminner<real,normalize>(n,k,th_inner,th_outer) {};
   // @}

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dminner2);
};

}; // end namespace

#endif
