#ifndef __dminner2d_h
#define __dminner2d_h

#include "config.h"

#include "dminner2.h"
#include "informed_modulator.h"
#include "fba2.h"
#include "matrix.h"

namespace libcomm {

/*!
   \brief   Iterative 2D implementation of Davey-MacKay Inner Code.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Implements a 2D version of the Davey-MacKay inner code, using iterative
   row/column decoding, where the sparse symbols are now two-dimensional.
*/

template <class real, bool normalize>
class dminner2d :
   public informed_modulator<bool,libbase::matrix> {
   //private fba2<real,bool,normalize> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>     array1d_t;
   // @}
private:
   // Implementations of channel-specific metrics for fba
   //real R(const int i, const array1b_t& r);
   // Atomic modem operations (private as these should never be used)
   const bool modulate(const int index) const
      { assert("Function should not be used."); return false; };
   const int demodulate(const bool& signal) const
      { assert("Function should not be used."); return 0; };
   const int demodulate(const bool& signal, const libbase::vector<double>& app) const
      { assert("Function should not be used."); return 0; };
protected:
   // Interface with derived classes
   //void advance() const;
   void domodulate(const int N, const libbase::matrix<int>& encoded, libbase::matrix<bool>& tx) {};
   void dodemodulate(const channel<bool>& chan, const libbase::matrix<bool>& rx, libbase::matrix<array1d_t>& ptable) {};
   void dodemodulate(const channel<bool>& chan, const libbase::matrix<bool>& rx, const libbase::matrix<array1d_t>& app, libbase::matrix<array1d_t>& ptable) {};

public:
   
   // Informative functions
   int num_symbols() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dminner2d);
};

}; // end namespace

#endif
