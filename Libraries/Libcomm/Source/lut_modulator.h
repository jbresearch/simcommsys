#ifndef __lut_modulator_h
#define __lut_modulator_h

#include "modulator.h"

namespace libcomm {

/*!
   \brief   LUT Modulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

class lut_modulator : public modulator<sigspace> {
protected:
   libbase::vector<sigspace> lut;   // Array of modulation symbols

protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx);
   void dodemodulate(const channel<sigspace>& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable);

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~lut_modulator() {};
   // @}

   // Atomic modem operations
   const sigspace modulate(const int index) const { return lut(index); };
   const int demodulate(const sigspace& signal) const;

   // Vector modem operations
   // (necessary because base is templated)
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx)
      { modulator<sigspace>::modulate(N, encoded, tx); };
   void demodulate(const channel<sigspace>& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable)
      { modulator<sigspace>::demodulate(chan, rx, ptable); };

   // Informative functions
   int num_symbols() const { return lut.size(); };
   double energy() const;  // average energy per symbol
};

}; // end namespace

#endif
