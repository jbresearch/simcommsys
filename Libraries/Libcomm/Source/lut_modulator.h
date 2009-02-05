#ifndef __lut_modulator_h
#define __lut_modulator_h

#include "blockmodem.h"

namespace libcomm {

/*!
   \brief   LUT Modulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Test inheritance of virtual functions in VS 2005
*/

class lut_modulator : public blockmodem<sigspace> {
protected:
   libbase::vector<sigspace> lut;   // Array of modulation symbols

protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx);
   void dodemodulate(const channel<sigspace>& chan, const libbase::vector<sigspace>& rx, libbase::vector< libbase::vector<double> >& ptable);

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~lut_modulator() {};
   // @}

   // Atomic modem operations
   const sigspace modulate(const int index) const { return lut(index); };
   const int demodulate(const sigspace& signal) const;

   // Vector modem operations
   // (necessary because inheriting methods from templated base)
   using blockmodem<sigspace>::modulate;
   using blockmodem<sigspace>::demodulate;

   // Informative functions
   int num_symbols() const { return lut.size(); };
   double energy() const;  // average energy per symbol
};

}; // end namespace

#endif
