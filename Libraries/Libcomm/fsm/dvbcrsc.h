#ifndef __dvbcrsc_h
#define __dvbcrsc_h

#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   DVB-Standard Circular Recursive Systematic Convolutional Coder.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 */

class dvbcrsc : public fsm {
   /*! \name Object representation */
   static const int csct[7][8]; //!< Circulation state correspondence table
   static const int k, n; //!< Number of input and output bits, respectively
   static const int nu; //!< Number of memory elements (constraint length)
   libbase::bitfield reg; //!< Present state (shift register)
   // @}
protected:
   /*! \name Internal functions */
   void init();
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   dvbcrsc()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   dvbcrsc(const dvbcrsc& x);
   ~dvbcrsc()
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   libbase::vector<int> state() const;
   void reset();
   void reset(libbase::vector<int> state);
   void resetcircular(libbase::vector<int> zerostate, int n);
   // FSM operations (advance/output/step)
   void advance(libbase::vector<int>& input);
   libbase::vector<int> output(libbase::vector<int> input) const;

   // FSM information functions
   int mem_order() const
      {
      return nu;
      }
   int num_states() const
      {
      return 1 << nu;
      }
   int num_inputs() const
      {
      return k;
      }
   int num_outputs() const
      {
      return n;
      }
   int num_symbols() const
      {
      return 2;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dvbcrsc);
};

} // end namespace

#endif

