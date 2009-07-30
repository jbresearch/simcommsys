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
   int state() const;
   void reset(int state = 0);
   void resetcircular(int zerostate, int n);
   // FSM operations (advance/output/step)
   void advance(int& input);
   int output(int input) const;

   // FSM information functions
   int num_states() const
      {
      return 1 << nu;
      }
   int num_inputs() const
      {
      return 1 << k;
      }
   int num_outputs() const
      {
      return 1 << n;
      }
   int mem_order() const
      {
      return nu;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(dvbcrsc);
};

} // end namespace

#endif

