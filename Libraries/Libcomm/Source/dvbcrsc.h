#ifndef __dvbcrsc_h
#define __dvbcrsc_h

#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   DVB-Standard Circular Recursive Systematic Convolutional Coder.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (13-14 Jul 2006)
   original version, made to conform with fsm 1.50.

   \version 1.10 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.11 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]

   \version 1.20 (3-4 Dec 2007)
   - updated output() as per fsm 1.70
   - removed implementation of step() in favor of the default provided by fsm

   \version 1.21 (13 Dec 2007)
   - modified parameter type for output from "const int&" to "int" (as in fsm 1.71)

   \version 1.22 (4 Jan 2008)
   - removed resetcircular(), which is now implemented in fsm()
   - consequently, also removed N and related code in reset() and advance()
   - added calls to underlying functions in reset() and advance()
*/

class dvbcrsc : public fsm {
   /*! \name Object representation */
   static const int csct[7][8];  //!< Circulation state correspondence table
   static const int k, n;        //!< Number of input and output bits, respectively
   static const int nu;          //!< Number of memory elements (constraint length)
   libbase::bitfield reg;        //!< Present state (shift register)
   // @}
protected:
   /*! \name Internal functions */
   void init();
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   dvbcrsc() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   dvbcrsc(const dvbcrsc& x);
   ~dvbcrsc() {};
   // @}

   // FSM state operations (getting and resetting)
   int state() const;
   void reset(int state=0);
   void resetcircular(int zerostate, int n);
   // FSM operations (advance/output/step)
   void advance(int& input);
   int output(int input) const;

   // FSM information functions
   int num_states() const { return 1<<nu; };
   int num_inputs() const { return 1<<k; };
   int num_outputs() const { return 1<<n; };
   int mem_order() const { return nu; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(dvbcrsc)
};

}; // end namespace

#endif

