#ifndef __ccbfsm_h
#define __ccbfsm_h

#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "vector.h"

namespace libcomm {

/*!
   \brief   Controller-Canonical Binary Finite State Machine.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Implements common elements of a controller-canonical binary fsm.
*/

class ccbfsm : public fsm {
protected:
   /*! \name Object representation */
   int k;   //!< Number of input bits
   int n;   //!< Number of output bits
   int nu;  //!< Number of memory elements (constraint length)
   int m;   //!< Memory order (longest input register)
   libbase::vector<libbase::bitfield> reg;   //!< Shift registers (one for each input bit)
   libbase::matrix<libbase::bitfield> gen;   //!< Generator sequence
   // @}
private:
   /*! \name Internal functions */
   void init(const libbase::matrix<libbase::bitfield>& generator);
   // @}
protected:
   /*! \name FSM helper operations */
   virtual libbase::bitfield determineinput(const int input) const = 0;
   virtual libbase::bitfield determinefeedin(const int input) const = 0;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   ccbfsm() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   ccbfsm(const libbase::matrix<libbase::bitfield>& generator);
   ccbfsm(const ccbfsm& x);
   ~ccbfsm() {};
   // @}

   // FSM state operations (getting and resetting)
   int state() const;
   void reset(int state=0);
   // FSM operations (advance/output/step)
   void advance(int& input);
   int output(int input) const;

   // FSM information functions
   int mem_order() const { return m; };
   int num_states() const { return 1<<nu; };
   int num_inputs() const { return 1<<k; };
   int num_outputs() const { return 1<<n; };

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

