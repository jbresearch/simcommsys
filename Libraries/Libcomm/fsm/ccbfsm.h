#ifndef __ccbfsm_h
#define __ccbfsm_h

#include "fsm.h"
#include "bitfield.h"
#include "matrix.h"
#include "vector.h"

namespace libcomm {

/*!
 * \brief   Controller-Canonical Binary Finite State Machine.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * Implements common elements of a controller-canonical binary fsm.
 * The generator matrix is serialized (and output in the descriptor) using the
 * usual convention as in Lin & Costello (ie. high-order bits are left-most,
 * and closest to the input junction). However, the state internally is held
 * such that the least-significant bit is left-most; this is according to the
 * convention described by Alex, such that from state 0, input 1, we always
 * get state 1, no matter how long the state register is. This difference in
 * notation requires the generator matrix to be reversed for internal use.
 * We do this reversal during initialization.
 */

class ccbfsm : public fsm {
protected:
   /*! \name Object representation */
   int k; //!< Number of input bits
   int n; //!< Number of output bits
   int nu; //!< Number of memory elements (constraint length)
   int m; //!< Memory order (longest input register)
   /*! \brief Shift registers (one for each input bit);
    * not all registers need be the same length
    */
   libbase::vector<libbase::bitfield> reg;
   libbase::matrix<libbase::bitfield> gen; //!< Generator sequence
   //! Generator sequence, in reversed form (as used internally)
   libbase::matrix<libbase::bitfield> revgen;
   // @}
private:
   /*! \name Internal functions */
   void init();
   // @}
protected:
   /*! \name FSM helper operations */
   //! Computes the actual input to be applied (works out tail)
   virtual libbase::vector<int>
   determineinput(libbase::vector<int> input) const = 0;
   //! Computes the memory register input
   virtual libbase::bitfield
   determinefeedin(libbase::vector<int> input) const = 0;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   ccbfsm()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   ccbfsm(const libbase::matrix<libbase::bitfield>& generator);
   ~ccbfsm()
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   libbase::vector<int> state() const;
   void reset();
   void reset(libbase::vector<int> state);
   // FSM operations (advance/output/step)
   void advance(libbase::vector<int>& input);
   libbase::vector<int> output(libbase::vector<int> input) const;

   // FSM information functions
   int mem_order() const
      {
      return m;
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

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

} // end namespace

#endif

