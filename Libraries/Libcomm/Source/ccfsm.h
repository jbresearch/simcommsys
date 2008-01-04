#ifndef __ccfsm_h
#define __ccfsm_h

#include "fsm.h"
#include "matrix.h"
#include "vector.h"

namespace libcomm {

/*!
   \brief   Controller-Canonical Finite State Machine.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (11-14 Dec 2007)
   - Initial version; implements common elements of a controller-canonical fsm,
     where each coefficient is an element in a finite field.
   - Derived from ccbfsm 1.00
   - The finite field is specified as a template parameter.
   - Added internal functions to convert from vector spaces in G to integer form
   - Modified return type for determineinput() to integer.
   - Modified parameter type for output from "const int&" to "int" (as in fsm 1.71)
   - Added function to compute convolution
   - Made convert() and convolve() protected, to make them available to derived classes
   - Modified definition of first convert() so that vector parameter is a const
   - Implemented correct description output

   \version 1.01 (3 Jan 2008)
   - fixed bug in advance() with memoryless systems
*/

template <class G> class ccfsm : public fsm {
protected:
   /*! \name Object representation */
   int k;   //!< Number of inputs (symbols per time-step)
   int n;   //!< Number of outputs (symbols per time-step)
   int nu;  //!< Total number of memory elements (constraint length)
   int m;   //!< Memory order (longest input register)
   libbase::vector< libbase::vector<G> > reg;  //!< Shift registers (one for each input)
   libbase::matrix< libbase::vector<G> > gen;  //!< Generator sequence
   // @}
private:
   /*! \name Internal functions */
   void init(const libbase::matrix< libbase::vector<G> >& generator);
   // @}
protected:
   /*! \name Helper functions */
   int convert(const libbase::vector<G>& x, int y=0) const;
   int convert(int x, libbase::vector<G>& y) const;
   G convolve(const G& s, const libbase::vector<G>& r, const libbase::vector<G>& g) const;
   // @}
   /*! \name FSM helper operations */
   virtual int determineinput(int input) const = 0;
   virtual libbase::vector<G> determinefeedin(int input) const = 0;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   ccfsm() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   ccfsm(const libbase::matrix< libbase::vector<G> >& generator);
   ccfsm(const ccfsm& x);
   ~ccfsm() {};
   // @}

   // FSM state operations (getting and resetting)
   int state() const;
   void reset(int state=0);
   // FSM operations (advance/output/step)
   void advance(int& input);
   int output(int input) const;

   // FSM information functions
   int mem_order() const { return m; };
   int num_states() const { return int(pow(G::elements(),nu)); };
   int num_inputs() const { return int(pow(G::elements(),k)); };
   int num_outputs() const { return int(pow(G::elements(),n)); };

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

}; // end namespace

#endif

