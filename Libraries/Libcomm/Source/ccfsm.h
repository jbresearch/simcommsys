#ifndef __ccfsm_h
#define __ccfsm_h

#include "vcs.h"
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

   \version 1.00 (11 Dec 2007)
   - Initial version; implements common elements of a controller-canonical fsm,
     where each coefficient is an element in a finite field.
   - The finite field is specified as a template parameter.
*/

template <class G> class ccfsm : public fsm {
protected:
   /*! \name Object representation */
   int k;   //!< Number of inputs (symbols per time-step)
   int n;   //!< Number of outputs (symbols per time-step)
   int nu;  //!< Total number of memory elements (constraint length)
   int m;   //!< Memory order (longest input register)
   libbase::vector<libbase::vector<G>> reg;  //!< Shift registers (one for each input)
   libbase::matrix<libbase::vector<G>> gen;  //!< Generator sequence
   // @}
private:
   /*! \name Internal functions */
   void init(const libbase::matrix<libbase::vector<G>>& generator);
   // @}
protected:
   /*! \name Implementation-dependent functions */
   virtual libbase::vector<G> determineinput(const int input) const = 0;
   virtual libbase::vector<G> determinefeedin(const int input) const = 0;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   ccfsm() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   ccfsm(const libbase::matrix<libbase::vector<G>>& generator);
   ccfsm(const ccfsm& x);
   ~ccfsm() {};
   // @}
   
   /*! \name FSM state operations (getting and resetting) */
   int state() const;
   void reset(int state=0);
   // @}
   /*! \name FSM operations (advance/output/step) */
   void advance(int& input);
   int output(const int& input) const;
   // @}

   /*! \name FSM information functions */
   //! Memory order (length of tail)
   int mem_order() const { return m; };
   //! Number of defined states
   int num_states() const { return G::elements()<<(nu-1); };
   //! Number of valid input combinations
   int num_inputs() const { return G::elements()<<(k-1); };
   //! Number of valid output combinations
   int num_outputs() const { return G::elements()<<(n-1); };
   // @}

   /*! \name Description & Serialization */
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
   // @}
};

}; // end namespace

#endif

