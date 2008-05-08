#ifndef __grscc_h
#define __grscc_h

#include "ccfsm.h"
#include "serializer.h"

namespace libcomm {

/*!
   \brief   Generalized Recursive Systematic Convolutional Code.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (13 Dec 2007)
   - Initial version; implements RSCC where polynomial coefficients are elements
     of a finite field.
   - Derived from rscc 1.70
   - The finite field is specified as a template parameter.

   \version 1.01 (4-6 Jan 2008)
   - removed serialization functions, which were redundant
   - removed resetcircular(), which is now implemented in fsm()
   - implemented getstategen() which returns the state-generator matrix, and
     getstatevec() which returns the state-vector, both in the format required
     for computing the circulation state
   - added circulation state correspondence table, and implemented initcsct()
     which initializes it
   - implemented resetcircular() using csct
*/

template <class G> class grscc : public ccfsm<G> {
private:
   /*! \name Object representation */
   libbase::matrix<int> csct; //!< Circulation state correspondence table
   // @}
   /*! \name Internal functions */
   int getstateval(const libbase::vector<G>& statevec) const;
   libbase::vector<G> getstatevec(int stateval) const;
   libbase::matrix<G> getstategen() const;
   void initcsct();
   // @}
protected:
   /*! \name FSM helper operations */
   int determineinput(int input) const;
   libbase::vector<G> determinefeedin(int input) const;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   grscc() {};
   // @}
public:
   /*! \name Constructors / Destructors */
   grscc(const libbase::matrix< libbase::vector<G> >& generator) : ccfsm<G>(generator) {};
   grscc(const grscc<G>& x) : ccfsm<G>(x) {};
   ~grscc() {};
   // @}

   // FSM state operations (getting and resetting)
   void resetcircular(int zerostate, int n);

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(grscc)
};

}; // end namespace

#endif

