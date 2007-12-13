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
   - Derived from grscc 1.70
   - The finite field is specified as a template parameter.
*/

template <class G> class grscc : public ccfsm<G> {
   /*! \name Serialization */
   static const libbase::serializer shelper;
   static void* create() { return new grscc<G>; };
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
   grscc(const libbase::matrix< libbase::vector<G> >& generator);
   grscc(const grscc<G>& x);
   ~grscc() {};
   // @}
   
   /*! \name Class management (cloning/naming) */
   grscc *clone() const { return new grscc<G>(*this); };
   const char* name() const { return shelper.name(); };
   // @}

   /*! \name FSM state operations (getting and resetting) */
   void resetcircular(int zerostate, int n);
   void resetcircular();
   // @}

   /*! \name Description & Serialization */
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
   // @}
};

}; // end namespace

#endif

