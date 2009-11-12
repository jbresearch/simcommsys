#ifndef __grscc_h
#define __grscc_h

#include "ccfsm.h"
#include "serializer.h"

namespace libcomm {

/*!
 * \brief   Generalized Recursive Systematic Convolutional Code.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements RSCC where polynomial coefficients are elements of a finite
 * field, which is specified as a template parameter.
 */

template <class G>
class grscc : public ccfsm<G> {
private:
   /*! \name Object representation */
   libbase::matrix<int> csct; //!< Circulation state correspondence table
   // @}
   /*! \name Internal functions */
   libbase::matrix<G> getstategen() const;
   // TODO: Separate circulation state stuff from this class
   // (not all RSC codes are suitable)
   void initcsct();
   // @}
protected:
   /*! \name FSM helper operations */
   libbase::vector<int> determineinput(const libbase::vector<int>& input) const;
   libbase::vector<G> determinefeedin(const libbase::vector<int>& input) const;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   grscc()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   grscc(const libbase::matrix<libbase::vector<G> >& generator) :
      ccfsm<G> (generator)
      {
      initcsct();
      }
   // @}

   // FSM state operations (getting and resetting)
   void resetcircular(const libbase::vector<int>& zerostate, int n);

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(grscc)
};

} // end namespace

#endif

