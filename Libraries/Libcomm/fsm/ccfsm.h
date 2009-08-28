#ifndef __ccfsm_h
#define __ccfsm_h

#include "fsm.h"
#include "matrix.h"
#include "vector.h"

namespace libcomm {

/*!
 * \brief   Controller-Canonical Finite State Machine.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Implements common elements of a controller-canonical fsm, where each
 * coefficient is an element in a finite field.
 * The finite field is specified as a template parameter.
 */

template <class G>
class ccfsm : public fsm {
protected:
   /*! \name Object representation */
   int k; //!< Number of input lines
   int n; //!< Number of output lines
   int nu; //!< Total number of memory elements (constraint length)
   int m; //!< Memory order (longest input register)
   libbase::vector<libbase::vector<G> > reg; //!< Shift registers (one for each input)
   libbase::matrix<libbase::vector<G> > gen; //!< Generator sequence
   // @}
private:
   /*! \name Internal functions */
   void init(const libbase::matrix<libbase::vector<G> >& generator);
   // @}
protected:
   /*! \name Helper functions */
   static int convert(const libbase::vector<G>& vec);
   static libbase::vector<G> convert(int val, int nu);
   G convolve(const G& s, const libbase::vector<G>& r,
         const libbase::vector<G>& g) const;
   // @}
   /*! \name FSM helper operations */
   /*!
    * \brief Determine the actual input that will be applied (resolve tail)
    * \param input Requested input - can be a valid input or the 'tail' value
    * \return The given value, or the value that must be applied to tail out
    */
   virtual libbase::vector<int>
   determineinput(libbase::vector<int> input) const = 0;
   /*!
    * \brief Determine the value that will be shifted into the register
    * \param input Requested input - can only be a valid input
    * \return Vector representation of the shift-in value
    */
   virtual libbase::vector<G>
   determinefeedin(libbase::vector<int> input) const = 0;
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   ccfsm()
      {
      }
   // @}
public:
   /*! \name Constructors / Destructors */
   ccfsm(const libbase::matrix<libbase::vector<G> >& generator);
   ccfsm(const ccfsm& x);
   ~ccfsm()
      {
      }
   // @}

   // FSM state operations (getting and resetting)
   libbase::vector<int> state() const;
   void reset();
   void reset(libbase::vector<int> state);
   // FSM operations (advance/output/step)
   void advance(libbase::vector<int> & input);
   libbase::vector<int> output(libbase::vector<int> input) const;

   // FSM information functions
   int mem_order() const
      {
      return m;
      }
   int mem_elements() const
      {
      return nu;
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
      return G::elements();
      }

   // Description & Serialization
   std::string description() const;
   std::ostream& serialize(std::ostream& sout) const;
   std::istream& serialize(std::istream& sin);
};

} // end namespace

#endif
