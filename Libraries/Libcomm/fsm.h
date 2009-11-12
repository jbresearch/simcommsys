#ifndef __fsm_h
#define __fsm_h

#include "config.h"
#include "serializer.h"
#include "vector.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
 * \brief   Finite State Machine.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * \todo separate support for circulation from this class
 */

class fsm : public libbase::serializable {
public:
   /*! \name Class constants */
   static const int tail; //!< A special input value to use when tailing out
   // @}

protected:
   /*! \name Object representation */
   int N; //!< Sequence length since last reset;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~fsm()
      {
      }
   // @}

   /*! \name Helper functions */
   static int convert(const libbase::vector<int>& vec, int S);
   static libbase::vector<int> convert(int val, int nu, int S);
   //! Convert input from vector to integer
   int convert_input(const libbase::vector<int>& vec)
      {
      assert(vec.size() == num_inputs());
      return fsm::convert(vec, num_symbols());
      }
   //! Convert input from integer to vector
   libbase::vector<int> convert_input(int val)
      {
      return fsm::convert(val, num_inputs(), num_symbols());
      }
   //! Convert output from vector to integer
   int convert_output(const libbase::vector<int>& vec)
      {
      assert(vec.size() == num_outputs());
      return fsm::convert(vec, num_symbols());
      }
   //! Convert output from integer to vector
   libbase::vector<int> convert_output(int val)
      {
      return fsm::convert(val, num_outputs(), num_symbols());
      }
   //! Convert state from vector to integer
   int convert_state(const libbase::vector<int>& vec)
      {
      assert(vec.size() == mem_elements());
      return fsm::convert(vec, num_symbols());
      }
   //! Convert state from integer to vector
   libbase::vector<int> convert_state(int val)
      {
      return fsm::convert(val, mem_elements(), num_symbols());
      }
   // @}

   /*! \name FSM state operations (getting and resetting) */
   /*!
    * \brief The current state
    * \return A vector representation of the current state
    * \invariant The state value should always be between 0 and num_states()-1
    *
    * By convention, lower index positions correspond to memory elements
    * nearer to the input side, and registers for lower-index inputs are
    * placed first
    *
    * Also, lower-order inputs get lower-order positions within the state
    * representation. Also, left-most memory elements (ie. those closest to
    * the input junction) are represented by lower index positions within
    * the state representation.
    *
    * Define the state vector as a column vector, as follows:
    * \f[ S_i = \begin{pmatrix}
    * S_{1,1} \\ S_{1,2} \\ \vdots \\ S_{1,k} \\
    * S_{2,1} \\ S_{2,2} \\ \vdots \\ S_{2,k} \\
    * \vdots \\ S_{\nu_k,k}
    * \end{pmatrix} \f]
    *
    * where \f$ k \f$ is the number of input lines and \f$ \nu_i \f$ is the
    * number of memory elements for input \f$ i \f$.
    * Element \f$ S_{1,i} \f$ is the left-most memory element for input
    * \f$ i \f$, and therefore the one to which the shift-in is applied.
    * It can be seen that the total length of the state vector is equal to
    * the total number of memory elements in the system, \f$ \nu \f$.
    */
   virtual libbase::vector<int> state() const = 0;
   /*!
    * \brief Reset to the 'zero' state
    *
    * \note This function has to be called once by each function re-implementing
    * it.
    */
   virtual void reset();
   /*!
    * \brief Reset to a specified state
    * \param state A vector representation of the state we want to set to
    *
    * \see state()
    *
    * \note This function has to be called once by each function re-implementing
    * it.
    */
   virtual void reset(const libbase::vector<int>& state);
   /*!
    * \brief Reset to the circulation state
    *
    * \param zerostate The final state for the input sequence, if we start at
    * the zero-state
    *
    * \param n The number of time-steps in the input sequence
    *
    * This method performs the initial state computation (and setting) for
    * circular encoding. It is assumed that this will be called after a
    * sequence of the form [reset(); loop step()/advance()] which the calling
    * class uses to determine the zero-state solution.
    */
   virtual void resetcircular(const libbase::vector<int>& zerostate, int n) = 0;
   /*!
    * \brief Reset to the circulation state, assuming we have just run through
    * the input sequence, starting with the zero-state
    *
    * This is a convenient form of the earlier method, where fsm keeps track
    * of the number of time-steps since the last reset operation as well as
    * the final state value.
    *
    * The calling class must ensure that this is consistent with the
    * requirements - that is, the initial reset must be to state zero and the
    * input sequence given since the last reset must be the same as the one
    * that will be used now.
    */
   void resetcircular();
   // @}

   /*! \name FSM operations (advance/output/step) */
   /*!
    * \brief Feeds the specified input and advances the state
    *
    * \param[in,out] input Vector representation of current input; if these
    * are the 'tail' value, they will be updated
    *
    * This method performs the state-change without also computing the output;
    * it is provided as a faster version of step(), for when the output doesn't
    * need to be computed.
    *
    * By convention, lower index positions correspond to lower-index inputs
    *
    * \note This function has to be called once by each function re-implementing
    * it.
    */
   virtual void advance(libbase::vector<int>& input);
   /*!
    * \brief Computes the output for the given input and the present state
    *
    * \param  input Vector representation of current input; may be the 'tail'
    * value.
    *
    * By convention, lower index positions correspond to lower-index inputs
    * and outputs.
    *
    * \return Vector representation of the output
    */
   virtual libbase::vector<int>
         output(const libbase::vector<int>& input) const = 0;
   /*!
    * \brief Feeds the specified input and returns the corresponding output,
    * advancing the state in the process
    *
    * \param[in,out] input Vector representation of current input; if these
    * are the 'tail' value, they will be updated
    *
    * \return Vector representation of the output
    *
    * \note Equivalent to output() followed by advance()
    */
   libbase::vector<int> step(libbase::vector<int>& input);
   // @}

   /*! \name FSM information functions - fundamental */
   //! Memory order (length of tail)
   virtual int mem_order() const = 0;
   //! Number of memory elements
   virtual int mem_elements() const = 0;
   //! Number of input lines
   virtual int num_inputs() const = 0;
   //! Number of output lines
   virtual int num_outputs() const = 0;
   //! Alphabet size of input/output symbols
   virtual int num_symbols() const = 0;
   // @}

   /*! \name FSM information functions - derived */
   //! Number of defined states
   int num_states() const
      {
      return int(pow(num_symbols(), mem_elements()));
      }
   //! Number of input combinations
   int num_input_combinations() const
      {
      return int(pow(num_symbols(), num_inputs()));
      }
   //! Number of output combinations
   int num_output_combinations() const
      {
      return int(pow(num_symbols(), num_outputs()));
      }
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(fsm)
};

} // end namespace

#endif

