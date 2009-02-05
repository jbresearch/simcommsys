#ifndef __fsm_h
#define __fsm_h

#include "config.h"
#include "serializer.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Finite State Machine.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

*/

class fsm {
public:
   /*! \name Object representation */
   static const int tail;        //!< A special input value to use when tailing out
   int N;                        //!< Sequence length since last reset;
   // @}

   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~fsm() {};
   // @}

   /*! \name FSM state operations (getting and resetting) */
   /*!
      \brief The current state
      \return A unique integer representation of the current state
      \invariant The state value should always be between 0 and num_states()-1
   */
   virtual int state() const = 0;
   /*!
      \brief Reset to a specified state
      \param state  A unique integer representation of the state we want to set to
      \invariant The state value should always be between 0 and num_states()-1
      \see state()
      \note This function has to be called once by each function re-implementing it.
   */
   virtual void reset(int state=0);
   /*!
      \brief Reset to the circulation state
      \param zerostate  The final state for the input sequence, if we start at the zero-state
      \param n  The number of time-steps in the input sequence

      This method performs the initial state computation (and setting) for
      circular encoding. It is assumed that this will be called after a
      sequence of the form [reset(); loop step()/advance()] which the calling
      class uses to determine the zero-state solution.

      Consider a convolutional code where the state \f$ S_i \f$ at timestep \f$ i \f$ is
      related to state \f$ S_{i-1} \f$ and input \f$ X_i \f$ by the relation:
      \f[ S_i = G \cdot S_{i-1} + X_i \f]

      Therefore, after \f$ N \f$ timesteps, the state is given by:
      \f[ S_N = G^N \cdot S_0 + \sum_{i=1}^{N} G^{N-i} \cdot X_i \f]

      Thus, the circulation state, defined such that \f$ S_c = S_N = S_0 \f$ is
      derived from the equation:
      \f[ S_c = \langle I + G^N \rangle ^{-1} \sum_{i=1}^{N} G^{N-i} \cdot X_i \f]

      and is obtainable only if \f$ I + G^N \f$ is invertible. It is worth noting
      that not all \f$ G \f$ matrices are suitable; also, the sequence length \f$ N \f$
      must not be a multiple of the period \f$ L \f$ of the recursive generator, defined
      by \f$ G^L = I \f$.

      Consider starting at the zero-intial-state and pre-encoding the input sequence;
      this gives us a final state:
      \f[ S_N^0 = \sum_{i=1}^{N} G^{N-i} \cdot X_i \f]

      Combining this with the equation for the circulation state, we get:
      \f[ S_c = \langle I + G^N \rangle ^{-1} S_N^0 \f]

      Note, however, that because of the periodicity of the system, this equation
      can be reduced to:
      \f[ S_c = \langle I + G^P \rangle ^{-1} S_N^0 \f]

      where \f$ P = N \mathrm{mod} L \f$. This can be obtained by a lookup table containing
      all combinations of \f$ P \f$ and \f$ S_N^0 \f$.
   */
   virtual void resetcircular(int zerostate, int n) = 0;
   /*!
      \brief Reset to the circulation state, assuming we have just run through the
             input sequence, starting with the zero-state

      This is a convenient form of the earlier method, where the fsm-derived
      class must keep track of the number of time-steps since the last reset
      operation as well as the final state value.

      The calling class must ensure that this is consistent with the
      requirements - that is, the initial reset must be to state zero and the
      input sequence given since the last reset must be the same as the one
      that will be used now.
   */
   void resetcircular();
   // @}

   /*! \name FSM operations (advance/output/step) */
   /*!
      \brief Feeds the specified input and advances the state
      \param[in,out]   input    Integer representation of current input; if this is the
                                'tail' value, it will be updated
      This method performs the state-change without also computing the output;
      it is provided as a faster version of step(), for when the output doesn't
      need to be computed.

      \note This function has to be called once by each function re-implementing it.
   */
   virtual void advance(int& input);
   /*!
      \brief Computes the output for the given input and the present state
      \param  input    Integer representation of current input; may be the 'tail' value
      \return Integer representation of the output
   */
   virtual int output(int input) const = 0;
   /*!
      \brief Feeds the specified input and returns the corresponding output,
             advancing the state in the process
      \param[in,out]   input    Integer representation of current input; if this is the
                                'tail' value, it will be updated
      \return Integer representation of the output
      \note Equivalent to output() followed by advance()
   */
   int step(int& input);
   // @}

   /*! \name FSM information functions */
   //! Memory order (length of tail)
   virtual int mem_order() const = 0;
   //! Number of defined states
   virtual int num_states() const = 0;
   //! Number of valid input combinations
   virtual int num_inputs() const = 0;
   //! Number of valid output combinations
   virtual int num_outputs() const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(fsm);
};

}; // end namespace

#endif

