#ifndef __fsm_h
#define __fsm_h

#include "config.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Finite State Machine.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.10 (4 Nov 2001)
   added a virtual function which outputs details on the finite state machine (this was
   only done before in a non-standard print routine). Added a stream << operator too.

   \version 1.20 (28 Feb 2002)
   added serialization facility

   \version 1.21 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.
   also changed use of iostream from global to std namespace.

   \version 1.22 (11 Mar 2002)
   changed the definition of the cloning operation to be a const member.

   \version 1.30 (11 Mar 2002)
   changed the stream << and >> functions to conform with the new serializer protocol,
   as defined in serializer 1.10. The stream << output function first writes the name
   of the derived class, then calls its serialize() to output the data. The name is
   obtained from the virtual name() function. The stream >> input function first gets
   the name from the stream, then (via serialize::call) creates a new object of the
   appropriate type and calls its serialize() function to get the relevant data. Also,
   changed the definition of stream << output to take the pointer to the fsm class
   directly, not by reference.

   \version 1.40 (27 Mar 2002)
   removed the descriptive output() and related stream << output functions, and replaced
   them by a function description() which returns a string. This provides the same
   functionality but in a different format, so that now the only stream << output
   functions are for serialization. This should make the notation much clearer while
   also simplifying description display in objects other than streams.

   \version 1.50 (8 Jan 2006)
   updated class to allow consistent support for circular trellis encoding (tail-biting)
   and also for the use of rate m/(m+1) LFSR codes. In practice this has been achieved
   by creating a generalized convolutional code class that uses state-space techniques
   to represent the system - the required codes, such as rate m/(m+1) LFSR including
   the DVB duo-binary codes can be easily derived. The following virtual functions
   were added to allow this:
   - advance(input) - this performs the state-change without also computing the output;
     it is provided as a faster version of step(), since the output doesn't need to be
     computed in the first iteration.
   - output(input) - this is added for completeness, and calculates the output, given
     the present state and input.
   - resetcircular(zerostate, N) - this performs the initial state computation (and
     setting) for circular encoding. It is assumed that this will be called after a
     sequence of the form [reset(); loop step()/advance()] which the calling class
     uses to determine the zero-state solution. N is the number of time-steps involved.
   - resetcircular() - is a convenient form of the above function, where the fsm-derived
     class must keep track of the number of time-steps since the last reset operation
     as well as the final state value. The calling class must ensure that this is
     consistent with the requirements - that is, the initial reset must be to state zero
     and the input sequence given since the last reset must be the same as the one that
     will be used now.
   It was elected to make all the above functions pure virtual - this requires that all
   derived classes must be updates accordingly; it is considered advantageous because
   there are just two derived classes which can therefore be easily updated. This avoids
   adding dependance complexity in the fsm class (there is less risk of unexpected
   behaviour).

   \version 1.60 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.70 (3-5 Dec 2007)
   - Updated output() so that the input value is a const and the function is also a const
   - provided a default implementation of step() using output() and advance()
   - cleaned up order of members and documentation
   - removed friend status of stream output operators

   \version 1.71 (13 Dec 2007)
   - modified parameter type for output from "const int&" to "int"

   \version 1.80 (4 Jan 2008)
   - made step() non-virtual since we don't want to re-implement it elsewhere
   - made resetcircular() non-virtual since we don't want to re-implement it elsewhere
   - implemented resetcircular() here; this required the addition of member N and related
     code in reset() and advance(). This also means that reset() and advance() now have
     to be called by each function re-implementing them.
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

   /*! \name Serialization Support */
   //! Cloning operation
   virtual fsm *clone() const = 0;
   //! Derived object's name
   virtual const char* name() const = 0;
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
      \cf state()
      \note This function has to be called once by each function re-implementing it.
   */
   virtual void reset(int state=0);
   /*!
      \brief Reset to the circulation state
      \param zerostate  The final state for the input sequence, if we start at the zero-state
      \param n  The number of time-steps in the input sequence

      Consider a convolutional code where the state \f$ S_i \f$ at timestep \f$ i \f$ is
      related to state \f$ S_{i-1} \f$ and input \f$ X_i \f$ by the relation:
      \f[ S_i = G \mdot S_{i-1} + X_i \f]

      Therefore, after \f$ N \f$ timesteps, the state is given by:
      \f[ S_N = G^N \mdot S_0 + \sum_{i=1}^{N} G^{N-i} \mdot X_i \f]

      Thus, the circulation state, defined such that \f$ S_c = S_N = S_0 \f$ is
      derived from the equation:
      \f[ S_c = \langle I + G^N \rangle ^{-1} \sum_{i=1}^{N} G^{N-i} \mdot X_i \f]

      and is obtainable only if \f$ I + G^N \f$ is invertible. It is worth noting
      that not all \f$ G \f$ matrices are suitable; also, the sequence length \f$ N \f$
      must not be a multiple of the period \f$ L \f$ of the recursive generator, defined
      by \f$ G^L = I \f$.

      Consider starting at the zero-intial-state and pre-encoding the input sequence;
      this gives us a final state:
      \f[ S_N^0 = \sum_{i=1}^{N} G^{N-i} \mdot X_i \f]

      Combining this with the equation for the circulation state, we get:
      \f[ S_c = \langle I + G^N \rangle ^{-1} S_N^0 \f]

      Note, however, that because of the periodicity of the system, this equation
      can be reduced to:
      \f[ S_c = \langle I + G^P \rangle ^{-1} S_N^0 \f]

      where \f$ P = N \mod L \f$. This can be obtained by a lookup table containing
      all combinations of \f$ P \f$ and \f$ S_N^0 \f$.
   */
   virtual void resetcircular(int zerostate, int n) = 0;
   /*!
      \brief Reset to the circulation state, assuming we have just run through the
             input sequence, starting with the zero-state 
   */
   void resetcircular();
   // @}

   /*! \name FSM operations (advance/output/step) */
   /*!
      \brief Feeds the specified input and advances the state
      \param[in,out]   input    Integer representation of current input; if this is the
                                'tail' value, it will be updated
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

   /*! \name Description & Serialization */
   //! Description output
   virtual std::string description() const = 0;
   //! Serialization output
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   //! Serialization input
   virtual std::istream& serialize(std::istream& sin) = 0;
   // @}
};

/*! \name Stream Output */
std::ostream& operator<<(std::ostream& sout, const fsm* x);
std::istream& operator>>(std::istream& sin, fsm*& x);
// @}

}; // end namespace

#endif

