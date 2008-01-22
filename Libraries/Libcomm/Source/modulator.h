#ifndef __modulator_h
#define __modulator_h

#include "config.h"
#include "sigspace.h"
#include "vector.h"
#include "matrix.h"
#include "channel.h"
#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Modulator Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.01 (26 Oct 2001)
   added a virtual destroy function (see interleaver.h)

   \version 1.02 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.
   also changed use of iostream from global to std namespace.

   \version 1.10 (13 Mar 2002)
   added a virtual function which outputs details on the modulator, together with a stream
   << operator too. Also added serialization facility. Created serialize and stream << and
   >> functions to conform with the new serializer protocol, as defined in serializer 1.10.
   The stream << output function first writes the name of the derived class, then calls its
   serialize() to output the data. The name is obtained from the virtual name() function.
   The stream >> input function first gets the name from the stream, then (via
   serialize::call) creates a new object of the appropriate type and calls its serialize()
   function to get the relevant data. Also added cloning function.

   \version 1.20 (17 Mar 2002)
   added functions to modulate a vector of encoded symbols and demodulate a vector of
   sigspace symbols (given a channel model), and new names for the functions that modulate
   or demodulate single elements (the operator[] functions are now just alternative
   notations). Also modified the information-returning functions to return non-const,
   since this won't make any difference.

   \version 1.30 (27 Mar 2002)
   removed the descriptive output() and related stream << output functions, and replaced
   them by a function description() which returns a string. This provides the same
   functionality but in a different format, so that now the only stream << output
   functions are for serialization. This should make the notation much clearer while
   also simplifying description display in objects other than streams.

   \version 1.31 (27 Mar 2002)
   changed the way we store the array of modulation symbols from a heap-allocated
   array to a vector; this significantly simplifies the memory management. Also,
   removed the member variable M, since the number of symbols is easily obtained from
   the vector. Also changed modulate() and operator[] to return sigspace object directly
   not by reference (should have been this way before since the functions are const).

   \version 1.32 (17 Jul 2006)
   in modulate, made an explicit conversion of round's output to int, to conform with the
   changes in itfunc 1.07.

   \version 1.33 (6 Oct 2006)
   modified for compatibility with VS .NET 2005:
   - in energy_function, modified use of pow to avoid ambiguity

   \version 1.40 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.41 (17 Oct 2007)
   changed class to conform with channel 1.52.

   \version 1.50 (25 Oct 2007)
   - extracted functionality of static LUT modulators by creating a new class lut_modulator
   - this class now better reflects its role as a generic base class for the modem layer
   - removed 'const' restriction on modulate and demodulate vector functions, to cater for watermark codes
     (where the random generator is updated during these functions); this also introduces support for other
     time-variant modulation schemes.

   \version 1.51 (30 Nov 2007)
   - added method to get modulation rate.

   \version 1.52 (22 Jan 2008)
   - Removed 'friend' declaration of stream operators.
*/

class modulator {
public:
   /*! \name Constructors / Destructors */
   virtual ~modulator() {};
   // @}
   /*! \name Serialization Support */
   virtual modulator *clone() const = 0;
   virtual const char* name() const = 0;
   // @}

   /*! \name Atomic modem operations */
   /*!
      \brief Modulate a single time-step
      \param   index Index into the symbol alphabet
      \return  Symbol corresponding to the given index
   */
   virtual const sigspace modulate(const int index) const = 0;
   /*!
      \brief Demodulate a single time-step
      \param   signal   Received signal
      \return  Index corresponding symbol that is closest to the received signal
   */
   virtual const int demodulate(const sigspace& signal) const = 0;
   //! \copydoc modulate(index)
   const sigspace operator[](const int index) const { return modulate(index); };
   //! \copydoc demodulate(signal)
   const int operator[](const sigspace& signal) const { return demodulate(signal); };
   // @}

   /*! \name Vector modem operations */
   /*!
      \brief Modulate a sequence of time-steps
      \param[in]  N        The number of possible values of each encoded element
      \param[in]  encoded  Sequence of values to be modulated
      \param[out] tx       Sequence of symbols corresponding to the given input
   */
   virtual void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<sigspace>& tx) = 0;
   /*!
      \brief Demodulate a sequence of time-steps
      \param[in]  chan     The channel model (used to obtain likelihoods)
      \param[in]  rx       Sequence of received symbols
      \param[out] ptable   Table of likelihoods of possible transmitted symbols
      
      \note \c ptable(i,d) \c is the a posteriori probability of having transmitted 
            symbol 'd' at time 'i'
   */
   virtual void demodulate(const channel& chan, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) = 0;
   // @}

   /*! \name Informative functions */
   //! Symbol alphabet size
   virtual int num_symbols() const = 0;
   //! Average energy per symbol
   virtual double energy() const = 0;
   //! Average energy per bit
   double bit_energy() const { return energy()/log2(num_symbols()); };
   //! Modulation rate (spectral efficiency) in bits/unit energy
   double rate() const { return 1.0/bit_energy(); };
   // @}

   /*! \name Description & Serialization */
   //! Object description output
   virtual std::string description() const = 0;
   //! Object serialization ouput
   virtual std::ostream& serialize(std::ostream& sout) const = 0;
   //! Object serialization input
   virtual std::istream& serialize(std::istream& sin) = 0;
   // @}
};

/*! \name Serialization */
std::ostream& operator<<(std::ostream& sout, const modulator* x);
std::istream& operator>>(std::istream& sin, modulator*& x);
// @}

}; // end namespace

#endif
