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
   \brief   Common Modulator Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (24 Jan 2008)
   - Contains common interface for modulator classes.
*/

template <class S> class basic_modulator {
public:
   /*! \name Constructors / Destructors */
   virtual ~basic_modulator() {};
   // @}
   /*! \name Serialization Support */
   virtual basic_modulator *clone() const = 0;
   virtual const char* name() const = 0;
   // @}

   /*! \name Atomic modem operations */
   /*!
      \brief Modulate a single time-step
      \param   index Index into the symbol alphabet
      \return  Symbol corresponding to the given index
   */
   virtual const S modulate(const int index) const = 0;
   /*!
      \brief Demodulate a single time-step
      \param   signal   Received signal
      \return  Index corresponding symbol that is closest to the received signal
   */
   virtual const int demodulate(const S& signal) const = 0;
   /*! \copydoc modulate() */
   const S operator[](const int index) const { return modulate(index); };
   /*! \copydoc demodulate() */
   const int operator[](const S& signal) const { return demodulate(signal); };
   // @}

   /*! \name Vector modem operations */
   /*!
      \brief Modulate a sequence of time-steps
      \param[in]  N        The number of possible values of each encoded element
      \param[in]  encoded  Sequence of values to be modulated
      \param[out] tx       Sequence of symbols corresponding to the given input

      \todo Remove parameter N, replacing 'int' type for encoded vector with something
            that also encodes the number of symbols in the alphabet
   */
   virtual void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<S>& tx) = 0;
   /*!
      \brief Demodulate a sequence of time-steps
      \param[in]  chan     The channel model (used to obtain likelihoods)
      \param[in]  rx       Sequence of received symbols
      \param[out] ptable   Table of likelihoods of possible transmitted symbols
      
      \note \c ptable(i,d) \c is the a posteriori probability of having transmitted 
            symbol 'd' at time 'i'
   */
   virtual void demodulate(const channel<S>& chan, const libbase::vector<S>& rx, libbase::matrix<double>& ptable) = 0;
   // @}

   /*! \name Informative functions */
   //! Symbol alphabet size
   virtual int num_symbols() const = 0;
   // @}

   /*! \name Description & Serialization */
   //! Object description output
   virtual std::string description() const = 0;
   //! Object serialization ouput
   virtual std::ostream& serialize(std::ostream& sout) const  { return sout; };
   //! Object serialization input
   virtual std::istream& serialize(std::istream& sin) { return sin; };
   // @}
};

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

   \version 1.53 (24 Jan 2008)
   - Changed reference from channel to channel<sigspace>

   \version 2.00 (28 Jan 2008)
   - Abstracted modulator class by templating, with the channel-symbol type as
     template parameter; this is meant to allow the use of channels that do not
     use signal-space transmission.
   - Signal-space specific functions are moved to a class specialization.
   - Common modulator interface moved to basic_modulator template.
   - This class cannot be instantiated as it is still abstract.

   \version 2.01 (12-13 Feb 2008)
   - Implemented as templated GF(q) modulator - template argument class must
     provide a method elements() that returns the field size.
   - Renamed template argument to G
   - Fixed error in modulate function (symbol splitting was incorrect)
   - Changed modulate and demodulate to be more similar to those in lut_modulator

   \todo Merge modulate and demodulate between this function and lut_modulator
*/

template <class G> class modulator : public basic_modulator<G> {
   static const libbase::serializer shelper;
   static void* create() { return new modulator<G>; };
public:
   // Serialization Support
   modulator<G> *clone() const { return new modulator<G>(*this); };
   const char* name() const { return shelper.name(); };

   // Atomic modem operations
   const G modulate(const int index) const { assert(index >= 0 && index < num_symbols()); return G(index); };
   const int demodulate(const G& signal) const { return signal; };

   // Vector modem operations
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<G>& tx);
   void demodulate(const channel<G>& chan, const libbase::vector<G>& rx, libbase::matrix<double>& ptable);

   // Informative functions
   int num_symbols() const { return G::elements(); };

   // Description & Serialization
   std::string description() const;
};

/*! \name Serialization */

template <class S> std::ostream& operator<<(std::ostream& sout, const modulator<S>* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

template <class S> std::istream& operator>>(std::istream& sin, modulator<S>*& x)
   {
   std::string name;
   sin >> name;
   x = (modulator<S> *) libbase::serializer::call("modulator", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (modulator): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

// @}

/*!
   \brief   Signal-Space Modulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (24 Jan 2008)
   - Elements specific to the signal-space channel moved to this implementation
     derived from the abstract class.
*/

template <> class modulator<sigspace> : public basic_modulator<sigspace> {
public:
   /*! \name Informative functions */
   //! Average energy per symbol
   virtual double energy() const = 0;
   //! Average energy per bit
   double bit_energy() const { return energy()/log2(num_symbols()); };
   //! Modulation rate (spectral efficiency) in bits/unit energy
   double rate() const { return 1.0/bit_energy(); };
   // @}
};

/*!
   \brief   Binary Modulator.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (24 Jan 2008)
   - Elements specific to the binary channel moved to this implementation
     derived from the abstract class.
*/

template <> class modulator<bool> : public basic_modulator<bool> {
   static const libbase::serializer shelper;
   static void* create() { return new modulator<bool>; };
public:
   // Serialization Support
   modulator<bool> *clone() const { return new modulator<bool>(*this); };
   const char* name() const { return shelper.name(); };

   // Atomic modem operations
   const bool modulate(const int index) const { assert(index >= 0 && index <= 1); return index & 1; };
   const int demodulate(const bool& signal) const { return signal; };

   // Vector modem operations
   void modulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx);
   void demodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable);

   // Informative functions
   int num_symbols() const { return 2; };

   // Description & Serialization
   std::string description() const;
};

}; // end namespace

#endif
