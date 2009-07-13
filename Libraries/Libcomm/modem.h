#ifndef __modem_h
#define __modem_h

#include "config.h"
#include "random.h"
#include "serializer.h"
#include "sigspace.h"
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

 Class defines common interface for modem classes.
 */

template <class S>
class basic_modem {
public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~basic_modem()
      {
      }
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
   const S operator[](const int index) const
      {
      return modulate(index);
      }
   /*! \copydoc demodulate() */
   const int operator[](const S& signal) const
      {
      return demodulate(signal);
      }
   // @}

   /*! \name Setup functions */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r)
      {
      }
   // @}

   /*! \name Informative functions */
   //! Symbol alphabet size at input
   virtual int num_symbols() const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}
};

/*!
 \brief   Modulator Base.
 \author  Johann Briffa

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$
 */

template <class S>
class modem : public basic_modem<S> {
   // Serialization Support
DECLARE_BASE_SERIALIZER(modem)
};

/*!
 \brief   Signal-Space Modulator Specialization.
 \author  Johann Briffa

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$
 */

template <>
class modem<sigspace> : public basic_modem<sigspace> {
public:
   /*! \name Informative functions */
   //! Average energy per symbol
   virtual double energy() const = 0;
   //! Average energy per bit
   double bit_energy() const
      {
      return energy() / log2(num_symbols());
      }
   //! Modulation rate (spectral efficiency) in bits/unit energy
   double rate() const
      {
      return 1.0 / bit_energy();
      }
   // @}

   // Serialization Support
DECLARE_BASE_SERIALIZER(modem)
};

/*!
 \brief   Q-ary Modulator Implementation.
 \author  Johann Briffa

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$

 Specific implementation of q-ary channel modulation.

 \note Template argument class must provide a method elements() that returns
 the field size.

 \todo Merge modulate and demodulate between this function and lut_modulator (?)
 */

template <class G>
class direct_modem_implementation {
public:
   // Atomic modem operations
   const G modulate(const int index) const
      {
      assert(index >= 0 && index < num_symbols());
      return G(index);
      }
   const int demodulate(const G& signal) const
      {
      return signal;
      }

   // Informative functions
   int num_symbols() const
      {
      return G::elements();
      }

   // Description
   std::string description() const;
};

/*!
 \brief   Binary Modulator Implementation Specialization.
 \author  Johann Briffa

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$

 Specific implementation of binary channel modulation.
 */

template <>
class direct_modem_implementation<bool> {
public:
   // Atomic modem operations
   const bool modulate(const int index) const
      {
      assert(index >= 0 && index <= 1);
      return index & 1;
      }
   const int demodulate(const bool& signal) const
      {
      return signal;
      }

   // Informative functions
   int num_symbols() const
      {
      return 2;
      }

   // Description
   std::string description() const;
};

/*!
 \brief   Direct Modulator Implementation.
 \author  Johann Briffa

 \par Version Control:
 - $Revision$
 - $Date$
 - $Author$
 */

template <class G>
class direct_modem : public modem<G> , protected direct_modem_implementation<G> {
public:
   /*! \name Type definitions */
   typedef direct_modem_implementation<G> Implementation;
   // @}
public:
   // Use implementation from base
   // Atomic modem operations
   const G modulate(const int index) const
      {
      return Implementation::modulate(index);
      }
   const int demodulate(const G& signal) const
      {
      return Implementation::demodulate(signal);
      }
   // Informative functions
   int num_symbols() const
      {
      return Implementation::num_symbols();
      }
   // Description
   std::string description() const
      {
      return Implementation::description();
      }
   //using direct_modem_implementation<G>::modulate;
   //using direct_modem_implementation<G>::demodulate;
   //using direct_modem_implementation<G>::num_symbols;
   //using direct_modem_implementation<G>::description;

   // Serialization Support
DECLARE_SERIALIZER(direct_modem);
};

} // end namespace

#endif
