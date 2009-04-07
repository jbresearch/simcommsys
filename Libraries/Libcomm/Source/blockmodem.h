#ifndef __blockmodem_h
#define __blockmodem_h

#include "modem.h"
#include "vector.h"
#include "matrix.h"
#include "size.h"
#include "channel.h"
#include "blockprocess.h"

namespace libcomm {

/*!
   \brief   Blockwise Modulator Common Interface.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Class defines common interface for blockmodem classes.

   \todo Templatize with respect to the type used for the likelihood table
*/

template <class S, template<class> class C=libbase::vector>
class basic_blockmodem :
   public modem<S>,
   public blockprocess {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>     array1d_t;
   // @}

private:
   /*! \name User-defined parameters */
   libbase::size<C> size;    //!< Input block size in symbols
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Setup function, called from set_blocksize() in base class
   virtual void setup() {};
   //! Validates block size, called from modulate() and demodulate()
   virtual void test_invariant() const { assert(size > 0); };
   //! \copydoc modulate()
   virtual void domodulate(const int N, const C<int>& encoded, C<S>& tx) = 0;
   //! \copydoc demodulate()
   virtual void dodemodulate(const channel<S>& chan, const C<S>& rx, C<array1d_t>& ptable) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~basic_blockmodem() {};
   // @}

   // Atomic modem operations
   // (necessary because overloaded methods hide those in templated base)
   using modem<S>::modulate;
   using modem<S>::demodulate;

   /*! \name Vector modem operations */
   /*!
      \brief Modulate a sequence of time-steps
      \param[in]  N        The number of possible values of each encoded element
      \param[in]  encoded  Sequence of values to be modulated
      \param[out] tx       Sequence of symbols corresponding to the given input

      \todo Remove parameter N, replacing 'int' type for encoded vector with
            something that also encodes the number of symbols in the alphabet.

      \note This function is non-const, to support time-variant modulation
            schemes such as DM inner codes.
   */
   void modulate(const int N, const C<int>& encoded, C<S>& tx);
   /*!
      \brief Demodulate a sequence of time-steps
      \param[in]  chan     The channel model (used to obtain likelihoods)
      \param[in]  rx       Sequence of received symbols
      \param[out] ptable   Table of likelihoods of possible transmitted symbols

      \note \c ptable(i)(d) \c is the a posteriori probability of having
            transmitted symbol 'd' at time 'i'

      \note This function is non-const, to support time-variant modulation
            schemes such as DM inner codes.
   */
   void demodulate(const channel<S>& chan, const C<S>& rx, C<array1d_t>& ptable);
   // @}

   /*! \name Setup functions */
   //! Sets input block size
   void set_blocksize(libbase::size<C> size) { assert(size > 0); this->size = size; this->setup(); };
   // @}

   /*! \name Informative functions */
   //! Gets input block size
   libbase::size<C> input_block_size() const { return size; };
   //! Gets output block size
   virtual libbase::size<C> output_block_size() const { return size; };
   // @}
};

/*!
   \brief   Blockwise Modulator Base.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Class defines base interface for blockmodem classes.
*/

template <class S, template<class> class C=libbase::vector>
class blockmodem : public basic_blockmodem<S,C> {
public:
   //! Virtual destructor
   virtual ~blockmodem() {};
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(blockmodem);
};

/*!
   \brief   Q-ary Blockwise Modulator.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Specific implementation of q-ary channel modulation.

   \note Template argument class must provide a method elements() that returns
         the field size.

   \todo Merge modulate and demodulate between this function and lut_modulator

   \todo Find out why using declarations are not working.
*/

template <class G>
class direct_blockmodem :
   public blockmodem<G>,
   protected direct_modem_implementation<G> {
public:
   /*! \name Type definitions */
   typedef direct_modem_implementation<G> Implementation;
   typedef libbase::vector<double>     array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<G>& tx);
   void dodemodulate(const channel<G>& chan, const libbase::vector<G>& rx, libbase::vector<array1d_t>& ptable);

public:
   // Use implementation from base
   // Atomic modem operations
   const G modulate(const int index) const { return Implementation::modulate(index); };
   const int demodulate(const G& signal) const { return Implementation::demodulate(signal); };
   // Informative functions
   int num_symbols() const { return Implementation::num_symbols(); };
   //using direct_modem_implementation<G>::modulate;
   //using direct_modem_implementation<G>::demodulate;
   //using direct_modem_implementation<G>::num_symbols;

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(direct_blockmodem);
};

/*!
   \brief   Binary Blockwise Modulator.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Specific implementation of binary channel modulation.

   \todo Find out why using declarations are not working.
*/

template <>
class direct_blockmodem<bool> :
   public blockmodem<bool>,
   protected direct_modem_implementation<bool> {
public:
   /*! \name Type definitions */
   typedef direct_modem_implementation<bool> Implementation;
   typedef libbase::vector<double>     array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx);
   void dodemodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::vector<array1d_t>& ptable);

public:
   // Use implementation from base
   // Atomic modem operations
   const bool modulate(const int index) const { return Implementation::modulate(index); };
   const int demodulate(const bool& signal) const { return Implementation::demodulate(signal); };
   // Informative functions
   int num_symbols() const { return Implementation::num_symbols(); };
   //using direct_modem_implementation<bool>::modulate;
   //using direct_modem_implementation<bool>::demodulate;
   //using direct_modem_implementation<bool>::num_symbols;

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(direct_blockmodem);
};

}; // end namespace

#endif
