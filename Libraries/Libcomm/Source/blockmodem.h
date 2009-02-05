#ifndef __blockmodem_h
#define __blockmodem_h

#include "modem.h"
#include "vector.h"
#include "matrix.h"
#include "channel.h"
#include "blockprocess.h"

namespace libcomm {

/*!
   \brief   Blockwise Modulator Base.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Class defines common interface for blockmodem classes.

   \todo Templatize with respect to the type used for the likelihood table

   \todo Separate block size definition from this class

   \todo Avoid using virtual inheritance

   \todo Test inheritance of virtual functions in VS 2005
*/

template <class S, template<class> class C=libbase::vector>
class blockmodem : public blockprocess, public virtual modem<S> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>     array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   int tau;    //!< Block size in symbols
   // @}

protected:
   /*! \name Interface with derived classes */
   //! Setup function, called from set_blocksize()
   virtual void setup() {};
   //! \copydoc modulate()
   virtual void domodulate(const int N, const C<int>& encoded, C<S>& tx) = 0;
   //! \copydoc demodulate()
   virtual void dodemodulate(const channel<S>& chan, const C<S>& rx, C<array1d_t>& ptable) = 0;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   blockmodem() { tau = 0; };
   //! Virtual destructor
   virtual ~blockmodem() {};
   // @}

   // Atomic modem operations
   // (necessary because inheriting methods from templated base)
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
   void set_blocksize(int tau) { assert(tau > 0); blockmodem::tau = tau; setup(); };
   // @}

   /*! \name Informative functions */
   //! Gets input block size
   int input_block_size() const { return tau; };
   //! Gets output block size
   virtual int output_block_size() const { return tau; };
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
*/

template <class G>
class direct_blockmodem : public blockmodem<G>, public direct_modem<G> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>     array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<G>& tx);
   void dodemodulate(const channel<G>& chan, const libbase::vector<G>& rx, libbase::vector<array1d_t>& ptable);

public:
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
*/

template <>
class direct_blockmodem<bool> : public blockmodem<bool>, public direct_modem<bool> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<double>     array1d_t;
   // @}
protected:
   // Interface with derived classes
   void domodulate(const int N, const libbase::vector<int>& encoded, libbase::vector<bool>& tx);
   void dodemodulate(const channel<bool>& chan, const libbase::vector<bool>& rx, libbase::vector<array1d_t>& ptable);

public:
   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(direct_blockmodem);
};

}; // end namespace

#endif
