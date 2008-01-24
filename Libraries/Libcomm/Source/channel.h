#ifndef __channel_h
#define __channel_h

#include "config.h"
#include "vector.h"
#include "matrix.h"

#include "randgen.h"
#include "sigspace.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Common Channel Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (24 Jan 2008)
   - Contains common interface for channel classes.

   \todo Think out and update cloning/serialization interface
*/

template <class S> class basic_channel {
protected:
   /*! \name Derived channel representation */
   libbase::randgen  r;
   // @}
protected:
   /*! \name Channel function overrides */
   /*!
      \brief Pass a single symbol through the substitution channel
      \param   s  Input (Tx) symbol
      \return  Output (Rx) symbol
   */
   virtual sigspace corrupt(const S& s) = 0;
   /*!
      \brief Determine the conditional likelihood for the received symbol
      \param   tx  Transmitted symbol being considered
      \param   rx  Received symbol
      \return  Likelihood \f$ P(rx|tx) \f$
   */
   virtual double pdf(const S& tx, const S& rx) const = 0;
   // @}
public:
   /*! \name Constructors / Destructors */
   virtual ~basic_channel() {};
   // @}
   /*! \name Serialization Support */
   virtual basic_channel *clone() const = 0;
   virtual const char* name() const = 0;
   // @}

   /*! \name Channel parameter handling */
   //! Reset function for random generator
   void seed(libbase::int32u const s) { r.seed(s); };
   //! Set the channel characteristic parameter
   virtual void set_parameter(const double x) = 0;
   //! Get the channel characteristic parameter
   virtual double get_parameter() const = 0;
   // @}

   /*! \name Channel functions */
   /*!
      \brief Pass a sequence of modulation symbols through the channel
      \param[in]  tx  Transmitted sequence of modulation symbols
      \param[out] rx  Received sequence of modulation symbols

      \note It is possible that the \c tx and \c rx parameters actually point to the same
            vector.

      \callergraph
   */
   virtual void transmit(const libbase::vector<S>& tx, libbase::vector<S>& rx) = 0;
   /*!
      \brief Determine the per-symbol likelihoods of a sequence of received modulation symbols
             corresponding to one transmission step
      \param[in]  tx       Set of possible transmitted symbols
      \param[in]  rx       Received sequence of modulation symbols
      \param[out] ptable   Likelihoods corresponding to each possible transmitted symbol

      \callergraph
   */
   virtual void receive(const libbase::vector<S>& tx, const libbase::vector<S>& rx, libbase::matrix<double>& ptable) const = 0;
   /*!
      \brief Determine the likelihood of a sequence of received modulation symbols, given
             a particular transmitted sequence
      \param[in]  tx       Transmitted sequence being considered
      \param[in]  rx       Received sequence of modulation symbols
      \return              Likelihood \f$ P(rx|tx) \f$

      \callergraph
   */
   virtual double receive(const libbase::vector<S>& tx, const libbase::vector<S>& rx) const = 0;
   /*!
      \brief Determine the likelihood of a sequence of received modulation symbols, given
             a particular transmitted symbol
      \param[in]  tx       Transmitted symbol being considered
      \param[in]  rx       Received sequence of modulation symbols
      \return              Likelihood \f$ P(rx|tx) \f$

      \callergraph
   */
   virtual double receive(const S& tx, const libbase::vector<S>& rx) const = 0;
   // @}

   /*! \name Description & Serialization */
   //! Object description output
   virtual std::string description() const = 0;
   //! Object serialization ouput
   virtual std::ostream& serialize(std::ostream &sout) const { return sout; };
   //! Object serialization input
   virtual std::istream& serialize(std::istream &sin) { return sin; };
   // @}
};

/*!
   \brief   Channel Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.01 (26 Oct 2001)
   added a virtual destroy function (see interleaver.h)

   \version 1.02 (6 Mar 2002)
   changed vcs version variable from a global to a static class variable.

   \version 1.10 (13 Mar 2002)
   added a virtual function which outputs details on the channel, together with a stream
   << operator too. Also added serialization facility. Created serialize and stream << and
   >> functions to conform with the new serializer protocol, as defined in serializer 1.10.
   The stream << output function first writes the name of the derived class, then calls its
   serialize() to output the data. The name is obtained from the virtual name() function.
   The stream >> input function first gets the name from the stream, then (via
   serialize::call) creates a new object of the appropriate type and calls its serialize()
   function to get the relevant data. Also added cloning function.

   \version 1.20 (17 Mar 2002)
   added a function which corrupts a vector of signals (called transmit). This implements
   the separation of functions from the codec block (as defined in codec 1.40), since
   transmission depends only on the channel, it should be implemented here.

   \version 1.30 (27 Mar 2002)
   removed the descriptive output() and related stream << output functions, and replaced
   them by a function description() which returns a string. This provides the same
   functionality but in a different format, so that now the only stream << output
   functions are for serialization. This should make the notation much clearer while
   also simplifying description display in objects other than streams.

   \version 1.31 (13 Apr 2002)
   changed vector resizing operation in transmit() to use the new format (vector 1.50).

   \version 1.40 (30 Oct 2006)
   - defined class and associated data within "libcomm" namespace.
   - removed use of "using namespace std", replacing by tighter "using" statements as needed.

   \version 1.50 (16 Oct 2007)
   - refactored the class to simplify inheritance:
      - set_eb() and set_snr() are now defined in this class, and call compute_parameters(), which
        is defined in derived classes.
      - consequently, various variables have now moved to this class, together with their getters
      - random generator has also moved into this class, together with its seeding functions
   - updated channel model to allow for insertions and deletions, as well as substitution errors.

   \version 1.51 (16 Oct 2007)
   - refactored further to simplify inheritance:
      - serialization functions are no longer pure virtual; this removes the need for derived classes
        to supply these, unless there is something specific to serialize.

   \version 1.52 (17 Oct 2007)
   - started direct work on implementing support for insertion/deletion:
      - observed that the channel base function corrupt() is only called from within this class (in the
        implementation of transmit(); similarly, pdf() is only called from within the modulator base
        class, in the implementation of demodulate().
      - corrupt() function has been moved into protected space; transmit() has been made virtual, and
        the default implementation still makes use of the corrupt() function from derived classes.
        What this means in practice is that derived classes implementing a DMC can simply implement
        corrupt() and rely on this class to make transmit() available to clients. This is exactly as they
        are doing so far. On the other hand, to implement classes with insertions and deletions, the
        transmit() class can be overridden such that the 'rx' vector doesn't have to be the same length
        as the 'tx' vector.
      - pdf() function has also been moved into protected space, and a new virtual function receive()
        has been created as an interface with clients. The new function receive() provides support for
        channels with insertion and deletion; just as with transmit(), this is not a pure virtual function,
        and a default implementation is given which calls pdf() for every corresponding pair [most of
        this functionality has been moved from modulator.demodulate().

   \version 1.53 (23 Oct 2007)
   - added a function for direct setting of 'No';
      - automatically updates the SNR value
      - automatically calls inherited class compute_parameters()
   - added functions to get values of Eb and No

   \version 1.54 (5 Nov 2007)
   - fixed error in set_no(), where snr_db was incorrectly assumed to depend on Eb as well

   \version 1.60 (15 Nov 2007)
   - refactored the transmit/receive interface: the functionality of receive is now divided
     between overloaded functions, distinguished by the parameters set:
      - receive(tx,rx,ptable) is for the traditional case, used to determine the likelihoods
        (as a matrix) of each of a set of possible transmitted symbols (as a vector) at each
        timestep; the argument list has changed in this case so that the transmitted sequence
        is a vector.
      - receive(tx,rx) is for determining the likelihood of a particular transmitted sequence;
        in this case the argument representing the transmitted sequence is a vector, while the
        likelihood is passed back as a return value, rather than through a ptable.
      - receive(tx,rx) is for determining the likelihood of a particular transmitted symbol;
        in this case the argument representing the transmitted symbol is a signal-space point,
        while the likelihood is passed back as a return value.

   \version 1.61 (17 Jan 2008)
   - Renamed set_snr/get_snr to set_parameter/get_parameter.
   - Hid get_eb/get_no in preparation for abstracting the channel class from its dependance
     on the signal-space representation.

   \version 1.62 (17 Jan 2008)
   - Made serialization functions virtual again (fixed bug introduced in rev. 461
   - Made get_eb/get_no available again, to allow their use in watermarkcode::demodulate();
     TODO: this has to change when abstracting the channel class.

   \version 2.00 (24 Jan 2008)
   - Abstracted channel class by templating, with the channel-symbol type as
     template parameter; this is meant to allow the creation of channels that
     are not tied to signal-space transmission (e.g. the binary symmetric
     channel).
   - Signal-space specific functions are moved to a class specialization.
   - Common channel interface moved to basic_channel template.
*/

template <class S> class channel : public basic_channel<S> {
};

/*! \name Serialization */

template <class S> std::ostream& operator<<(std::ostream& sout, const channel<S>* x)
   {
   sout << x->name() << "\n";
   x->serialize(sout);
   return sout;
   }

template <class S> std::istream& operator>>(std::istream& sin, channel<S>*& x)
   {
   std::string name;
   sin >> name;
   x = (channel<S> *) libbase::serializer::call("channel", name);
   if(x == NULL)
      {
      std::cerr << "FATAL ERROR (channel): Type \"" << name << "\" unknown.\n";
      exit(1);
      }
   x->serialize(sin);
   return sin;
   }

// @}

/*!
   \brief   Signal-Space Channel.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (24 Jan 2008)
   - Elements specific to the signal-space channel model moved to this implementation
     derived from the abstract class.
*/

template <> class channel<sigspace> : public basic_channel<sigspace> {
private:
   /*! \name User-defined parameters */
   double   snr_db;  //!< Equal to \f$ 10 \log_{10} ( \frac{E_b}{N_0} ) \f$
   // @}
   /*! \name Internal representation */
   double   Eb;      //!< Average signal energy per information bit \f$ E_b \f$
   double   No;      //!< Half the noise energy/modulation symbol for a normalised signal \f$ N_0 \f$.
   // @}
private:
   /*! \name Internal functions */
   void compute_noise();
   // @}
protected:
   /*! \name Channel function overrides */
   /*!
      \brief Determine channel-specific parameters based on given SNR

      \note \f$ E_b \f$ is fixed by the overall modulation and coding system. The simulator
            determines \f$ N_0 \f$ according to the given SNR (assuming unit signal energy), so
            that the actual band-limited noise energy is given by \f$ E_b N_0 \f$.
   */
   virtual void compute_parameters(const double Eb, const double No) {};
   // @}
public:
   /*! \name Constructors / Destructors */
   channel();
   // @}

   /*! \name Channel parameter handling */
   //! Set the bit-equivalent signal energy
   void set_eb(const double Eb);
   //! Set the normalized noise energy
   void set_no(const double No);
   //! Get the bit-equivalent signal energy
   double get_eb() const { return Eb; };
   //! Get the normalized noise energy
   double get_no() const { return No; };
   //! Set the signal-to-noise ratio
   void set_parameter(const double snr_db);
   //! Get the signal-to-noise ratio
   double get_parameter() const { return snr_db; };

   /*! \name Channel functions */
   void transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx);
   void receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const;
   double receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx) const;
   double receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const;
   // @}
};

}; // end namespace

#endif

