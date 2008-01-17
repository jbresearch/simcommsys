#ifndef __bsid_h
#define __bsid_h

#include "config.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <math.h>

namespace libcomm {

/*!
   \brief   Binary substitution/insertion/deletion channel.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \version 1.00 (12-16 Oct 2007)
   - Initial version; implementation of a binary substitution, insertion, and deletion channel.
   - \b Note: this class is still unfinished, and only implements the BSC channel right now

   \version 1.01 (17 Oct 2007)
   - changed class to conform with channel 1.52.

   \version 1.10 (18 Oct 2007)
   - added transmit() and receive() functions to actually handle insertions and deletions
   - kept corrupt() and pdf() to be used internally for dealing with substitution errors
   - added specification of Pd and Pi during creation, defaulting to zero (effectively gives a BSC)
   - added serialization of Pd and Pi

   \version 1.20 (23 Oct 2007)
   - added functions to set channel parameters directly (as Ps, Pd, Pi)
   - implemented the receive() function to return Q_m(s) as defined by Davey; one will need to
     first update Ps, depending on whether the receive() is operating wrt the actual channel (ie the
     actual substitution error) or wrt the sparse vector (ie the vector average density).

   \version 1.21 (29 Oct 2007)
   - updated clone() to return this object's type, rather than its base class type. [cf. Stroustrup 15.6.2]

   \version 1.22 (1 Nov 2007)
   - implemented receive() for a sequence of transmitted symbols
      - uses the forward-backward algorithm by including a derived fba object.
      - to simplify matters, double-precision math is used in fba.
      - this required the creation of a specific implementation, including P() and Q().
      - added I and xmax parameters to this class (variables and constructor).
   - added protected default constructor for use by create()
   - changed constructor to use internal functions for setting parameters.
   - added getters for channel parameters
   - updated serialization to include all parameters
   - fixed include-once definition

   \version 1.22 (2 Nov 2007)
   - removed Pd and Pi from serialization and from construction
   - now setting defaults for Ps,Pd,Pi to zero in all constructors, through a new
     function init()
   - added boolean construction parameters varyPs, varyPd, varyPi, to indicate
     what should be changed when the SNR is updated; all default to true; these
     are held by protected variables, so that they can be accessed by derived classes.

   \version 1.23 (5-6 Nov 2007)
   - updated transmit() to cater for the usual case where the tx and rx vectors
     are actually the same.
   - fixed error in receive(), when tau=1, where the special case of m=-1 was
     not handled.
   - changed varyPx variables from protected to private; these are only changed
     on initialization or serialization - derived classes should delegate to this
     class's serialization routines as needed.
   - added getters for I and xmax (watermarkcode needs them to set up fba)
     \b TODO: this should probably change, separating or integrating bsid & fba
   - fixed ptable and getF indexing errors in receive() for M=1.

   \version 1.24 (14 Nov 2007)
   - optimized receive() for the case when tau=1

   \version 1.30 (15 Nov 2007)
   - implemented refactoring changes in channel 1.60
   - inlined the single-symbol receive() and pdf()
   - inlined myfba::P and Q
   - reduced pdf() using ternary operator
   - added pre-computed parameters, to reduce work in single-symbol receive();
     also updated single-timestep receive() accordingly.
   - updated compute_parameters() to use set_ps/i/d instead of direct-access
   - added pre-computed parameter to reduce work in myfba::P()

   \version 1.31 (28 Nov 2007)
   - moved call to init() from default constructor to end of serialization input

   \version 1.40 (6 Dec 2007)
   - removed I and xmax from user-defined parameters, instead determining the value
     from the current channel parameters; this allows much smaller values (and
     therefore faster simulations) at low error rates.
   - added N as a user-defined parameter, since this is required to determine
     I and xmax
*/

class bsid : public channel {
   /*! \name Serialization */
   static const libbase::serializer shelper;
   static void* create() { return new bsid; };
   // @}
private:
   /*! \name User-defined parameters */
   double   Ps;         //!< Bit-substitution probability \f$ P_s \f$
   double   Pd;         //!< Bit-deletion probability \f$ P_d \f$
   double   Pi;         //!< Bit-insertion probability \f$ P_i \f$
   int      N;          //!< Block size in bits over which we want to synchronize
   bool     varyPs;     //!< Flag to indicate that \f$ P_s \f$ should change with SNR
   bool     varyPd;     //!< Flag to indicate that \f$ P_d \f$ should change with SNR
   bool     varyPi;     //!< Flag to indicate that \f$ P_i \f$ should change with SNR
   // @}
   /*! \name Pre-computed parameters */
   //! Assumed limit for insertions between two time-steps
   /*!
      \f[ I = \left\lceil \frac{ \log{P_r} - \log N }{ \log p } \right\rceil - 1 \f]
      where \f$ P_r \f$ is an arbitrary probability of having a block of size \f$ N \f$
      with at least one event of more than \f$ I \f$ insertions between successive
      time-steps. In this class, this value is fixed at \f$ P_r = 10^{-12} \f$.
      \note The smallest allowed value is \f$ I = 1 \f$
   */
   int      I;
   //! Assumed maximum drift over a whole \c N -bit block
   /*!
      \f[ x_{max} = 5 \sqrt{N p (1-p)} \f]
      where \f$ p = P_i = P_d \f$. This is based directly on Davey's suggestion that
      \f$ x_{max} \f$ should be "several times larger" than the standard deviation of
      the synchronization drift over one block, given by \f$ \sigma = \sqrt{N p (1-p)} \f$
      \note The smallest allowed value is \f$ x_{max} = I \f$
   */
   int      xmax;
   //! Receiver coefficient \f$ a_1 = 1-P_i-P_d \f$
   double   a1;
   //! Receiver coefficient \f$ a_2 = \frac{1}{2} P_i P_d \f$
   double   a2;
   //! Receiver coefficient set
   /*!
      \f[ a_3(m) = \frac{1}{2^m (1-P_i) (1-P_d)}, m \in (0, \ldots x_{max}) \f]
   */
   libbase::vector<double> a3;
   // @}
private:
   /*! \name Internal functions */
   void init();
   void precompute();
   // @}
protected:
   /*! \name Constructors / Destructors */
   //! Default constructor
   bsid() {};
   // @}
   /*! \name Channel function overrides */
   void compute_parameters(const double Eb, const double No);
   sigspace corrupt(const sigspace& s);
   double pdf(const sigspace& tx, const sigspace& rx) const;
   // @}
public:
   /*! \name Constructors / Destructors */
   bsid(const int N, const bool varyPs=true, const bool varyPd=true, const bool varyPi=true);
   // @}
   /*! \name Serialization Support */
   bsid *clone() const { return new bsid(*this); };
   const char* name() const { return shelper.name(); };
   // @}

   /*! \name Channel parameter setters */
   //! Set the bit-substitution probability
   void set_ps(const double Ps);
   //! Set the bit-deletion probability
   void set_pd(const double Pd);
   //! Set the bit-insertion probability
   void set_pi(const double Pi);
   // @}

   /*! \name Channel parameter getters */
   //! Get the current bit-substitution probability
   double get_ps() const { return Ps; };
   //! Get the current bit-deletion probability
   double get_pd() const { return Pd; };
   //! Get the current bit-insertion probability
   double get_pi() const { return Pi; };
   // @}

   /*! \name FBA decoder parameter getters */
   //! Get the current assumed limit for insertions between two time-steps
   int get_I() const { return I; };
   //! Get the current assumed maximum drift over a whole N-bit block
   int get_xmax() const { return xmax; };
   // @}

   /*! \name Channel functions */
   void transmit(const libbase::vector<sigspace>& tx, libbase::vector<sigspace>& rx);
   void receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx, libbase::matrix<double>& ptable) const;
   double receive(const libbase::vector<sigspace>& tx, const libbase::vector<sigspace>& rx) const;
   double receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const;
   // @}

   /*! \name Description & Serialization */
   //! Object description output
   std::string description() const;
   //! Object serialization ouput
   std::ostream& serialize(std::ostream& sout) const;
   //! Object serialization input
   std::istream& serialize(std::istream& sin);
   // @}
};

inline double bsid::pdf(const sigspace& tx, const sigspace& rx) const
   {
   return (tx != rx) ? Ps : 1-Ps;
   }

inline double bsid::receive(const sigspace& tx, const libbase::vector<sigspace>& rx) const
   {
   // Compute sizes
   const int m = rx.size()-1;
   // set of possible transmitted symbols for one transmission step
   if(m == -1) // just a deletion, no symbols received
      return Pd;
   // Work out the probabilities of each possible signal
   return (a1 * pdf(tx,rx(m)) + a2) * a3(m);
   }

}; // end namespace

#endif

