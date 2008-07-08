#ifndef __bsid_h
#define __bsid_h

#include "config.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include <boost/multi_array.hpp>
#include <math.h>

namespace libcomm {

/*!
   \brief   Binary substitution/insertion/deletion channel.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

class bsid : public channel<bool> {
private:
   /*! \name Internally-used types */
   typedef boost::multi_array<double,2> array2d_t;
   typedef boost::multi_array<double,1> array1d_t;
   // @}
   /*! \name User-defined parameters */
   bool     varyPs;     //!< Flag to indicate that \f$ P_s \f$ should change with parameter
   bool     varyPd;     //!< Flag to indicate that \f$ P_d \f$ should change with parameter
   bool     varyPi;     //!< Flag to indicate that \f$ P_i \f$ should change with parameter
   // @}
   /*! \name Channel-state parameters */
   double   Ps;         //!< Bit-substitution probability \f$ P_s \f$
   double   Pd;         //!< Bit-deletion probability \f$ P_d \f$
   double   Pi;         //!< Bit-insertion probability \f$ P_i \f$
   mutable int N;       //!< Block size in bits over which we want to synchronize
   // @}
   /*! \name Pre-computed parameters */
   mutable int I;       //!< Assumed limit for insertions between two time-steps
   mutable int xmax;    //!< Assumed maximum drift over a whole \c N -bit block
   mutable array2d_t Rtable;  //!< Receiver coefficient set
   mutable array1d_t Ptable;  //!< Forward recursion 'P' function lookup
   // @}
public:
   /*! \name FBA decoder parameter computation */
   static int compute_I(int N, double p);
   static int compute_xmax(int N, double p, int I);
   static int compute_xmax(int N, double p);
   static void compute_Rtable(array2d_t& Rtable, int xmax, double Ps, double Pd, double Pi);
   static void compute_Ptable(array1d_t& Ptable, int xmax, double Pd, double Pi);
   // @}
private:
   /*! \name Internal functions */
   void precompute() const;
   void init();
   // @}
protected:
   // Channel function overrides
   bool corrupt(const bool& s);
   double pdf(const bool& tx, const bool& rx) const;
public:
   /*! \name Constructors / Destructors */
   bsid(const bool varyPs=true, const bool varyPd=true, const bool varyPi=true);
   // @}

   /*! \name Channel parameter handling */
   void set_parameter(const double p);
   double get_parameter() const;
   // @}

   /*! \name Channel parameter setters */
   //! Set the bit-substitution probability
   void set_ps(const double Ps);
   //! Set the bit-deletion probability
   void set_pd(const double Pd);
   //! Set the bit-insertion probability
   void set_pi(const double Pi);
   //! Set the block size
   void set_blocksize(int N) const;
   // @}

   /*! \name Channel parameter getters */
   //! Get the current bit-substitution probability
   double get_ps() const { return Ps; };
   //! Get the current bit-deletion probability
   double get_pd() const { return Pd; };
   //! Get the current bit-insertion probability
   double get_pi() const { return Pi; };
   // @}

   // Channel functions
   void transmit(const libbase::vector<bool>& tx, libbase::vector<bool>& rx);
   void receive(const libbase::vector<bool>& tx, const libbase::vector<bool>& rx, libbase::matrix<double>& ptable) const;
   double receive(const libbase::vector<bool>& tx, const libbase::vector<bool>& rx) const;
   double receive(const bool& tx, const libbase::vector<bool>& rx) const;

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(bsid)
};

inline double bsid::pdf(const bool& tx, const bool& rx) const
   {
   return (tx != rx) ? Ps : 1-Ps;
   }

inline double bsid::receive(const bool& tx, const libbase::vector<bool>& rx) const
   {
   // Compute sizes
   const int m = rx.size()-1;
   // Return result from table
   return (m < 0) ? Pd : Rtable[tx != rx(m)][m];
   }

}; // end namespace

#endif
