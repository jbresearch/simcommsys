#ifndef __bsid2d_h
#define __bsid2d_h

#include "config.h"
#include "channel.h"
#include "itfunc.h"
#include "serializer.h"
#include "multi_array.h"
#include <math.h>

namespace libcomm {

/*!
   \brief   2D binary substitution/insertion/deletion channel.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Derive repeated workings from bsid
*/

class bsid2d : public channel<bool,libbase::matrix> {
public:
   /*! \name Type definitions */
   typedef boost::assignable_multi_array<double,2> array2d_t;
   typedef libbase::vector<bool>       array1b_t;
   typedef libbase::vector<double>     array1d_t;
   typedef libbase::matrix<bool>       array2b_t;
   typedef libbase::matrix<int>        array2i_t;
   typedef libbase::matrix<array1d_t>  array2vd_t;
   // @}
private:
   /*! \name User-defined parameters */
   bool     varyPs;     //!< Flag to indicate that \f$ P_s \f$ should change with parameter
   bool     varyPd;     //!< Flag to indicate that \f$ P_d \f$ should change with parameter
   bool     varyPi;     //!< Flag to indicate that \f$ P_i \f$ should change with parameter
   // @}
   /*! \name Channel-state parameters */
   double   Ps;         //!< Bit-substitution probability \f$ P_s \f$
   double   Pd;         //!< Bit-deletion probability \f$ P_d \f$
   double   Pi;         //!< Bit-insertion probability \f$ P_i \f$
   int      M;          //!< Vertical block size (rows) over which we want to synchronize
   int      N;          //!< Horizontal block size (columns) over which we want to synchronize
   // @}
private:
   /*! \name Internal functions */
   void init();
   void computestate(int& insertions, bool& transmit);
   void computestate(array2i_t& insertions, array2b_t& transmit);
   static void cumsum(array2i_t& m, int dim);
   // @}
protected:
   // Channel function overrides
   bool corrupt(const bool& s);
   double pdf(const bool& tx, const bool& rx) const;
public:
   /*! \name Constructors / Destructors */
   bsid2d(const bool varyPs=true, const bool varyPd=true, const bool varyPi=true);
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
   void set_blocksize(int M, int N);
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
   void transmit(const array2b_t& tx, array2b_t& rx);

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(bsid2d);
};

inline double bsid2d::pdf(const bool& tx, const bool& rx) const
   {
   return (tx != rx) ? Ps : 1-Ps;
   }

}; // end namespace

#endif
