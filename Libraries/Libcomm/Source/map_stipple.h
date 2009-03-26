#ifndef __map_stipple_h
#define __map_stipple_h

#include "map_straight.h"

namespace libcomm {

/*!
   \brief   Stipple Mapper.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   This class defines an punctured version of the straight mapper, suitable
   for turbo codes, where:
   - all information symbols are transmitted
   - parity symbols are taken from successive sets to result in an overall
     rate of 1/2
   For a two-set turbo code, this corresponds to odd/even puncturing.

   \note This supersedes the puncture_stipple class; observe though that the
         number of sets here corresponds to the definition used in the turbo
         code, and is one less than that for puncture_stipple.
*/

template <template<class> class C=libbase::vector>
class map_stipple :
   public map_straight<C> {
public:
   /*! \name Type definitions */
   typedef map_straight<C> Base;
   typedef libbase::vector<double>     array1d_t;
   // @}

private:
   /*! \name User-defined parameters */
   int sets;   //!< Number of turbo code parallel sets
   // @}
   /*! \name Internal object representation */
   mutable C<bool> pattern;   //!< Pre-computed puncturing pattern
   // @}

protected:
   // Pull in base class variables
   using Base::tau;
   using Base::M;

protected:
   // Interface with mapper
   void advance() const;
   void dotransform(const C<int>& in, C<int>& out) const;
   void doinverse(const C<array1d_t>& pin, C<array1d_t>& pout) const;

public:
   /*! \name Constructors / Destructors */
   virtual ~map_stipple() {};
   // @}

   // Informative functions
   double rate() const { return (sets+1)/2.0; };
   int output_block_size() const { return this->tau*2; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_stipple);
};

}; // end namespace

#endif
