#ifndef __interleaver_h
#define __interleaver_h

#include "config.h"
#include "serializer.h"
#include "random.h"
#include "matrix.h"
#include "vector.h"
#include "logrealfast.h"

#include <iostream>
#include <string>

namespace libcomm {

/*!
   \brief   Interleaver Base.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   \todo Make 'interleaver' a template class, with the template parameter
         indicating the transform/inverse matrix type support.
*/

class interleaver {
public:
   /*! \name Constructors / Destructors */
   //! Virtual destructor
   virtual ~interleaver() {};
   // @}

   /*! \name Intra-frame Operations */
   //! Seeds any random generators from a pseudo-random sequence
   virtual void seedfrom(libbase::random& r) {};
   virtual void advance() {};
   // @}

   /*! \name Transform Functions */
   /*!
      \brief Forward transform
      \param[in] in Source sequence
      \param[out] out Interleaved sequence
      \note 'in' and 'out' cannot be the same.
   */
   virtual void transform(const libbase::vector<int>& in, libbase::vector<int>& out) const = 0;
   /*!
      \brief Forward transform
      \param[in] in Matrix representing the likelihoods of each possible symbol
      \param[out] out Matrix with likelihoods of interleaved sequence
      \note 'in' and 'out' cannot be the same.
   */
   virtual void transform(const libbase::matrix<double>& in, libbase::matrix<double>& out) const = 0;
   /*!
      \brief Inverse transform
      \param[in] in  Matrix representing the likelihoods of each possible symbol
                     for the interleaved sequence
      \param[out] out Matrix with likelihoods of straight sequence
      \note 'in' and 'out' cannot be the same.
   */
   virtual void inverse(const libbase::matrix<double>& in, libbase::matrix<double>& out) const = 0;
   //! \copydoc interleaver::transform(const libbase::matrix<double>, libbase::matrix<double>)
   virtual void transform(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const = 0;
   //! \copydoc interleaver::inverse(const libbase::matrix<double>, libbase::matrix<double>)
   virtual void inverse(const libbase::matrix<libbase::logrealfast>& in, libbase::matrix<libbase::logrealfast>& out) const = 0;
   // @}

   /*! \name Description */
   //! Description output
   virtual std::string description() const = 0;
   // @}

   // Serialization Support
   DECLARE_BASE_SERIALIZER(interleaver)
};

}; // end namespace

#endif
