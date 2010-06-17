#ifndef __direct_blockembedder_h
#define __direct_blockembedder_h

#include "blockembedder.h"
#include "embedder.h"

namespace libcomm {

/*!
 * \brief   Position-Independent Blockwise Data Embedder/Extractor.
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * This class is a template definition for position-independent block
 * embedders; this needs to be specialized for actual use. Template parameter
 * defaults are provided here.
 */

template <class S, template <class > class C = libbase::vector,
      class dbl = double>
class direct_blockembedder : public blockembedder<S, C, dbl> {
};

/*!
 * \brief   Position-Independent Vector Data Embedder/Extractor
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Vector implementation of a position-independent block embedder.
 */

template <class S, class dbl>
class direct_blockembedder<S, libbase::vector, dbl> : public blockembedder<S,
      libbase::vector, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   embedder<S> *implementation; //! Implementation of embedding/extraction functions
   // @}
protected:
   // Interface with derived classes
   void doembed(const int N, const libbase::vector<int>& data,
         const libbase::vector<S>& host, libbase::vector<S>& tx);
   void doextract(const channel<S, libbase::vector>& chan,
         const libbase::vector<S>& rx, libbase::vector<array1d_t>& ptable);
public:
   // Informative functions
   int num_symbols() const
      {
      return implementation->num_symbols();
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(direct_blockembedder)
};

/*!
 * \brief   Position-Independent Matrix Data Embedder/Extractor
 * \author  Johann Briffa
 *
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 *
 * Matrix implementation of a position-independent block embedder.
 */

template <class S, class dbl>
class direct_blockembedder<S, libbase::matrix, dbl> : public blockembedder<S,
      libbase::matrix, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<dbl> array1d_t;
   // @}
private:
   /*! \name User-defined parameters */
   embedder<S> *implementation; //! Implementation of embedding/extraction functions
   // @}
protected:
   // Interface with derived classes
   void doembed(const int N, const libbase::matrix<int>& data,
         const libbase::matrix<S>& host, libbase::matrix<S>& tx);
   void doextract(const channel<S, libbase::matrix>& chan,
         const libbase::matrix<S>& rx, libbase::matrix<array1d_t>& ptable);
public:
   // Informative functions
   int num_symbols() const
      {
      return implementation->num_symbols();
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(direct_blockembedder)
};

} // end namespace

#endif
