#ifndef __qim_h
#define __qim_h

#include "config.h"
#include "embedder.h"

namespace libcomm {

/*!
 * \brief   Quantization Index Modulation (QIM) Embedder/Extractor.
 * \author  Johann Briffa
 *
 * \par Version Control:
 * - $Revision$
 * - $Date$
 * - $Author$
 */

template <class S>
class qim : public embedder<S> {
private:
   /*! \name User-defined parameters */
   int M; //! Alphabet size in symbols
   S delta; //! QIM bin width
   double alpha; //! QIM distortion-compensation factor (0 < alpha <= 1)
   // @}
protected:
   //! Verifies that object is in a valid state
   void test_invariant() const
      {
      assert(M >= 2);
      assert(delta > 0);
      assert(alpha > 0.0 && alpha <= 1.0);
      }
   //! Embedder without distortion compensation
   const S Q(const int i, const double s) const
      {
      return (round((s / delta - i) / M) * M + i) * delta;
      }
public:
   qim(const int M = 2, const S delta = 1, const double alpha = 1) :
      M(M), delta(delta), alpha(alpha)
      {
      }

   // Atomic embedder operations
   const S embed(const int i, const S s) const
      {
      return Q(i, s * alpha) + (1 - alpha) * s;
      }
   const int extract(const S& rx) const
      {
      // Find the symbol with the smallest discrepancy
      int d = 0;
      S best = abs(rx - embed(d, rx));
      for (int i = 1; i < M; i++)
         {
         const S diff = abs(rx - embed(i, rx));
         if (diff < best)
            {
            best = diff;
            d = i;
            }
         }
      return d;
      }

   // Informative functions
   int num_symbols() const
      {
      return M;
      }

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER( qim)
};

} // end namespace

#endif
