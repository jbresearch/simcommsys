#ifndef __repacc_h
#define __repacc_h

#include "config.h"
#include "codec_softout.h"
#include "codec_softout_flattened.h"
#include "uncoded.h"
#include "fsm.h"
#include "interleaver.h"
#include "safe_bcjr.h"

namespace libcomm {

/*!
 * \brief   Repeat-Accumulate (RA) codes.
 * \author  Johann Briffa
 * 
 * \section svn Version Control
 * - $Revision$
 * - $Date$
 * - $Author$
 * 
 * These codes are decoded using the MAP decoder, rather than the
 * sum-product algorithm.
 * 
 * \todo Avoid divisions when computing extrinsic information
 * 
 * \todo Implement accumulator as mapcc
 * 
 * \todo Generalize repeater and accumulator
 */

template <class real, class dbl = double>
class repacc : public codec_softout<libbase::vector, dbl> ,
      protected safe_bcjr<real, dbl> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int> array1i_t;
   typedef libbase::vector<dbl> array1d_t;
   typedef libbase::matrix<dbl> array2d_t;
   typedef libbase::vector<array1d_t> array1vd_t;
   // @}
private:
   // Shorthand for class hierarchy
   typedef repacc<real, dbl> This;
   typedef safe_bcjr<real, dbl> BCJR;
private:
   /*! \name User-defined parameters */
   //! Interleaver between repeater and accumulator
   interleaver<dbl> *inter;
   //! MAP representation of repetition code
   codec_softout_flattened<uncoded<dbl> , dbl> rep;
   fsm *acc; //!< Encoder representation of accumulator
   int iter; //!< Number of iterations to perform
   bool endatzero; //!< Flag to indicate that trellises are terminated
   // @}
protected:
   /*! \name Internal object representation */
   bool initialised; //!< Flag to indicate when memory is initialised
   array1vd_t rp; //!< Intrinsic source statistics (natural)
   array2d_t ra; //!< Extrinsic accumulator-input statistics (natural)
   array2d_t R; //!< Intrinsic accumulator-output statistics (interleaved)
   // @}
protected:
   /*! \name Internal functions */
   void init();
   void free();
   void reset();
   //! Memory allocator (for internal use only)
   void allocate();
   //! Determine the number of timesteps for the accumulator
   int acc_timesteps() const
      {
      // Inherit sizes
      const int Nr = rep.output_block_size();
      const int k = acc->num_inputs();
      const int nu = tail_length();
      return Nr / k + nu;
      }
   // @}
   // Internal codec operations
   void resetpriors();
   void setpriors(const array1vd_t& ptable);
   void setreceiver(const array1vd_t& ptable);
public:
   /*! \name Constructors / Destructors */
   repacc();
   ~repacc()
      {
      free();
      }
   // @}

   // Codec operations
   void seedfrom(libbase::random& r);
   void encode(const array1i_t& source, array1i_t& encoded);
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   libbase::size_type<libbase::vector> input_block_size() const
      {
      // Inherit sizes
      const int N = rep.input_block_size();
      return libbase::size_type<libbase::vector>(N);
      }
   libbase::size_type<libbase::vector> output_block_size() const
      {
      // Inherit sizes
      const int n = acc->num_outputs();
      const int tau = acc_timesteps();
      return libbase::size_type<libbase::vector>(n * tau);
      }
   int num_inputs() const
      {
      return acc->num_symbols();
      }
   int num_outputs() const
      {
      return acc->num_symbols();
      }
   int tail_length() const
      {
      return endatzero ? acc->mem_order() : 0;
      }
   int num_iter() const
      {
      return iter;
      }

   /*! \name Codec information functions - internal */
   int num_repeats() const
      {
      return int(round(log(rep.num_outputs()) / log(rep.num_inputs())));
      }
   const interleaver<dbl> *get_inter() const
      {
      return inter;
      }
   // @}

   // Description
   std::string description() const;

   // Serialization Support
DECLARE_SERIALIZER(repacc);
};

} // end namespace

#endif
