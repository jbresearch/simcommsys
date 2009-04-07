#ifndef __mapcc_h
#define __mapcc_h

#include "config.h"

#include "codec_softout.h"
#include "fsm.h"
#include "bcjr.h"
#include "itfunc.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

namespace libcomm {

/*!
   \brief   Maximum A-Posteriori Decoder.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   \todo Check for both flags (for terminated and circular trellises) being set.
*/

template <class real>
class mapcc : public codec_softout<double>, private bcjr<real> {
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>     array1i_t;
   typedef libbase::vector< libbase::vector<double> >  array2d_t;
   // @}
private:
   fsm      *encoder;
   double   rate;
   int      tau;           //!< Block length (including tail, if any)
   bool     endatzero;     //!< True for terminated trellis
   bool     circular;      //!< True for circular trellis
   int      m;             //!< encoder memory order
   int      M;             //!< Number of states
   int      K;             //!< Number of input combinations
   int      N;             //!< Number of output combinations
   libbase::matrix<double> R;            //!< BCJR a-priori statistics
protected:
   /*! \name Internal functions */
   void init();
   void free();
   void reset();
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   mapcc();
   // @}
public:
   /*! \name Constructors / Destructors */
   mapcc(const fsm& encoder, const int tau, const bool endatzero, const bool circular=false);
   ~mapcc() { free(); };
   // @}

   // Codec operations
   void encode(const array1i_t& source, array1i_t& encoded);
   void translate(const array2d_t& ptable);
   void softdecode(array2d_t& ri);
   void softdecode(array2d_t& ri, array2d_t& ro);

   // Codec information functions - fundamental
   libbase::size<libbase::vector> input_block_size() const
      { return libbase::size<libbase::vector>(endatzero ? tau-m : tau); };
   libbase::size<libbase::vector> output_block_size() const
      { return libbase::size<libbase::vector>(tau); };
   int num_inputs() const { return K; };
   int num_outputs() const { return N; };
   int tail_length() const { return m; };
   int num_iter() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(mapcc);
};

}; // end namespace

#endif

