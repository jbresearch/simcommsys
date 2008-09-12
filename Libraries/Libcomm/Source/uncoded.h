#ifndef __uncoded_h
#define __uncoded_h

#include "config.h"

#include "codec_softout.h"
#include "fsm.h"
#include "serializer.h"
#include <stdlib.h>
#include <math.h>

namespace libcomm {

/*!
   \brief   Uncoded transmission.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

*/

class uncoded : public codec_softout<double> {
private:
   /*! \name Internally-used types */
   typedef libbase::vector<int>     array1i_t;
   typedef libbase::matrix<double>  array2d_t;
   // @}
private:
   fsm   *encoder;
   int   tau;  //!< block length
   array1i_t lut;
   array2d_t R;
protected:
   void init();
   void free();
   uncoded();
public:
   /*! \name Constructors / Destructors */
   uncoded(const fsm& encoder, const int tau);
   ~uncoded() { free(); };
   // @}

   // Codec operations
   void encode(array1i_t& source, array1i_t& encoded);
   void translate(const array2d_t& ptable);
   void decode(array2d_t& ri);
   void decode(array2d_t& ri, array2d_t& ro);

   // Codec information functions - fundamental
   int block_size() const { return tau; };
   int num_inputs() const { return encoder->num_inputs(); };
   int num_outputs() const { return encoder->num_outputs(); };
   int tail_length() const { return 0; };
   int num_iter() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(uncoded)
};

}; // end namespace

#endif

