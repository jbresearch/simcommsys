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
public:
   /*! \name Type definitions */
   typedef libbase::vector<int>        array1i_t;
   typedef libbase::vector<double>     array1d_t;
   typedef libbase::vector<array1d_t>  array1vd_t;
   // @}
private:
   /*! \name User-specified parameters */
   fsm   *encoder;
   int   tau;  //!< block length
   // @}
   /*! \name Computed parameters */
   array1i_t lut;
   array1vd_t R;
   // @}
protected:
   /*! \name Internal functions */
   void init();
   void free();
   // @}
   /*! \name Constructors / Destructors */
   //! Default constructor
   uncoded();
   // @}
public:
   /*! \name Constructors / Destructors */
   uncoded(const fsm& encoder, const int tau);
   ~uncoded() { free(); };
   // @}

   // Codec operations
   void encode(const array1i_t& source, array1i_t& encoded);
   void translate(const array1vd_t& ptable);
   void softdecode(array1vd_t& ri);
   void softdecode(array1vd_t& ri, array1vd_t& ro);

   // Codec information functions - fundamental
   int input_block_size() const { return tau; };
   int output_block_size() const { return tau; };
   int num_inputs() const { return encoder->num_inputs(); };
   int num_outputs() const { return encoder->num_outputs(); };
   int tail_length() const { return 0; };
   int num_iter() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(uncoded);
};

}; // end namespace

#endif

