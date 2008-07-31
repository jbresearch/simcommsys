#ifndef __map_straight_h
#define __map_straight_h

#include "mapper.h"

namespace libcomm {

/*!
   \brief   Mapper Interface.
   \author  Johann Briffa

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$

   This class defines a straight symbol mapper with:
   * forward transform from modulator
   * inverse transform from the various codecs.
*/

class map_straight : public mapper {
private:
   /*! \name User-defined parameters */
   int N;   //!< Number of possible values of each encoder output
   int M;   //!< Number of possible values of each modulation symbol
   int S;   //!< Number of possible values of each translation symbol
   // @}
   /*! \name Internal object representation */
   int s1;  //!< Number of modulation symbols per encoder output
   int s2;  //!< Number of modulation symbols per translation symbol
   // @}
public:
   // Vector mapper operations
   void transform(const libbase::vector<int>& in, libbase::vector<int>& out);
   void inverse(const libbase::matrix<double>& pin, libbase::matrix<double>& pout);

   // Setup functions
   void set_parameters(const int N, const int M, const int S);

   // Informative functions
   double rate() const { return 1; };

   // Description
   std::string description() const;

   // Serialization Support
   DECLARE_SERIALIZER(map_straight)
};

}; // end namespace

#endif
