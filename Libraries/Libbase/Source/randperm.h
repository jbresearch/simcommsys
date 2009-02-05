#ifndef __randperm_h
#define __randperm_h

#include "config.h"
#include "random.h"
#include "vector.h"

namespace libbase {

/*!
   \brief   Random Permutation Class.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Defines a random permutation of the set {0,1,..N-1} for given size N.
*/

class randperm {
private:
   /*! \name Object representation */
   //! Table to hold permutation values
   vector<int> lut;
   // @}

public:
   /*! \name Constructors / Destructors */
   //! Default constructor
   randperm() {};
   //! Principal constructor
   randperm(const int N, random& r) { init(N, r); }
   //! Virtual destructor
   virtual ~randperm() {};
   // @}

   /*! \name Random permutation interface */
   /*! \brief Permutation setup function
      \param N Size of permutation
      \param r Random generator to use in creating permutation
   
      Sets up a random permutation of the set {0,1,..N-1} for given size N.
   */
   void init(const int N, random& r);
   //! Return indexed value
   int operator() (const int i) const { return lut(i); };
   // @}

   /*! \name Informative functions */
   //! The size of the permutation
   int size() const { return lut.size(); };
   // @}
};

}; // end namespace

#endif
