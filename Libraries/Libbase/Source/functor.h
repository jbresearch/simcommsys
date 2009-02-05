#ifndef __functor_h
#define __functor_h

namespace libbase {

/*!
   \brief   Base Function Pointer.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Abstract base functor class for functions that take no argmuents and return
   nothing.
*/

class functor {
public:
   /*! \name Constructors / Destructors */
   virtual ~functor() {};
   // @}

   /*! \name Function calling interface */
   //! Call using bracket operator notation
   virtual void operator()(void)=0;
   //! Call using functor method notation
   virtual void call(void)=0;
   // @}
};


/*!
   \brief   Function Pointer.
   \author  Johann Briffa

   \section svn Version Control
   - $Revision$
   - $Date$
   - $Author$

   Specific functor class for class member functions.
*/

template <class T>
class specificfunctor : public functor {
private:
   /*! \name User-defined parameters */
   T* object;                       //!< Pointer to object
   void (T::*fptr)(void);           //!< Pointer to member function
   // @}
public:
   /*! \name Constructors / Destructors */
   /*! \brief Main constructor
       \param _object Pointer to an object
       \param _fptr Pointer to a member function
   */
   specificfunctor(T* _object, void(T::*_fptr)(void))
      { object = _object;  fptr=_fptr; };
   // @}

   // Function calling interface
   void operator()(void)  { (*object.*fptr)(); };
   void call(void)        { (*object.*fptr)(); };
};

}; // end namespace

#endif
