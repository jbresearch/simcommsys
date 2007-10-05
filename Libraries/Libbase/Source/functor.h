#ifndef __functor_h
#define __functor_h

/*
  Version 1.00 (25 Apr 2007)
  * Initial version, defining a functor class for functions that take no argmuents
    and return nothing.
*/

namespace libbase {

// abstract base class
class functor {
public:
   virtual void operator()(void)=0;  // call using operator
   virtual void call(void)=0;        // call using function
};

// derived template class
template <class T> class specificfunctor : public functor {
private:
   void (T::*fptr)(void);           // pointer to member function
   T* object;                       // pointer to object
public:
   // constructor - takes pointer to an object and pointer to a member function
   specificfunctor(T* _object, void(T::*_fptr)(void))  { object = _object;  fptr=_fptr; };
   // override calling operators
   virtual void operator()(void)  { (*object.*fptr)();};
   virtual void call(void)        { (*object.*fptr)();};
};

}; // end namespace

#endif
