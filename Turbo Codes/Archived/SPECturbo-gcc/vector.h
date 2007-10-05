#ifndef __vector_h
#define __vector_h

#include "config.h"
#include "vcs.h"
#include <iostream.h>
#include <stdlib.h>

extern const vcs vector_version;

template <class T> class vector {
   bool	initialised;
   int	xsize;
   T		*data;
   void validate(const int x);		// check that the suggested size is valid
   void alloc(const int x);		// blind - they don't validate the state/parameters!
   void free();				// ditto - free does not set 'initialised' on exit...
public:
   vector(const int x=0);		// constructor
   vector(const vector<T>& m);		// copy constructor
   ~vector();                               
   vector<T>& operator=(const vector<T>& m);	// vector copy
   void init(const int x);
   T& operator()(const int x);
   T operator()(const int x) const;
   int size() const { return xsize; };
};

// Private functions

template <class T> inline void vector<T>::validate(const int x)
   {
   if(x < 1)
      {
      cerr << "FATAL ERROR (vector): invalid vector size (" << x << ")\n";
      exit(1);
      }
   }
   
template <class T> inline void vector<T>::alloc(const int x)
   {
   xsize = x;
   data = new T[x];
   initialised = true;
   }

template <class T> inline void vector<T>::free()
   {
   delete[] data;
   }
   
// Public functions
   
template <class T> inline vector<T>::vector(const int x)
   {
   initialised = false;
   if(x!=0)		// special condition for an uninitialised vector
      {
      validate(x);
      alloc(x);
      }
   }

template <class T> inline vector<T>::vector(const vector<T>& m)
   {
   alloc(m.xsize);
   for(int i=0; i<m.xsize; i++)
      data[i] = m.data[i];
   }

template <class T> inline vector<T>::~vector()
   {
   if(initialised)
      free();
   }

template <class T> inline void vector<T>::init(const int x)
   {
   if(initialised)
      free();
   validate(x);
   alloc(x);
   }

template <class T> inline vector<T>& vector<T>::operator=(const vector<T>& m)
   {         
   // Destroy whatever we have
   if(initialised)
      free();
   // Initialise a new one by copying the given vector
   alloc(m.xsize);
   for(int i=0; i<m.xsize; i++)
      data[i] = m.data[i];
   return *this;
   }

template <class T> inline T& vector<T>::operator()(const int x) 
   {
   if(x<0 || x>=xsize)
      {
      cerr << "FATAL ERROR (vector): vector index out of range (" << x << ")\n";
      exit(1);
      }
   return data[x];
   }

template <class T> inline T vector<T>::operator()(const int x) const
   {
   if(x<0 || x>=xsize)
      {
      cerr << "FATAL ERROR (vector): vector index out of range (" << x << ")\n";
      exit(1);
      }
   return data[x];
   }


#endif

   
