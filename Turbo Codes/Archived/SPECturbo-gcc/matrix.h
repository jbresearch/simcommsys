#ifndef __matrix_h
#define __matrix_h

#include "config.h"
#include "vcs.h"
#include <iostream.h>
#include <stdlib.h>

extern const vcs matrix_version;

template <class T> class matrix {
   bool	initialised;
   int		xsize, ysize;
   T		**data;
   void validate(const int x, const int y);		// check that the suggested size is valid
   void alloc(const int x, const int y);		// blind - they don't validate the state/parameters!
   void free();							// ditto - free does not set 'initialised' on exit...
public:
   matrix(const int x=0, const int y=0);		// constructor
   matrix(const matrix<T>& m);			// copy constructor
   ~matrix();                               
   matrix<T>& operator=(const matrix<T>& m);	// matrix copy
   void init(const int x, const int y);
   int x_size() const { return xsize; };
   int y_size() const { return ysize; };
   T& operator()(const int x, const int y);
   T operator()(const int x, const int y) const;
};

// Private functions

template <class T> inline void matrix<T>::validate(const int x, const int y)
   {
   if(x < 1 || y < 1)
      {
      cerr << "FATAL ERROR (matrix): invalid matrix size (" << x << ", " << y << ")\n";
      exit(1);
      }
   }
   
template <class T> inline void matrix<T>::alloc(const int x, const int y)
   {
   typedef T* Tp;
   xsize = x;
   ysize = y;
   data = new Tp[x];
   for(int i=0; i<x; i++)
      data[i] = new T[y];
   initialised = true;
   }

template <class T> inline void matrix<T>::free()
   {
   for(int i=0; i<xsize; i++)
      delete[] data[i];
   delete[] data;
   }
   
// Public functions
   
template <class T> inline matrix<T>::matrix(const int x, const int y)
   {
   initialised = false;
   if(x!=0 || y!=0)		// special condition for an uninitialised matrix
      {
      validate(x, y);
      alloc(x, y);
      }
   }

template <class T> inline matrix<T>::matrix(const matrix<T>& m)
   {
   alloc(m.xsize, m.ysize);
   for(int i=0; i<m.xsize; i++)
      for(int j=0; j<m.ysize; j++)
         data[i][j] = m.data[i][j];
   }

template <class T> inline matrix<T>::~matrix()
   {
   if(initialised)
      free();
   }

template <class T> inline void matrix<T>::init(const int x, const int y)
   {
   if(initialised)
      free();
   validate(x, y);
   alloc(x, y);
   }

template <class T> inline matrix<T>& matrix<T>::operator=(const matrix<T>& m)
   {         
   // Destroy whatever we have
   if(initialised)
      free();
   // Initialise a new one by copying the given matrix
   alloc(m.xsize, m.ysize);
   for(int i=0; i<m.xsize; i++)
      for(int j=0; j<m.ysize; j++)
         data[i][j] = m.data[i][j];
   return *this;
   }

template <class T> inline T& matrix<T>::operator()(const int x, const int y) 
   {
   if(x<0 || y<0 || x>=xsize || y>=ysize)
      {
      cerr << "FATAL ERROR (matrix): matrix index out of range (" << x << ", " << y << ")\n";
      exit(1);
      }
   return data[x][y];
   }

template <class T> inline T matrix<T>::operator()(const int x, const int y) const
   {
   if(x<0 || y<0 || x>=xsize || y>=ysize)
      {
      cerr << "FATAL ERROR (matrix): matrix index out of range (" << x << ", " << y << ")\n";
      exit(1);
      }
   return data[x][y];
   }

// 3D Matrices

template <class T> class matrix3 {
   bool	initialised;
   int		xsize, ysize, zsize;
   T		***data;
   void validate(const int x, const int y, const int z);
   void alloc(const int x, const int y, const int z);
   void free();
public:
   matrix3(const int x=0, const int y=0, const int z=0);
   matrix3(const matrix3<T>& m);
   ~matrix3();
   matrix3<T>& operator=(const matrix3<T>& m);
   void init(const int x, const int y, const int z);
   int x_size() const { return xsize; };
   int y_size() const { return ysize; };
   int z_size() const { return zsize; };
   T& operator()(const int x, const int y, const int z);
   T operator()(const int x, const int y, const int z) const;
};

// Private functions

template <class T> inline void matrix3<T>::validate(const int x, const int y, const int z)
   {
   if(x < 1 || y < 1|| z < 1)
      {
      cerr << "FATAL ERROR (matrix): invalid matrix size (" << x << ", " << y << ", " << z << ")\n";
      exit(1);
      }
   }
   
template <class T> inline void matrix3<T>::alloc(const int x, const int y, const int z)
   {
   typedef T** Tpp;
   typedef T* Tp;
   xsize = x;
   ysize = y;
   zsize = z;
   data = new Tpp[x];
   for(int i=0; i<x; i++)
      {
      data[i] = new Tp[y];
      for(int j=0; j<y; j++)
         data[i][j] = new T[z];
      }
   initialised = true;
   }

template <class T> inline void matrix3<T>::free()
   {
   for(int i=0; i<xsize; i++)
      {
      for(int j=0; j<ysize; j++)
         delete[] data[i][j];
      delete[] data[i];      
      }
   delete[] data;
   }
   
// Public functions
   
template <class T> inline matrix3<T>::matrix3(const int x, const int y, const int z)
   {
   initialised = false;
   if(x!=0 || y!=0 || z!=0)	// special condition for an uninitialised matrix
      {
      validate(x, y, z);
      alloc(x, y, z);
      }
   }

template <class T> inline matrix3<T>::matrix3(const matrix3<T>& m)
   {
   alloc(m.xsize, m.ysize, m.zsize);
   for(int i=0; i<m.xsize; i++)
      for(int j=0; j<m.ysize; j++)
         for(int k=0; k<m.zsize; k++)
            data[i][j][k] = m.data[i][j][k];
   }

template <class T> inline matrix3<T>::~matrix3()
   {
   if(initialised)
      free();
   }

template <class T> inline void matrix3<T>::init(const int x, const int y, const int z)
   {
   if(initialised)
      free();
   validate(x, y, z);
   alloc(x, y, z);
   }

template <class T> inline T& matrix3<T>::operator()(const int x, const int y, const int z)
   {
   if(x<0 || y<0 || z<0 || x>=xsize || y>=ysize || z>=zsize)
      {
      cerr << "FATAL ERROR (matrix): matrix index out of range (" << x << ", " << y << ", " << z << ")\n";
      exit(1);
      }
   return data[x][y][z];
   }

template <class T> inline T matrix3<T>::operator()(const int x, const int y, const int z) const
   {
   if(x<0 || y<0 || z<0 || x>=xsize || y>=ysize || z>=zsize)
      {
      cerr << "FATAL ERROR (matrix): matrix index out of range (" << x << ", " << y << ", " << z << ")\n";
      exit(1);
      }
   return data[x][y][z];
   }

#endif

   
