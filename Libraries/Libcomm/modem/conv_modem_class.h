#include <iostream>
#include <vector>

class Gamma_Storage
   {
   private:
      unsigned int state, bitshift;
      double gamma;
   
   public:
      Gamma_Storage(unsigned int state, unsigned int bitshift, double gamma)
      {
      this->state = state;
      this->bitshift = bitshift;
      this->gamma = gamma;
      }

      Gamma_Storage()
         {
         state = bitshift = 0;
         gamma = 0.0;
         }
      
      /*Getters*/
      unsigned int getstate(){return state;}
      unsigned int getbitshift(){return bitshift;}
      double getgamma(){return gamma;}
      /*Setters*/
      void setstate(unsigned int state){this->state = state;}
      void setbitshift(unsigned int bitshift){this->bitshift = bitshift;}
      void setgamma(double gamma){this->gamma = gamma;}

      void print();
   };