#include <iostream>
#include <vector>

using namespace std;

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

class state_bs_storage
   {
   private:
      double alpha,beta;
   public:
      vector<Gamma_Storage> gamma;
      /*Getters*/
      double getalpha(){return alpha;}
      double getbeta(){return beta;}
      /*Setters*/
      void setalpha(double alpha){this->alpha += alpha;}
      void setbeta(double beta){this->beta += beta;}
      /*Normalizers*/
      void normalpha(double alpha_total);
      void normbeta(double beta_total);

      /*Constructors*/
      state_bs_storage()
         {
         this->alpha = 0;
         this->beta = 0;
         }

      state_bs_storage(double alpha)
         {
         this->alpha = alpha;
         this->beta = 0;
         }
   };

class b_storage
   {
   private: 
      unsigned int min_bs;
   public:
      
      /*Setters*/
      void setmin_bs(unsigned int min_bs){this->min_bs = min_bs;}
      /*Getters*/
      unsigned int getmin_bs(){return min_bs;}

      vector< vector< state_bs_storage > > state_bs_vector;

      b_storage(int no_cols)
         {
         state_bs_vector.resize(no_cols);
         min_bs = 0;
         }
      //b_storage(unsigned int no_cols)
      //   {
      //   //vector< vector<double> > matrix;
      //   //now we have an empty 2D-matrix of size (0,0). Resizing it with one single command:
      //   //matrix.resize( num_of col , vector<double>( num_of_row , init_value ) );
      //   // and we are good to go ... 
      //   state_bs_vector.resize(no_cols);
      //   }
   };