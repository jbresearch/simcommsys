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
      int min_bs;
   public:
      
      /*Setters*/
      void setmin_bs(unsigned int min_bs){this->min_bs = min_bs;}
      /*Getters*/
      int getmin_bs(){return min_bs;}

      vector< vector< state_bs_storage > > state_bs_vector;

      b_storage(int no_cols)
         {
         state_bs_vector.resize(no_cols);
         min_bs = 0;
         }
   };

class state_output
   {
   private:
      unsigned int next_state, output;
   public:
      /*Setters*/
      void set_next_state(unsigned int next_state){ this->next_state = next_state; }
      void set_output(unsigned int output){ this->output = output; }
      /*Getters*/
      unsigned int get_next_state(void){ return next_state; }
      unsigned int get_output(void){ return output; }

      /*Constructor*/
      state_output()
         {
         next_state = 0;
         output = 0;
         }
   };

class dynamic_symbshift
   {
   private:
      unsigned int min, max;
   public:

      dynamic_symbshift(){ min = max = 0; }

      dynamic_symbshift(unsigned int min, unsigned int max)
         {
         this->min = min;
         this->max = max;
         }

      void setminmax(unsigned int min, unsigned int max)
         {
         this->min = min;
         this->max = max;
         }

      unsigned int getmin(){ return min; }
      unsigned int getmax(){ return max; }
   };

class levenshtein_storage
   {
   public:
      char location;
      double value;

      levenshtein_storage()
         {
         location = 0;
         value = 0.0;
         }
   };