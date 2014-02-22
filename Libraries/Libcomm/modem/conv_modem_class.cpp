#include "conv_modem_class.h"

void Gamma_Storage::print()
   {
   std::cout << state << " " << bitshift << " " << gamma << std::endl;
   }

void state_bs_storage::normalpha(double alpha_total)
   {
   alpha /= alpha_total;
   }

void state_bs_storage::normbeta(double beta_total)
   {
   beta /= beta_total;
   }