/*!
 * \file
 *
 * Copyright (c) 2010 Johann A. Briffa
 *
 * This file is part of SimCommSys.
 *
 * SimCommSys is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SimCommSys is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SimCommSys.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "serializer_libcomm.h"
#include "fsm.h"

#include <iostream>

namespace testfsm {

/*!
 * \brief   Test program for FSM objects
 * \author  Johann Briffa
 *
 * Serializes a FSM object from standard input; computes and displays
 * state table.
 */

int main(int argc, char *argv[])
   {
   using std::cin;
   using std::cout;
   using std::cerr;

   const libcomm::serializer_libcomm my_serializer_libcomm;

   // get a new fsm from stdin
   cerr << "Enter FSM details:" << std::endl;
   libcomm::fsm *encoder;
   cin >> encoder;
   cout << encoder->description() << std::endl;

   // compute and display state table
   cout << "PS\tIn\tOut\tNS" << std::endl;
   for (int ps = 0; ps < encoder->num_states(); ps++)
      for (int in = 0; in < encoder->num_input_combinations(); in++)
         {
         cout << ps << '\t';
         cout << in << '\t';
         encoder->reset(encoder->convert_state(ps));
         libbase::vector<int> ip = encoder->convert_input(in);
         cout << encoder->convert_output(encoder->step(ip)) << '\t';
         cout << encoder->convert_state(encoder->state()) << std::endl;
         }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testfsm::main(argc, argv);
   }
