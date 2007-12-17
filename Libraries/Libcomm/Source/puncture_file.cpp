/*!
   \file

   \par Version Control:
   - $Revision$
   - $Date$
   - $Author$
*/

#include "puncture_file.h"
#include <stdio.h>
#include <string.h>
#include <sstream>

namespace libcomm {

const libbase::serializer puncture_file::shelper("puncture", "file", puncture_file::create);

// constructor / destructor

puncture_file::puncture_file(const char *fname, const int tau, const int sets)
   {
   // initialise the pattern matrix
   pattern.init(tau,sets);
   // load the puncturing matrix from the supplied file
   const char *fs = strrchr(fname, libbase::DIR_SEPARATOR);
   const char *fp = (fs==NULL) ? fname : fs+1;
   filename = fp;

   FILE *file = fopen(fname, "rb");
   if(file == NULL)
      {
      std::cerr << "FATAL ERROR (puncture): Cannot open puncturing pattern file (" << fname << ").\n";
      exit(1);
      }
   // note ordering within file is different from matrix!
   for(int s=0; s<sets; s++)
      for(int t=0; t<tau; t++)
         {
         int temp;
         while(fscanf(file, "%d", &temp) == 0)
            fscanf(file, "%*[^\n]\n");
         pattern(t,s) = temp != 0;
         }
   fclose(file);
   // fill-in remaining variables
   init(pattern);
   }

// description output

std::string puncture_file::description() const
   {
   std::ostringstream sout;
   sout << "Punctured (" << filename << ", " << get_outputs() << "/" << get_inputs() << ")";
   return sout.str();
   }

// object serialization - saving

std::ostream& puncture_file::serialize(std::ostream& sout) const
   {
   sout << filename << "\n";
   sout << pattern;
   return sout;
   }

// object serialization - loading

std::istream& puncture_file::serialize(std::istream& sin)
   {
   sin >> filename;
   sin >> pattern;
   init(pattern);
   return sin;
   }

}; // end namespace
