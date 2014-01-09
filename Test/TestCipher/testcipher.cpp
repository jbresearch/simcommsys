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

#include "cputimer.h"
#include "crypto/elgamal.h"
#include "crypto/knowdisclog_zpschnorr.h"
#include "math/gmp_bigint.h"

#include <boost/program_options.hpp>
#include <list>
#include <fstream>
#include <sstream>
#include <iostream>

namespace testcipher {

/*!
 * \brief   Test class for running ElGamal cipher
 * \author  Johann Briffa
 *
 * The class takes a BigInteger template parameter.
 */

template <class BigInteger>
class testset {
public:
   //! Main process
   static void process()
      {
      testset<BigInteger> my_testset;

      // Stores key pairs and public shares.
      std::list<libbase::keypair<BigInteger> > shares;
      libbase::group<BigInteger> grp;
      BigInteger publicKey;

      // Load Existing Data
      std::ifstream groupfile("group_data.txt");
      groupfile >> std::hex >> std::noshowbase >> grp;

      my_testset.loadKeys(shares, 5);

      std::ifstream pubkeyfile("CombinedPublicKey.pubKey");
      pubkeyfile >> std::hex >> std::noshowbase >> publicKey;

      // Create ciphers. Change the first argument to adjust how
      // many ciphers to make
      my_testset.createCiphers(100, grp, publicKey);

      // Load the ciphers back up from the file
      std::list<libbase::ciphertext<BigInteger> > ciphers =
            my_testset.loadCiphers();

      // Do partial decryptions, combine and store recovered plaintext
      std::list<BigInteger> plaintexts;
      typedef typename std::list<libbase::ciphertext<BigInteger> >::iterator
            iterator;
      for (iterator it = ciphers.begin(); it != ciphers.end(); it++)
         {
         std::list<BigInteger> partialDecryptions = my_testset.decryptCipher(
               grp, shares, *it);
         plaintexts.push_back(libbase::elgamal<BigInteger>::combineShares(
               partialDecryptions, it->get_myr(), grp.get_p()));
         }

      // Save recovered plaintexts to file
      my_testset.saveRecoveredPlaintexts(plaintexts);

      // Compare two files of plaintexts (original and recovered) to check
      // they are the same
      std::cout << "Recovered Plain Texts Are Identical: ";
      std::cout << my_testset.comparePlainTexts("plaintexts.txt",
            "recoveredPlainText.txt") << std::endl;
      }

   std::list<BigInteger> decryptCipher(libbase::group<BigInteger> grp,
         std::list<libbase::keypair<BigInteger> > shares, libbase::ciphertext<
               BigInteger> cipherText)
      {
      std::list<BigInteger> partialDecryptions;
      typedef typename std::list<libbase::keypair<BigInteger> >::iterator
            iterator;
      for (iterator it = shares.begin(); it != shares.end(); it++)
         {
         partialDecryptions.push_back(libbase::elgamal<BigInteger>::decrypt(
               grp.get_p(), it->get_pri_key(), cipherText.get_gr()));

         std::cout << "Decryption Proof is: ";
         std::cout << libbase::equalitydisclog_zpschnorr<BigInteger>::verify(
               grp, libbase::elgamal<BigInteger>::createDecryptionProof(grp,
                     it->get_pri_key(), cipherText.get_gr())) << std::endl;
         }
      return partialDecryptions;
      }

   void createKeys(std::list<libbase::keypair<BigInteger> > shares, std::list<
         BigInteger> publicShares, libbase::group<BigInteger> grp)
      {
      for (int i = 0; i < 5; i++)
         {
         libbase::keypair<BigInteger> keyPair = libbase::keygenerator<
               BigInteger>::generateKeyShare(grp);

         libbase::knowdisclog_zpschnorr<BigInteger> proof =
               libbase::knowdisclog_zpschnorr<BigInteger>::createProof(grp,
                     keyPair.get_pri_key(), keyPair.get_pub_key());

         std::cout << "Proof is: ";
         std::cout << libbase::knowdisclog_zpschnorr<BigInteger>::verifyProof(
               grp, proof);

         shares.push_back(keyPair);
         publicShares.push_back(keyPair.get_pub_key());
         }
      }

   void saveKeys(const std::list<libbase::keypair<BigInteger> >& shares)
      {
      typedef typename std::list<libbase::keypair<BigInteger> >::const_iterator
            iterator;
      int i = 0;
      for (iterator it = shares.begin(); it != shares.end(); i++, it++)
         {
         std::ostringstream s;
         s << "KeyPairShare_" << i << ".keypair";
         std::ofstream kpfile(s.str().c_str());
         kpfile << std::hex << std::noshowbase << *it;
         }
      }

   void loadKeys(std::list<libbase::keypair<BigInteger> >& shares,
         int sharesToLoad)
      {
      for (int i = 0; i < sharesToLoad; i++)
         {
         std::ostringstream s;
         s << "KeyPairShare_" << i << ".keypair";
         std::ifstream kpfile(s.str().c_str());
         libbase::keypair<BigInteger> kp;
         kpfile >> std::hex >> std::noshowbase >> kp;
         shares.push_back(kp);
         }
      }

   void createCiphers(int numberToCreate, libbase::group<BigInteger> grp,
         BigInteger publicKey)
      {
      // Get Random plain text that is in the subgroup
      std::ofstream cipher_file("ciphers.txt");
      std::ofstream plain_file("plaintexts.txt");
      cipher_file << std::hex << std::noshowbase;
      plain_file << std::hex << std::noshowbase;
      for (int x = 0; x < numberToCreate; x++)
         {
         BigInteger plainText = grp.sample();
         libbase::ciphertext<BigInteger> cipherText = libbase::elgamal<
               BigInteger>::encrypt(plainText, grp, publicKey);
         cipher_file << cipherText.get_gr() << "," << cipherText.get_myr()
               << std::endl;
         plain_file << plainText << std::endl;
         }
      }

   void saveRecoveredPlaintexts(std::list<BigInteger> recovered)
      {
      std::ofstream file("recoveredPlainText.txt");
      file << std::hex << std::noshowbase;
      typedef typename std::list<BigInteger>::iterator iterator;
      for (iterator it = recovered.begin(); it != recovered.end(); it++)
         {
         file << *it << std::endl;
         }
      }

   std::list<libbase::ciphertext<BigInteger> > loadCiphers()
      {
      std::ifstream file("ciphers.txt");
      std::list<libbase::ciphertext<BigInteger> > ciphers;
      std::string currentLine;
      while (getline(file, currentLine))
         {
         BigInteger bigIntOne, bigIntTwo;
         char c;
         std::istringstream is(currentLine);
         is >> std::hex >> std::noshowbase;
         is >> bigIntOne >> c >> bigIntTwo;
         ciphers.push_back(
               libbase::ciphertext<BigInteger>(bigIntOne, bigIntTwo));
         }
      return ciphers;
      }

   bool comparePlainTexts(std::string originalFile, std::string recoveredFile)
      {
      std::ifstream original(originalFile.c_str());
      std::ifstream recovered(recoveredFile.c_str());
      std::string currentOriginalLine, currentRecoveredLine;
      while (getline(original, currentOriginalLine) && getline(recovered,
            currentRecoveredLine))
         {
         if (currentOriginalLine.compare(currentRecoveredLine) != 0)
            {
            return false;
            }
         }
      if (currentOriginalLine.empty())
         {
         if (getline(recovered, currentRecoveredLine))
            {
            return false;
            }
         }
      else if (currentRecoveredLine.empty())
         {
         if (getline(original, currentOriginalLine))
            {
            return false;
            }
         }
      return true;
      }
};

// explicit instantiations

#ifdef USE_GMP
template class testset<libbase::gmp_bigint>;
#endif

/*!
 * \brief   Test for ElGamal cipher
 * \author  Johann Briffa
 */

int main(int argc, char *argv[])
   {
   libbase::cputimer tmain("Main timer");

   // Set up user parameters
   namespace po = boost::program_options;
   po::options_description desc("Allowed options");
   desc.add_options()("help,h", "print this help message");
   desc.add_options()("type,t", po::value<std::string>()->default_value("gmp"),
         "big-integer type");
   po::variables_map vm;
   po::store(po::parse_command_line(argc, argv, desc), vm);
   po::notify(vm);

   // Validate user parameters
   if (vm.count("help"))
      {
      std::cerr << desc << std::endl;
      return 1;
      }
   // Shorthand access for parameters
   const std::string type = vm["type"].as<std::string> ();

   // Main process
   if (type == "gmp")
#ifdef USE_GMP
      testset<libbase::gmp_bigint>::process();
#else
      std::cerr << "GNU MP support not compiled in." << std::endl;
#endif
   else
      {
      std::cerr << "Unrecognized big-integer type: " << type << std::endl;
      return 1;
      }

   return 0;
   }

} // end namespace

int main(int argc, char *argv[])
   {
   return testcipher::main(argc, argv);
   }
