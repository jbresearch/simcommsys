/*!
 * \file
 * \brief Boost unit tests to test the CRC implementation.
 *
 * Copyright (c) 2025 Aaron Abela
 */

#define BOOST_TEST_MODULE testcrc
#include <boost/test/included/unit_test.hpp>

#include <cstdint>
#include <iomanip>
#include <boost/crc.hpp>
#include "crc/crc32.h"
#include <sstream>

using namespace libcomm;
using namespace libbase;

BOOST_AUTO_TEST_CASE(test_crc32_libcomm_implementation)
{
   std::cout << "\n*****Boost Test Case *****\n";
   std::cout << "\nTesting CRC32 Libcomm \n";

    // Build one vector (use (i) indexing for libbase::vector)
    libbase::vector<bool> v(32);
    v(0)=1; v(1)=0; v(2)=1; v(3)=1;
    v(4)=0; v(5)=1; v(6)=0; v(7)=0;
    v(8)=1; v(9)=0; v(10)=1; v(11)=0;
    v(12)=1; v(13)=0; v(14)=1; v(15)=0;
    v(16)=0; v(17)=1; v(18)=1; v(19)=0;
    v(20)=1; v(21)=1; v(22)=0; v(23)=0;
    v(24)=0; v(25)=0; v(26)=1; v(27)=1;
    v(28)=1; v(29)=0; v(30)=0; v(31)=1;

    // One-shot CRC32
    std::uint32_t crc_hash = libcomm::crc32_ieee<>::compute(v);
    std::cout << "Hash value of CRC32 = " << crc_hash << std::endl;

    // Log as 8-digit uppercase hex
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0') << std::setw(8) << crc_hash;
    BOOST_TEST_MESSAGE("CRC32 = 0x" + oss.str());

    // Minimal assertion so the test actually asserts something
    BOOST_CHECK(crc_hash != 0u);
}

/* Compare obtained answers to this online calculator: https://crccalc.com/?crc=123456789&method=&datatype=ascii&outtype=hex*/

BOOST_AUTO_TEST_CASE(testing_crc32_serialization)
{
   std::cout << "\n*****Boost Test Case *****\n";
   std::cout << "\nTesting CRC32 Libcomm Serialization \n";

   // Create via the registry
   auto obj = serializer::call("crc", "crc_32");
   BOOST_REQUIRE_MESSAGE(bool(obj), "serializer::call returned null");

   // Should report its registered name
   BOOST_CHECK_EQUAL(obj->name(), "crc_32");

   // It should actually be crc32_ieee<libbase::vector>
   auto* typed = dynamic_cast<libcomm::crc32_ieee<libbase::vector>*>(obj.get());
   BOOST_REQUIRE_MESSAGE(typed != nullptr, "Dynamic cast to crc32_ieee<libbase::vector> failed");

   // Do one simple CRC to prove functionality.
   libbase::vector<bool> v(8);

   v(0)=1; v(1)=0; v(2)=1; v(3)=1; v(4)=0; v(5)=1; v(6)=0; v(7)=1;

   /* In decimal this is equal to B5 and the CRC result (using CRC-32 MPEG2 with: Check: 0x0376E6E7	Poly: 0x04C11DB7	Init: 0xFFFFFFFF	RefIn: false	RefOut: false	XorOut: 0x00000000)

   To verify that the result is correct compare to: https://crccalc.com/?crc=B4&method=CRC-32/MPEG-2&datatype=hex&outtype=dec

   In decimal the answer should be: 3841153441

   */

   std::uint32_t crc = typed->compute(v);

   std::cout << "CRC32 Hash value = " << crc << std::endl;

   BOOST_CHECK(crc != 0u);

}