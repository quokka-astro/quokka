#ifndef MERSENNETWISTER_HPP_
#define MERSENNETWISTER_HPP_
/*
   This is an implementation of the Mersenne Twister which was written by
   Michael Brundage. The original source code, which was copied and pasted into
   this file, can be found at
   http://qbrundage.com/michaelb/pubs/essays/random_number_generation, and a
   description of the Mersenne Twister random number generator can be found
   here: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html It has a
   claimed periodicity of 2^{19937}-1, which is pretty incredible.  The
   algorithm as-is is mainly intended for Monte Carlo realizations, and is not
   intended to be used for cryptography.  See the website for more information.

   This code was placed into the public domain by Michael Brundage, and was put
   into Enzo by Brian W. O'Shea on 11 December 2007.  It uses the system random
   number generator as an initial seed, and the user must specify a seed for the
   system random number generator.

*/
#include <array>
#include <bits/stdint-uintn.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

#define MT_LEN 624
#define MT_IA 397
#define MT_IB (MT_LEN - MT_IA)
#define UPPER_MASK 0x80000000
#define LOWER_MASK 0x7FFFFFFF
#define MATRIX_A 0x9908B0DF
#define TWIST(b, i, j) ((b)[i] & UPPER_MASK) | ((b)[j] & LOWER_MASK)
#define MAGIC(s) (((s)&1) * MATRIX_A)

void mt_init(unsigned int seed);

void mt_read(std::ifstream &input);

void mt_write(std::ofstream &output);

auto mt_random() -> uint64_t;

#endif // MERSENNETWISTER_HPP_
