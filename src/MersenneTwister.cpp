#include "MersenneTwister.hpp"

uint64_t mt_index;
std::array<uint64_t, MT_LEN> mt_buffer;

void mt_init(unsigned int seed) {

  srand(seed);

  for (int i = 0; i < MT_LEN; i++) {
    mt_buffer[i] = (static_cast<uint64_t>(rand()));
  }

  mt_index = 0;
}

void mt_read(std::ifstream &input) {
  input >> mt_index;

  for (int i = 0; i < MT_LEN; i++) {
    input >> mt_buffer[i];
  }
}

void mt_write(std::ofstream &output) {
  output << mt_index << '\n';

  for (int i = 0; i < MT_LEN; i++) {
    output << mt_buffer[i] << '\n';
  }
}

auto mt_random() -> uint64_t {
  uint64_t *b = mt_buffer.data();
  uint64_t idx = mt_index;
  uint64_t s = 0;
  int i = 0;

  if (idx == MT_LEN * sizeof(uint64_t)) {
    idx = 0;
    i = 0;
    for (; i < MT_IB; i++) {
      s = TWIST(b, i, i + 1);
      b[i] = b[i + MT_IA] ^ (s >> 1) ^ MAGIC(s);
    }
    for (; i < MT_LEN - 1; i++) {
      s = TWIST(b, i, i + 1);
      b[i] = b[i - MT_IB] ^ (s >> 1) ^ MAGIC(s);
    }

    s = TWIST(b, MT_LEN - 1, 0);
    b[MT_LEN - 1] = b[MT_IA - 1] ^ (s >> 1) ^ MAGIC(s);
  }
  mt_index = idx + sizeof(uint64_t);
  return *reinterpret_cast<uint64_t *>(reinterpret_cast<unsigned char *>(b) + idx);
}
