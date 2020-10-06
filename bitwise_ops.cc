inline int32_t MultiplyByQuantizedMultiplier(int64_t x,
                                             int32_t quantized_multiplier,
                                             int shift) {
  // Inputs:
  // - quantized_multiplier has fixed point at bit 31
  // - shift is -31 to +7 (negative for right shift)
  //
  // Assumptions: The following input ranges are assumed
  // - quantize_scale>=0  (the usual range is (1<<30) to (1>>31)-1)
  // - scaling is chosen so final scaled result fits in int32_t
  // - input x is in the range -(1<<47) <= x < (1<<47)
  assert(quantized_multiplier >= 0);
  assert(shift >= -31 && shift < 8);

  int32_t reduced_multiplier = (quantized_multiplier + (1 << 15)) >> 16;
  int total_shift = 15 - shift;
  x = (x * (int64_t)reduced_multiplier) + ((int64_t)1 << (total_shift - 1));
  int32_t result = x >> total_shift;
  return result;
}