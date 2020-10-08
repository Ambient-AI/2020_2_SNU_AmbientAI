/*
  max, min: Representative dataset을 통해 얻은 float 값의 min max
  quant_min, quant_max: -128, 127
*/
void GetAsymmetricQuantizationParams(
    float min, float max, const int quant_min, const int quant_max,
    QuantizationParametersT* quantization_params) {
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  // Adjust the boundaries to guarantee 0 is included.
  min = std::min(static_cast<float>(min), 0.0f);
  max = std::max(static_cast<float>(max), 0.0f);
  const float scale = (max - min) / (quant_max_float - quant_min_float);
  // Scale can be zero if min and max are exactly 0.0f.
  float zero_point_from_min = quant_min_float;
  if (scale != 0) {
    zero_point_from_min = quant_min_float - min / scale;
  }
  int64_t zero_point;
  if (zero_point_from_min < quant_min_float) {
    zero_point = static_cast<int64_t>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    zero_point = static_cast<int64_t>(quant_max);
  } else {
    zero_point = static_cast<int64_t>(std::round(zero_point_from_min));
  }
  quantization_params->min = std::vector<float>(1, min);
  quantization_params->max = std::vector<float>(1, max);
  quantization_params->scale = std::vector<float>(1, scale);
  quantization_params->zero_point = std::vector<int64_t>(1, zero_point);
}