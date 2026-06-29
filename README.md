# GGUF Edge AI Evaluation Checklist

## ASR

### 1. Model Conversion
- [x] Convert OmniLingual checkpoint to Hugging Face format using https://github.com/ahmedadelattia/omnilingual_to_hf
- [x] Validate HF model inference
- [ ] Export FP16 GGUF model
- [ ] Generate quantized GGUF variants:
  - [ ] Q8_0
  - [ ] Q6_K
  - [ ] Q5_K_M
  - [ ] Q4_K_M
- [ ] Generate Imatrix variants

### 2. Benchmarking
- [ ] Evaluate baseline HF model on Paza datasets
- [ ] Evaluate all GGUF variants
- [ ] Measure:
  - [ ] WER/CER
  - [ ] Latency
  - [ ] Throughput
  - [ ] Peak RAM
  - [ ] Model size
- [ ] Identify best accuracy/memory/performance tradeoff

### 3. Edge Inference Runtime
- [ ] Implement memory-efficient C++ inference
- [ ] Implement memory-efficient Rust inference
- [ ] Benchmark CPU performance
- [ ] Benchmark memory usage
- [ ] Select production runtime

### 4. Android Deployment
- [ ] Build Android inference library
- [ ] Integrate GGUF models
- [ ] Add microphone input
- [ ] Add offline transcription
- [ ] Benchmark latency, RAM, and battery usage

### 5. Documentation
- [ ] Publish benchmark results
- [ ] Document conversion workflow
- [ ] Document deployment workflow
- [ ] Publish quantization recommendations

---

## TTS

- [ ] Convert model to HF format
- [ ] Convert HF model to GGUF
- [ ] Generate quantized variants
- [ ] Benchmark quality, latency, RAM, and model size
- [ ] Implement C++/Rust inference
- [ ] Deploy on Android
- [ ] Document results

---

## Translation

- [ ] Convert model to HF format
- [ ] Convert HF model to GGUF
- [ ] Generate quantized variants
- [ ] Benchmark quality, latency, RAM, and model size
- [ ] Implement C++/Rust inference
- [ ] Deploy on Android
- [ ] Document results

