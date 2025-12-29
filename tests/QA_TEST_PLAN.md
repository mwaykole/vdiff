# dfastllm Comprehensive QA Test Plan

## Executive Summary
This document outlines all test cases required for production-ready quality assurance of the dfastllm hybrid diffusion-AR inference engine.

**Total Test Cases: 150+**  
**Categories: 12**

---

## Test Categories Overview

| Category | Test Cases | Priority |
|----------|------------|----------|
| 1. Configuration | 15 | Critical |
| 2. Engine Initialization | 12 | Critical |
| 3. Hybrid Engine | 18 | Critical |
| 4. Entropy Controller | 12 | High |
| 5. MoR Decoder | 10 | High |
| 6. APD Decoder | 10 | High |
| 7. Continuous Batching | 15 | Critical |
| 8. API Endpoints | 20 | Critical |
| 9. Generation Quality | 15 | High |
| 10. Performance | 12 | High |
| 11. Edge Cases | 18 | Medium |
| 12. Security & Reliability | 13 | Critical |

---

## 1. Configuration Tests (15 cases)

### 1.1 DFastLLMConfig
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| CFG-001 | Create config with defaults | All defaults applied |
| CFG-002 | Create config with custom values | Custom values applied |
| CFG-003 | Create config from environment | Env vars override defaults |
| CFG-004 | Invalid model path | Raise ValidationError |
| CFG-005 | Invalid dtype value | Use "auto" as fallback |
| CFG-006 | Negative max_model_len | Raise ValueError |
| CFG-007 | Zero diffusion_steps | Raise ValueError |
| CFG-008 | tensor_parallel_size > available GPUs | Raise error or warn |
| CFG-009 | Serialization to_dict() | All fields serialized |
| CFG-010 | Deserialization from_dict() | Config reconstructed |

### 1.2 BaseConfig Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| CFG-011 | Inheritance from BaseConfig | to_dict() works |
| CFG-012 | Config validation | validate() called |
| CFG-013 | Enum serialization | Enum values as strings |
| CFG-014 | Optional fields None handling | None serialized correctly |
| CFG-015 | Config immutability | Fields can be updated |

---

## 2. Engine Initialization Tests (12 cases)

### 2.1 Model Loading
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENG-001 | Load valid HuggingFace model | Model loaded successfully |
| ENG-002 | Load non-existent model | ModelLoadError raised |
| ENG-003 | Load with trust_remote_code=False | Fail for remote code models |
| ENG-004 | Load with trust_remote_code=True | Success |
| ENG-005 | Load with dtype=float16 | Model in float16 |
| ENG-006 | Load with dtype=bfloat16 | Model in bfloat16 |

### 2.2 Tokenizer Loading
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENG-007 | Load tokenizer with model | Tokenizer loaded |
| ENG-008 | Custom tokenizer path | Custom tokenizer used |
| ENG-009 | Missing tokenizer | Use model tokenizer |

### 2.3 Engine State
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENG-010 | Initial state | UNINITIALIZED |
| ENG-011 | After init | READY |
| ENG-012 | After shutdown | STOPPED |

---

## 3. Hybrid Engine Tests (18 cases)

### 3.1 HybridMode Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| HYB-001 | DEER mode initialization | Mode set to deer |
| HYB-002 | SPEC_DIFF mode initialization | Mode set to spec_diff |
| HYB-003 | SEMI_AR mode initialization | Mode set to semi_ar |
| HYB-004 | ADAPTIVE mode initialization | Mode set to adaptive |
| HYB-005 | Invalid mode string | Raise ValueError |
| HYB-006 | Mode from environment variable | Mode applied |

### 3.2 HybridConfig Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| HYB-007 | Default draft_block_size | 8 tokens |
| HYB-008 | Custom draft_block_size | Custom value applied |
| HYB-009 | draft_block_size=0 | Raise ValueError |
| HYB-010 | verification_threshold range | 0.0 to 1.0 |
| HYB-011 | adaptive=True enables dynamic adjustment | Params adjusted at runtime |

### 3.3 HybridStats Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| HYB-012 | Initial stats all zeros | All counters = 0 |
| HYB-013 | update() increments correctly | Counters incremented |
| HYB-014 | draft_acceptance_rate calculation | accepted / (accepted + rejected) |
| HYB-015 | to_dict() serialization | All stats in dict |
| HYB-016 | reset() clears stats | All counters = 0 |

### 3.4 Integration Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| HYB-017 | Generate with DEER mode | Valid output |
| HYB-018 | Generate with adaptive mode switching | Mode switches based on context |

---

## 4. Entropy Controller Tests (12 cases)

### 4.1 EntropyConfig Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENT-001 | Default thresholds | high=2.0, low=0.5 |
| ENT-002 | Custom thresholds | Custom values applied |
| ENT-003 | high_threshold <= low_threshold | Raise ValueError |
| ENT-004 | Strategy COMBINED | Strategy applied |
| ENT-005 | Strategy DRAFT_LENGTH | Affects draft length only |

### 4.2 EntropyStats Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENT-006 | Initial stats | All zeros |
| ENT-007 | to_dict() percentages | Correct percentages |

### 4.3 EntropyComputer Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENT-008 | compute() on uniform logits | Max entropy |
| ENT-009 | compute() on peaked logits | Low entropy |
| ENT-010 | compute_normalized() | Values 0-1 |
| ENT-011 | compute_top_k() | Top-k entropy |

### 4.4 Controller Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| ENT-012 | get_draft_length() adaptive | Longer for low entropy |

---

## 5. MoR Decoder Tests (10 cases)

### 5.1 MoRConfig Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| MOR-001 | Default recursion range | 1 to 4 |
| MOR-002 | Custom recursion range | Custom values |
| MOR-003 | min > max recursions | Raise ValueError |
| MOR-004 | enabled=False | Skip MoR processing |

### 5.2 MoRStats Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| MOR-005 | Token difficulty tracking | easy/medium/hard counts |
| MOR-006 | avg_recursions_per_token | Correct average |
| MOR-007 | compute_saved_pct | Correct percentage |

### 5.3 Routing Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| MOR-008 | High confidence → min recursions | 1 recursion |
| MOR-009 | Low confidence → max recursions | 4 recursions |
| MOR-010 | Medium confidence → proportional | 2-3 recursions |

---

## 6. APD Decoder Tests (10 cases)

### 6.1 APDConfig Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| APD-001 | Default max_parallel_tokens | 8 |
| APD-002 | Custom max_parallel_tokens | Custom value |
| APD-003 | acceptance_threshold range | 0.0 to 1.0 |
| APD-004 | dllm_weight and ar_weight | Correct weights |

### 6.2 APD Functionality Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| APD-005 | Parallel token generation | Multiple tokens/step |
| APD-006 | Token verification | Reject low-confidence tokens |
| APD-007 | Fallback to sequential | When parallel fails |
| APD-008 | enabled=False | Sequential generation |

### 6.3 APDStats Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| APD-009 | parallel_token_count | Correct count |
| APD-010 | acceptance_rate | Correct percentage |

---

## 7. Continuous Batching Tests (15 cases)

### 7.1 BatchedRequest Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| BAT-001 | Create request with defaults | Request created |
| BAT-002 | Priority ordering | CRITICAL > HIGH > NORMAL > LOW |
| BAT-003 | Arrival time ordering | Earlier requests first |

### 7.2 BatcherConfig Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| BAT-004 | Default max_batch_size | 8 |
| BAT-005 | Custom batch configuration | Custom values applied |
| BAT-006 | max_batch_size=0 | Raise ValueError |

### 7.3 ContinuousBatcher Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| BAT-007 | Submit single request | Request queued |
| BAT-008 | Submit multiple requests | Batch formed |
| BAT-009 | Batch size limit respected | Max batch not exceeded |
| BAT-010 | Wait time trigger | Batch sent after wait |

### 7.4 PrefixCache Tests
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| BAT-011 | Cache hit | Cached value returned |
| BAT-012 | Cache miss | None returned |
| BAT-013 | LRU eviction | Oldest item evicted |
| BAT-014 | min_prefix_length filter | Short prefixes not cached |
| BAT-015 | get_stats() | Correct hit rate |

---

## 8. API Endpoint Tests (20 cases)

### 8.1 Health Endpoints
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| API-001 | GET /health | 200 OK |
| API-002 | GET /health when uninitialized | 503 |
| API-003 | GET /ready | 200 when ready |
| API-004 | GET /metrics | Prometheus format |

### 8.2 Completions API
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| API-005 | POST /v1/completions valid | 200 with completion |
| API-006 | POST /v1/completions stream=true | SSE stream |
| API-007 | POST /v1/completions missing prompt | 400 error |
| API-008 | POST /v1/completions max_tokens=0 | Error or empty |
| API-009 | POST /v1/completions temperature range | 0-2 accepted |
| API-010 | POST /v1/completions stop sequences | Generation stops |

### 8.3 Chat Completions API
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| API-011 | POST /v1/chat/completions valid | 200 with message |
| API-012 | POST /v1/chat/completions stream | SSE stream |
| API-013 | POST /v1/chat/completions empty messages | 400 error |
| API-014 | POST /v1/chat/completions system message | System prompt applied |
| API-015 | POST /v1/chat/completions multi-turn | Context maintained |

### 8.4 Models API
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| API-016 | GET /v1/models | List of models |
| API-017 | GET /v1/models/{id} | Model details |

### 8.5 Error Handling
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| API-018 | Invalid JSON body | 400 with error |
| API-019 | Unsupported endpoint | 404 |
| API-020 | Server overloaded | 503 with retry-after |

---

## 9. Generation Quality Tests (15 cases)

### 9.1 Output Validity
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| GEN-001 | Generate coherent text | Grammatically correct |
| GEN-002 | Generate with prompt | Contextually relevant |
| GEN-003 | Generate code | Valid syntax |
| GEN-004 | Generate math | Correct calculations |

### 9.2 Sampling Parameters
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| GEN-005 | temperature=0 deterministic | Same output each time |
| GEN-006 | temperature=1.0 varied | Different outputs |
| GEN-007 | top_p=0.9 | Nucleus sampling |
| GEN-008 | top_k=50 | Top-k sampling |
| GEN-009 | max_tokens respected | Output within limit |

### 9.3 Special Cases
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| GEN-010 | Unicode input | Correct handling |
| GEN-011 | Emoji input | Correct handling |
| GEN-012 | Mixed language | Appropriate response |
| GEN-013 | Very short prompt | Valid completion |
| GEN-014 | Very long prompt | Handle or truncate |
| GEN-015 | Context window boundary | Graceful handling |

---

## 10. Performance Tests (12 cases)

### 10.1 Throughput
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| PERF-001 | Single request throughput | >30 tokens/sec |
| PERF-002 | Batch throughput | >100 tokens/sec |
| PERF-003 | Concurrent requests (10) | Linear scaling |
| PERF-004 | Concurrent requests (50) | Graceful degradation |

### 10.2 Latency
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| PERF-005 | Time to first token (TTFT) | <500ms |
| PERF-006 | P50 latency | <1s for 50 tokens |
| PERF-007 | P95 latency | <2s for 50 tokens |
| PERF-008 | P99 latency | <5s for 50 tokens |

### 10.3 Resource Usage
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| PERF-009 | GPU memory utilization | <90% |
| PERF-010 | GPU memory leak check | No growth over time |
| PERF-011 | CPU utilization | Reasonable |
| PERF-012 | Memory (RAM) stability | No leaks |

---

## 11. Edge Case Tests (18 cases)

### 11.1 Input Edge Cases
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| EDGE-001 | Empty string input | Handle gracefully |
| EDGE-002 | Whitespace only input | Handle gracefully |
| EDGE-003 | Single character input | Valid completion |
| EDGE-004 | Max context length input | Process or truncate |
| EDGE-005 | Binary data in input | Error or filter |
| EDGE-006 | Special tokens in input | Handle correctly |

### 11.2 Parameter Edge Cases
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| EDGE-007 | temperature=-0.1 | Clamp to 0 or error |
| EDGE-008 | temperature=10.0 | Clamp to 2 or error |
| EDGE-009 | max_tokens=1 | Single token output |
| EDGE-010 | max_tokens=10000 | Limit to max_model_len |
| EDGE-011 | top_p=0.0 | Error or handle |
| EDGE-012 | top_p=1.0 | All tokens considered |

### 11.3 Concurrency Edge Cases
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| EDGE-013 | 100 concurrent requests | Queue or reject |
| EDGE-014 | Request during shutdown | Complete or fail gracefully |
| EDGE-015 | Request timeout | Timeout error returned |

### 11.4 Error Recovery
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| EDGE-016 | OOM during generation | Recover and respond |
| EDGE-017 | GPU error during inference | Recover and respond |
| EDGE-018 | Network interrupt | Clean failure |

---

## 12. Security & Reliability Tests (13 cases)

### 12.1 Authentication
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| SEC-001 | Valid API key | Request processed |
| SEC-002 | Invalid API key | 401 Unauthorized |
| SEC-003 | Missing API key (when required) | 401 Unauthorized |
| SEC-004 | API key in header vs query | Both work |

### 12.2 Rate Limiting
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| SEC-005 | Within rate limit | Requests processed |
| SEC-006 | Exceed rate limit | 429 Too Many Requests |
| SEC-007 | Rate limit reset | Requests resume |

### 12.3 Input Validation
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| SEC-008 | XSS in input | Sanitized or rejected |
| SEC-009 | SQL injection in input | Sanitized or rejected |
| SEC-010 | Path traversal in model path | Rejected |

### 12.4 Reliability
| ID | Test Case | Expected Result |
|----|-----------|-----------------|
| SEC-011 | Graceful shutdown | Pending requests complete |
| SEC-012 | Long-running stability (1hr) | No degradation |
| SEC-013 | Recovery after crash | State restored |

---

## Test Execution Matrix

### Priority Levels
- **P0 (Critical)**: Must pass for any release
- **P1 (High)**: Should pass for production release
- **P2 (Medium)**: Should pass for stable release
- **P3 (Low)**: Nice to have

### Environment Requirements
| Test Category | GPU Required | Network Required | Duration |
|---------------|--------------|------------------|----------|
| Configuration | No | No | <1 min |
| Engine Init | Yes | Yes | <5 min |
| Hybrid Engine | Yes | No | <2 min |
| API Tests | Yes | Yes | <5 min |
| Performance | Yes | No | <10 min |
| Stress Tests | Yes | No | <30 min |

---

## Sign-off Criteria

### Release Readiness
- [ ] All P0 tests passing
- [ ] ≥95% of P1 tests passing
- [ ] ≥90% of P2 tests passing
- [ ] No known critical bugs
- [ ] Performance benchmarks met

### Production Deployment
- [ ] Stability test (1 hour) passing
- [ ] Memory leak test passing
- [ ] API compatibility verified
- [ ] Documentation updated
