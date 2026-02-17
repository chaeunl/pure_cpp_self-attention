## Pure C++ implementation for self-attention

Pure C++ (no external libraries) implementations of multi-head self-attention, including a standard version (`main.cpp`) and a FlashAttention-v1 version (`flashattn.cpp`). Weights and inputs are loaded from a pretrained `bert-base-uncased` model.

### Prerequisites

A C++17-compatible compiler:

- **Linux / macOS**: GCC 8+ or Clang 7+
- **Windows**: MinGW-w64 (e.g. via [MSYS2](https://www.msys2.org/) UCRT64 environment) or Visual Studio 2017+

### Build

```bash
g++ -std=c++17 -O2 -o main.exe main.cpp
g++ -std=c++17 -O2 -o flashattn.exe flashattn.cpp
```

> **Windows note**: If `g++` is not on your `PATH`, launch the MSYS2 UCRT64 terminal or add its `bin/` directory to your `PATH` first.

### Usage

#### Benchmark mode (default)

Runs the forward pass multiple times and prints elapsed time in milliseconds.

```bash
./main.exe
./flashattn.exe
```

Output:
```
total time: 440
```

#### Verify mode

Runs a single forward pass and compares each intermediate tensor against PyTorch reference outputs stored in `bert-base-uncased/reference/`. Reports `[PASS]` or `[FAIL]` with max and mean absolute error (tolerance: `1e-4`).

```bash
./main.exe --verify
./flashattn.exe --verify
```

Output example (`main.exe`):
```
[PASS] query_states   max_err=7.15256e-06  mean_err=4.08236e-07
[PASS] key_states     max_err=6.19888e-06  mean_err=4.03036e-07
[PASS] value_states   max_err=5.00679e-06  mean_err=2.73196e-07
[PASS] attn_probs     max_err=1.37091e-06  mean_err=6.69274e-09
[PASS] attn_output    max_err=3.8147e-06   mean_err=1.7728e-07
[PASS] output_states  max_err=2.78652e-06  mean_err=1.91397e-07
```

`flashattn.exe` checks the same tensors except `attn_probs`, since FlashAttention does not materialize the full attention matrix.

### Reference data

The `bert-base-uncased/` directory contains binary dumps from a PyTorch `bert-base-uncased` model, organized into two subdirectories:

#### `params/` — Model parameters

| File | Shape | Description |
|---|---|---|
| `query.weight`, `query.bias` | (768, 768), (768) | Q projection parameters |
| `key.weight`, `key.bias` | (768, 768), (768) | K projection parameters |
| `value.weight`, `value.bias` | (768, 768), (768) | V projection parameters |
| `output.weight`, `output.bias` | (768, 768), (768) | O projection parameters |

#### `reference/` — Input and ground-truth intermediate tensors

| File | Shape | Description |
|---|---|---|
| `hidden_states` | (2, 128, 768) | Input to the attention layer |
| `query_states` | (2, 128, 768) | Q projection output |
| `key_states` | (2, 128, 768) | K projection output |
| `value_states` | (2, 128, 768) | V projection output |
| `attn_scores` | (2, 12, 128, 128) | Q*K^T / sqrt(d) before softmax |
| `attn_probs` | (2, 12, 128, 128) | softmax(attn_scores) |
| `attn_output` | (2, 128, 768) | attn_probs * V |
| `output_states` | (2, 128, 768) | Final output after O projection |
