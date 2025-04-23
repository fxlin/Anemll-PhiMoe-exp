# fxl
is "DeepSeek-R1-8B" deepseek-r1-llama-8b??? guess so? 

--------------


# Converting DeepSeek Model

> [!Important]
> This guide details the process of converting the DeepSeek-R1-8B model for use with Apple Neural Engine.
> - Model is split into 10  chunks (8 FFN+PREFILL) due to its large size
> - Each chunk must be under 1GB for iOS compatibility
> - Conversion process may take several an hor or more and require 32GB of strage. 64GB RAM is recommended for conversion.
> - Requires macOS 15 or later
> - This model is quantized to 6-bit LUT, its a bit slower vs 4-bit LUT, but provides better quality.

## Model Conversion Steps

### 1. Convert Embedding part (1)
> Creates `DeepSeek_embeddings.mlpackage`
```bash
python -m anemll.ane_converter.llama_converter \
    --part 1 \
    --model "../DeepSeekR1-8B/" \
    --context-length 1024 \
    --prefix "DeepSeek"
```

### 2. Convert LM Head part (3)
> Creates `DeepSeek_lm_head_lut6.mlpackage` with 6-bit LUT quantization
```bash
python -m anemll.ane_converter.llama_converter \
    --part 3 \
    --lut 6 \
    --model "../DeepSeekR1-8B/" \
    --context-length 1024 \
    --batch-size 64 \
    --prefix "DeepSeek"
```

### 3. Convert FFN part (2)
> Creates 8 chunk files: `DeepSeek_FFN_PF_lut6_chunk_[01-08]of08.mlpackage`
```bash
python -m anemll.ane_converter.llama_converter \
    --part 2 \
    --lut 6 \
    --chunk 8 \
    --context-length 1024 \
    --batch-size 64 \
    --model "../DeepSeekR1-8B/" \
    --prefix "DeepSeek"
```

### 4. Convert Prefill part (2_prefill)
> Creates prefill model chunks for KV cache optimization
```bash
python -m anemll.ane_converter.llama_converter \
    --part 2_prefill \
    --lut 6 \
    --chunk 8 \
    --context-length 1024 \
    --batch-size 64 \
    --model "../DeepSeekR1-8B/" \
    --prefix "DeepSeek"
```

### 5. Combine Models
> Merges FFN and prefill chunks to reduce weight size by 50%
```bash
python ./anemll/utils/combine_models.py \
    --lut 6 \
    --chunk 8 \
    --prefix "DeepSeek"
```

### 6. Compile Models
> Converts to MLModelC format for device inference
```bash
python ./anemll/utils/compile_models.py 1 --prefix "DeepSeek"
python ./anemll/utils/compile_models.py 3 --lut 6 --prefix "DeepSeek"
python ./anemll/utils/compile_models.py 2 --lut 6 --chunk 8 --prefix "DeepSeek"
```

### 7. Test with chat.py
> There are two chat interfaces available:

#### Basic Chat (chat.py)
Basic chat interface for testing the model.

#### Full Conversation Chat (chat_full.py)
Advanced chat interface with:
- Full conversation history maintenance
- Automatic history truncation
- Dynamic context window shifting
- Generation statistics

Both can be run in several ways:

#### Option 1: Specify model directory
```bash
python ./tests/chat.py \
    --d /path/to/models/anemll-DeepSeek-8B-ctx1024-v2/ \
    --embed DeepSeek_embeddings \
    --lmhead DeepSeek_lm_head_lut6 \
    --ffn DeepSeek_FFN_PF_lut6_chunk_01of08
```

> [!Note]
> - The `--d` option specifies the directory containing all model files
> - Model paths become relative to this directory
> - Context length (1024) will be detected from directory name (ctx1024)
> - Tokenizer will use the model directory by default

#### Option 2: Run from external folder 
```bash
cd /path/to/models/anemll-DeepSeek-8B-ctx1024-v2/
python ./tests/chat.py \
    --embed DeepSeek_embeddings \
    --lmhead DeepSeek_lm_head_lut6 \
    --ffn DeepSeek_FFN_PF_lut6_chunk_01of08 \
    --tokenizer /path/to/original/DeepSeekR1-8B/
```
> Note: If you used [convert_model.sh](convert_model.md) to convert the model, you can also run full chat using the generated meta.yaml:
```

> [!Note]
> - Change to the directory containing model files
> - Use relative paths for model files
> - Specify external tokenizer path explicitly
> - Context length will be detected from current directory name


#### Context Length
- If directory name contains `ctxNNNN` (e.g. ctx1024), that value is used
- Can be overridden with `--context-length` argument
- Must match the context length used during model conversion
- Default is 512 if not specified

#### Tokenizer Path
1. Uses `--tokenizer` path if provided
2. Otherwise uses `--d` directory if provided
3. Falls back to current directory
4. Must contain tokenizer files (tokenizer.model, etc.)

## Output Files

After conversion and compilation, you should have:

### MLPackage Files (Intermediate)
- `DeepSeek_embeddings.mlpackage`
- `DeepSeek_lm_head_lut6.mlpackage`
- Eight FFN chunk files:
  - `DeepSeek_FFN_PF_lut6_chunk_01of08.mlpackage` through
  - `DeepSeek_FFN_PF_lut6_chunk_08of08.mlpackage`

### MLModelC Files (Final)
> These are the compiled files used for inference
- `DeepSeek_embeddings.mlmodelc/`
- `DeepSeek_lm_head_lut6.mlmodelc/`
- Eight FFN chunk directories:
  - `DeepSeek_FFN_PF_lut6_chunk_01of08.mlmodelc/` through
  - `DeepSeek_FFN_PF_lut6_chunk_08of08.mlmodelc/`

> [!Note]
> The .mlmodelc files are actually directories containing the compiled model assets.
> These are the files that should be included in your application for inference.

## Troubleshooting

### Memory Issues
- Close other applications
- Ensure sufficient free RAM
- Monitor Activity Monitor during conversion

### Disk Space
- Ensure at least 50GB free space
- Clean up temporary files if needed
- Monitor disk usage during conversion

### Compilation Errors
- Verify Xcode installation
- Check Command Line Tools are installed
- Ensure all previous steps completed successfully
