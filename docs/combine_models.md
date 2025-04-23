# fxl 4/17/25
combine then split again??
 a combined model -- coreml's "multifunction model"
combine b/c mltools can merge redudnantw eights in "infer" and "prefill" (they indeed 
    share same weights
    (only "FFN_PF" seem needed, after merging
input: mlpackages for multiple model "parts"

---------------------

# Combine Models Documentation

The `combine_models.py` utility is a crucial component in the ANEMLL workflow that optimizes model storage by merging Feed Forward Network (FFN) and Prefill model chunks into Multi-Function Chunks.

## Purpose

The primary purpose of this tool is to reduce the overall model weight size by approximately 50% by combining FFN and KV pre-fill models that share the same weights into unified Multi-Function Chunks.

## Location
```
./anemll/utils/combine_models.py
```

## Usage

Basic command structure:
```bash
python ./anemll/utils/combine_models.py [OPTIONS]
```

### Command Line Arguments

- `--lut`: LUT quantization bits (typically 6)
- `--chunk`: Number of chunks the model is split into
- `--input-dir`: (Optional) Input directory containing the MLPackage files
- `--output-dir`: (Optional) Output directory for combined MLPackage files

## Example Usage

Basic usage with 6-bit quantization and 2 chunks:
```bash
python ./anemll/utils/combine_models.py --lut 6 --chunk 2
```

## Input Files

The utility expects the following MLPackage files to be present:
- FFN chunks: `llama_FFN_lut{N}_chunk_{X}of{Y}.mlpackage`
- Prefill chunks: `llama_prefill_lut{N}_chunk_{X}of{Y}.mlpackage`

Where:
- `N` is the LUT bits
- `X` is the current chunk number
- `Y` is the total number of chunks

## Output Files

The tool generates combined MLPackage files with the following naming convention:
```
llama_FFN_PF_lut{N}_chunk_{X}of{Y}.mlpackage
```

For example, with 6-bit LUT and 2 chunks:
- `llama_FFN_PF_lut6_chunk_01of02.mlpackage`
- `llama_FFN_PF_lut6_chunk_02of02.mlpackage`

## Process Flow

1. Loads the corresponding FFN and Prefill chunks
2. Combines the models while maintaining their respective functionalities
3. Optimizes the shared weights
4. Saves the combined models as new MLPackage files

## Benefits

1. **Storage Optimization**: Reduces the total model size by approximately 50%
2. **Memory Efficiency**: Eliminates redundant weight storage
3. **Performance**: No impact on inference performance
4. **iOS Compatibility**: Helps maintain file size under iOS 1GB limit

## Integration in Workflow

This utility is typically used after converting the individual model parts and before compiling the models for deployment:

1. Convert model parts using ANE_converter
2. Combine FFN and Prefill chunks using combine_models
3. Compile final models using compile_models

## Notes

1. Ensure all input MLPackage files are present before running the combination
2. The number of chunks specified must match the original conversion
3. LUT quantization bits must match the original conversion
4. Backup original MLPackage files before combining

## Related Tools

- [ANE_converter.py](ANE_converter.md): Creates initial MLPackage files
- [compile_models.py](compile_models.md): Compiles combined models for deployment

For the complete conversion workflow, refer to the [convert.md](convert.md) documentation. 