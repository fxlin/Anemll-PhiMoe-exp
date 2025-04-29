#!/bin/bash

# fxl: based on convert_model.sh

# Get script directory for relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add project root to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Default values
CONTEXT_LENGTH=512
BATCH_SIZE=64
LUT_PART1=""  # No LUT for embeddings
LUT_PART2=4   # FFN and prefill
LUT_PART3=6   # LM head
RESTART_STEP=1
ONLY_STEP=""  # Run only this step if set
PREFIX="phimoe"  # Default prefix for model names
MODEL_PATH=""
OUTPUT_DIR=""
NUM_CHUNKS=2   # Default number of chunks

# Initialize SKIP_CHECK before parsing arguments
SKIP_CHECK=false

# Initialize SKIP_CHECK before parsing arguments
SKIP_CHECK=false

# Function to print usage
print_usage() {
    echo "Usage: $0 --model <path_to_model> --output <output_directory> [options]"
    echo "Options:"
    echo "  --model         Path to the model directory (required)"
    echo "  --output        Output directory for converted models (required)"
    echo "  --context       Context length (default: 512)"
    echo "  --batch         Batch size (default: 64)"
    echo "  --lut1          LUT bits for embeddings (default: none)"
    echo "  --lut2          LUT bits for FFN/prefill (default: 4)"
    echo "  --lut3          LUT bits for LM head (default: 6)"
    echo "  --restart       Restart from specific step (1-8, default: 1)"
    echo "  --only          Run only specified step and exit (1-8)"
    echo "  --prefix        Prefix for model names (default: phimoe)"
    echo "  --chunk         Number of chunks to split FFN/prefill (default: 2)"
    echo "  --skip-check    Skip the dependency check step"
    echo "  --skip-check    Skip the dependency check step"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --context)
            CONTEXT_LENGTH="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lut1)
            LUT_PART1="$2"
            shift 2
            ;;
        --lut2)
            LUT_PART2="$2"
            shift 2
            ;;
        --lut3)
            LUT_PART3="$2"
            shift 2
            ;;
        --restart)
            RESTART_STEP="$2"
            shift 2
            ;;
        --only)
            ONLY_STEP="$2"
            shift 2
            ;;
        --chunk)
            NUM_CHUNKS="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        --skip-check)
            SKIP_CHECK=true
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            print_usage
            ;;
    esac
done

# Validate required parameters
if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Model path and output directory are required"
    print_usage
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_PATH"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory"
        exit 1
    fi
fi

# Convert paths to absolute paths
MODEL_PATH="$(cd "$(dirname "$MODEL_PATH")" && pwd)/$(basename "$MODEL_PATH")"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)" || {
    # If output directory doesn't exist, get absolute path another way
    OUTPUT_DIR="$(cd "$(dirname "$OUTPUT_DIR")" && pwd)/$(basename "$OUTPUT_DIR")"
}

# Step 0: Check dependencies
if [ "$SKIP_CHECK" = false ]; then
    "$SCRIPT_DIR/check_dependencies.sh" --model "$MODEL_PATH" --output "$OUTPUT_DIR" "$@"
    if [ $? -ne 0 ]; then
        echo "Dependency check failed. Aborting."
        exit 1
    fi
fi

# Step 0: Check dependencies
if [ "$SKIP_CHECK" = false ]; then
    "$SCRIPT_DIR/check_dependencies.sh" --model "$MODEL_PATH" --output "$OUTPUT_DIR" "$@"
    if [ $? -ne 0 ]; then
        echo "Dependency check failed. Aborting."
        exit 1
    fi
fi

# Function to run step if restart_step is less than or equal to step number
run_step() {
    local step=$1
    local description=$2
    local cmd=$3
    
    if [ $RESTART_STEP -le $step ]; then
        # Skip if ONLY_STEP is set and doesn't match current step
        if [ ! -z "$ONLY_STEP" ] && [ "$ONLY_STEP" != "$step" ]; then
            return
        fi
        
        echo "Step $step: $description"
        bash -c "$cmd"
        if [ $? -ne 0 ]; then
            echo "Error in step $step: $description"
            exit 1
        fi
        # Exit after running the only step if specified
        if [ ! -z "$ONLY_STEP" ] && [ "$ONLY_STEP" = "$step" ]; then
            echo "Completed step $step (--only mode)"
            exit 0
        fi
    else
        # Only show skip message if not in ONLY_STEP mode
        if [ -z "$ONLY_STEP" ]; then
            echo "Skipping step $step: $description"
        fi
    fi
}

# Step 1: Convert Embeddings (Part 1)
LUT1_PARAM=""
if [ ! -z "$LUT_PART1" ]; then
    LUT1_PARAM="--lut $LUT_PART1"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "1" ]; then
    run_step 1 "Converting Embeddings" "python -m anemll.ane_converter.phimoe_converter \
        --part 1 \
        $LUT1_PARAM \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 1: Converting Embeddings"
fi

# Step 2: Convert LM Head (Part 3)
LUT3_PARAM=""
if [ ! -z "$LUT_PART3" ]; then
    LUT3_PARAM="--lut $LUT_PART3"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then
    run_step 2 "Converting LM Head" "python -m anemll.ane_converter.phimoe_converter \
        --part 3 \
        $LUT3_PARAM \
        --context-length $CONTEXT_LENGTH \
        --context-length $CONTEXT_LENGTH \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 2: Converting LM Head"
fi

# Step 3: Convert FFN (Part 2)
LUT2_PARAM=""
if [ ! -z "$LUT_PART2" ]; then
    LUT2_PARAM="--lut $LUT_PART2"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then
    run_step 3 "Converting FFN" "python -m anemll.ane_converter.phimoe_converter \
        --part 2 \
        $LUT2_PARAM \
        --chunk $NUM_CHUNKS \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 3: Converting FFN"
fi

if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then
    run_step 4 "Converting Prefill" "python -m anemll.ane_converter.phimoe_converter \
        --part 2_prefill \
        $LUT2_PARAM \
        --chunk $NUM_CHUNKS \
        --context-length $CONTEXT_LENGTH \
        --batch-size $BATCH_SIZE \
        --prefix \"$PREFIX\" \
        --model \"$MODEL_PATH\" \
        --output \"$OUTPUT_DIR\""
else
    echo "Skipping step 4: Converting Prefill"
fi

# Step 5: Combine Models
if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then
    if [ ! -z "$LUT_PART2" ]; then
        run_step 5 "Combining Models" "python \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
            --chunk $NUM_CHUNKS \
            $LUT2_PARAM \
            --prefix \"$PREFIX\" \
            --input \"$OUTPUT_DIR\" \
            --output \"$OUTPUT_DIR\""
    else
        run_step 5 "Combining Models" "python \"$PROJECT_ROOT/anemll/utils/combine_models.py\" \
            --chunk $NUM_CHUNKS \
            --prefix \"$PREFIX\" \
            --input \"$OUTPUT_DIR\" \
            --output \"$OUTPUT_DIR\""
    fi
else
    echo "Skipping step 5: Combining Models"
fi

# Step 6: Compile Models - Always run compilation for all parts that have LUT specified  
# fxl: NB to compile model for embedding, PF_FFN (merged)(ie transfomrers), and LM head
run_step 6 "Compiling Models Part 1" "python \"$PROJECT_ROOT/anemll/utils/compile_models.py\" 1 ${LUT_PART1:+--lut $LUT_PART1} --prefix \"$PREFIX\" --input \"$OUTPUT_DIR\" --output \"$OUTPUT_DIR\""
run_step 6 "Compiling Models Part 3" "python \"$PROJECT_ROOT/anemll/utils/compile_models.py\" 3 ${LUT_PART3:+--lut $LUT_PART3} --prefix \"$PREFIX\" --input \"$OUTPUT_DIR\" --output \"$OUTPUT_DIR\""
if [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "2" ]; then    
    run_step 6 "Compiling Models Part 2" "python \"$PROJECT_ROOT/anemll/utils/compile_models.py\" 2 ${LUT_PART2:+--lut $LUT_PART2} --chunk $NUM_CHUNKS --prefix \"$PREFIX\" --input \"$OUTPUT_DIR\" --output \"$OUTPUT_DIR\""
fi

# Step 7: Copy tokenizer files and create meta.yaml
if [ "$MODEL_PATH" != "$OUTPUT_DIR" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH")
    run_step 7 "Copying tokenizer files and creating meta.yaml" "
        # Copy tokenizer files if they exist
        (cp \"$MODEL_PATH/tokenizer.json\" \"$OUTPUT_DIR/\" || true) && \
        (cp \"$MODEL_PATH/tokenizer_config.json\" \"$OUTPUT_DIR/\" || true) && \
        
        # Create config.json if it doesn't exist
        if [ ! -f \"$OUTPUT_DIR/config.json\" ]; then
            echo \"Creating config.json for iOS tokenizer...\" && \
            python -m anemll.ane_converter.create_config_json --output \"$OUTPUT_DIR/config.json\"
        fi && \
        
        # Create meta.yaml
        python3 - \"$MODEL_NAME\" \"$CONTEXT_LENGTH\" \"$BATCH_SIZE\" \
            \"${LUT_PART1:-none}\" \"${LUT_PART2:-none}\" \"${LUT_PART3:-none}\" \
            $NUM_CHUNKS \"$PREFIX\" \"$OUTPUT_DIR/meta.yaml\" <<'EOF_PY'
import sys
MODEL_NAME = sys.argv[1]
CONTEXT = sys.argv[2]
BATCH = sys.argv[3]
LUT_EMB = sys.argv[4]
LUT_FFN = sys.argv[5]
LUT_LMH = sys.argv[6]
NUM_CHUNKS = sys.argv[7]
PREFIX = sys.argv[8]
OUTFILE = sys.argv[9]

# Construct model names with LUT suffixes if specified
embeddings_name = f'{PREFIX}_embeddings' + (f'_lut{LUT_EMB}' if LUT_EMB != 'none' else '')
lmhead_name = f'{PREFIX}_lm_head' + (f'_lut{LUT_LMH}' if LUT_LMH != 'none' else '')
ffn_base = f'{PREFIX}_FFN_PF' + (f'_lut{LUT_FFN}' if LUT_FFN != 'none' else '')

# Add .mlmodelc extension to model paths
embeddings_path = f'{embeddings_name}.mlmodelc'
lmhead_path = f'{lmhead_name}.mlmodelc'
ffn_path = f'{ffn_base}.mlmodelc'

meta = f'''model_info:
  name: anemll-{MODEL_NAME}-ctx{CONTEXT}
  version: 0.3.0
  description: |
    Demonstarates running {MODEL_NAME} on Apple Neural Engine
    Context length: {CONTEXT}
    Batch size: {BATCH}
    Chunks: {NUM_CHUNKS}
  license: MIT
  author: Anemll
  framework: Core ML
  language: Python
  parameters:
    context_length: {CONTEXT}
    batch_size: {BATCH}
    lut_embeddings: {LUT_EMB}
    lut_ffn: {LUT_FFN}
    lut_lmhead: {LUT_LMH}
    num_chunks: {NUM_CHUNKS}
    model_prefix: {PREFIX}
    embeddings: {embeddings_path}
    lm_head: {lmhead_path}
    ffn: {ffn_path}
'''
with open(OUTFILE, 'w') as f:
    f.write(meta)
EOF_PY
    "
fi


# Step 8: Test with chat.py
run_step 8 "Testing with chat.py" "python \"$PROJECT_ROOT/tests/chat.py\" \
    --meta \"$OUTPUT_DIR/meta.yaml\" \
    --prompt \"Who are you ?\""

# Print chat.py command for reference
echo -e "\nTo chat with the model, use:"
echo -e "\nOption 1 - Using meta.yaml (recommended):"
echo "python $PROJECT_ROOT/tests/chat.py \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""
echo -e "\nOr for full conversation mode:"
echo "python $PROJECT_ROOT/tests/chat_full.py \\"
echo "    --meta \"$OUTPUT_DIR/meta.yaml\""

echo -e "\nOption 2 - Manual configuration:"
EMBEDDINGS_NAME="${PREFIX}_embeddings${LUT_PART1:+_lut$LUT_PART1}"
LMHEAD_NAME="${PREFIX}_lm_head${LUT_PART3:+_lut$LUT_PART3}"
FFN_BASE="${PREFIX}_FFN_PF${LUT_PART2:+_lut$LUT_PART2}"

echo "python $PROJECT_ROOT/tests/chat.py \\"
echo "    --embed $EMBEDDINGS_NAME \\"
echo "    --lmhead $LMHEAD_NAME \\"
echo "    --ffn ${FFN_BASE}_chunk_01of$(printf "%02d" $NUM_CHUNKS) \\"
echo "    --tokenizer \"$OUTPUT_DIR\" \\"
echo "    --context-length $CONTEXT_LENGTH \\"
echo "    --d \"$OUTPUT_DIR\""

echo "Conversion completed successfully!" 