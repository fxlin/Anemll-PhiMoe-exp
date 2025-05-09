
# based on: profile_coreml.py, simplified
# cf: https://apple.github.io/coremltools/source/coremltools.models.html#coremltools.models.model.MLModel.get_spec

# will prepare all inputs, based on model input specs
#     as well as state tensors
#   TBD: to test each function in a"multifunction" model

import os
import sys
import argparse
import logging
import time
import platform
import subprocess
import webbrowser
import json
from typing import Optional, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import coremltools as ct
import numpy as np

def get_model_size(model_path):
    """Get the size of the model in bytes and megabytes."""
    total_size = 0
    weights_size = 0
    
    if os.path.isdir(model_path):
        # Directory (mlmodelc or mlpackage)
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                file_size = os.path.getsize(fp)
                total_size += file_size
                
                # Rough approximation of weight files - this is just an estimation
                if "weight" in f.lower() or "model" in f.lower() or f.endswith('.bin'):
                    weights_size += file_size
    else:
        # Single file
        total_size = os.path.getsize(model_path)
        weights_size = total_size  # For single files, assume all are weights
    
    return {
        "size_bytes": total_size,
        "size_mb": total_size / (1024 * 1024),
        "weights_bytes": weights_size,
        "weights_mb": weights_size / (1024 * 1024),
        "weights_percentage": (weights_size / total_size * 100) if total_size > 0 else 0
    }

def get_system_info() -> Dict[str, Any]:
    """Get basic system information"""
    import platform
    import subprocess
    
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    
    # Get more detailed Mac info if on macOS
    if platform.system() == "Darwin":
        # Get Mac model
        try:
            mac_model = subprocess.check_output(["sysctl", "-n", "hw.model"]).decode("utf-8").strip()
            system_info["mac_model"] = mac_model
        except:
            system_info["mac_model"] = "Unknown Mac"
        
        # Check if Apple Silicon
        try:
            processor_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode("utf-8").strip()
            system_info["cpu_model"] = processor_info
            system_info["is_apple_silicon"] = "Apple" in processor_info
        except:
            system_info["is_apple_silicon"] = False
            
        # Get memory
        try:
            memory_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode("utf-8").strip())
            system_info["memory_gb"] = round(memory_bytes / (1024**3), 1)
        except:
            system_info["memory_gb"] = "Unknown"
            
        # Get a cleaner macOS version - e.g., "macOS 13.4 Ventura"
        try:
            # Get macOS version using sw_vers
            macos_version = subprocess.check_output(["sw_vers", "-productVersion"]).decode("utf-8").strip()
            macos_name = "macOS"
            
            # Determine macOS name based on version
            version_major = int(macos_version.split('.')[0])
            if version_major == 10:
                minor = int(macos_version.split('.')[1])
                if minor == 15:
                    macos_name = "macOS Catalina"
                elif minor == 14:
                    macos_name = "macOS Mojave"
                elif minor == 13:
                    macos_name = "macOS High Sierra"
                else:
                    macos_name = "macOS"
            elif version_major == 11:
                macos_name = "macOS Big Sur"
            elif version_major == 12:
                macos_name = "macOS Monterey"
            elif version_major == 13:
                macos_name = "macOS Ventura"
            elif version_major == 14:
                macos_name = "macOS Sonoma"
            elif version_major == 15:
                macos_name = "macOS Sequoia"
            
            system_info["os_display"] = f"{macos_name} {macos_version}"
        except:
            system_info["os_display"] = "macOS Unknown Version"
    
    return system_info

def main():
    parser = argparse.ArgumentParser(description="Profile a local CoreML model without triggering downloads")
    parser.add_argument("--model", required=True, help="Path to the CoreML model file (.mlpackage or .mlmodelc)")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations for profiling (default: 20)")
    parser.add_argument("--cpu-only", action="store_true", help="Force the model to run on CPU only (default: False)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model not found: {args.model}")
        return 1

    if args.cpu_only:
        compute_unit = ct.ComputeUnit.CPU_ONLY
    else:
        compute_unit = ct.ComputeUnit.CPU_AND_NE

    logger.info(f"Profiling model: {args.model}, compute unit: {compute_unit}")

    system_info = get_system_info()
    model_size_info = get_model_size(args.model)

    try:
        model = ct.models.MLModel(args.model, compute_units=compute_unit)
        print(f"Successfully loaded model: {args.model}")

        # Get inputs from spec
        if not hasattr(model, 'get_spec'):
            raise RuntimeError("Model does not have a get_spec() method")

        spec = model.get_spec().description.input
        if not spec:
            raise RuntimeError("Model spec has no input descriptions")

        inputs = {}
        for input_desc in spec:
            name = input_desc.name

            print(f"Processing input: {name}")

            # state feature...
            if input_desc.type.WhichOneof("Type") == "stateFeatureType":
                # Create dummy state tensors (matching required shape)
                shape = tuple(input_desc.type.stateFeatureType.shape)
                dtype = np.float32
                inputs[name] = np.zeros(shape, dtype=dtype)
                print(f"  {name}: [StateTensor] shape={shape}, dtype={dtype}")
                continue

            if input_desc.type.WhichOneof("Type") != "multiArrayType":
                raise ValueError(f"Unsupported input type for input '{name}'")
            
            # ... input 
            shape = tuple(input_desc.type.multiArrayType.shape)
            dtype = np.float32  # default
            if input_desc.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE:
                dtype = np.float64
            elif input_desc.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32:
                dtype = np.float32
            elif input_desc.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16:
                dtype = np.float16
            elif input_desc.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT32:
                dtype = np.int32
            elif input_desc.type.multiArrayType.dataType == ct.proto.FeatureTypes_pb2.ArrayFeatureType.INT64:
                dtype = np.int64
            else:
                raise ValueError(f"Unsupported input dtype for input '{name}'")

            inputs[name] = np.random.rand(*shape).astype(dtype) if np.issubdtype(dtype, np.floating) else np.random.randint(0, 10, size=shape, dtype=dtype)

        state = model.make_state()

        print(f"Prepared random input tensors:")
        for k, v in inputs.items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

        print("Warming up model...")
        for _ in range(5):
            model.predict(inputs, state=state)

        print(f"Running {args.iterations} iterations for profiling...")
        inference_times = []
        for _ in range(args.iterations):
            start_time = time.time()
            model.predict(inputs, state=state)
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        inference_time_ms = np.median(inference_times)
        total_data_size = model_size_info['size_bytes'] + sum(np.prod(v.shape) * v.itemsize for v in inputs.values())
        throughput_gb_s = (total_data_size / (inference_time_ms / 1000)) / 1e9

        print(f"\nMedian inference time: {inference_time_ms:.2f} ms")
        print(f"Throughput: {throughput_gb_s:.2f} GB/s (based on weights + I/O)")
        print(f"Model size: {model_size_info['size_mb']:.2f} MB")
        print(f"Weights size: {model_size_info['weights_mb']:.2f} MB ({model_size_info['weights_percentage']:.1f}% of total)")

    except Exception as e:
        logger.error(f"Error profiling model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\nProfile complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 