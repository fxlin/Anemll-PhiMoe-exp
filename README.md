fxl  2025-04

comments and addition of PhiMoe 3.5 model. as exp

4-17-25 meta thought:

fxl, 4-17-25. basically construct the compute graph for llama model inference
      the key here is to partition the model (grou players, also do incremental prefill with kvcache...
      also, can map the op to ANE friendly ops (conv2d, ANE-softmax etc
      understdoo 2/5
i guess this is necessary, otherwise 1. model too large to load 2. some ops are not optimized for ANEs 3. dynamic input shape ... can be a problem 
is this really needed for all models to be supported??? 

how about otehr LLMs for ANE projects? they dont do this right? or, models are already traced for them? (like they already have the compute graph


why LUT quant?? why cannot use normal q4, q8, etc?  also supported, can try 

# must use python3.9 (python3.11 missing pytorch2.6 package... which is needed for Half inference...)

# environment 
source ~/workspace-apple-silicon/myenv-python39/bin/activate
cd workspace-apple-silicon/Anemll/

# get pre converted models
git clone https://huggingface.co/anemll/anemll-Meta-Llama-3.2-1B-LUT8_ctx512_0.3.0

# self convert, rerequire macos 15 and coreml compiler installed.
## non instruct model. can convert, but cannot chat.
./anemll/utils/convert_model.sh --model ~/models/Llama-3.2-1B/ --output ~/models/Llama-3.2-1B-coreml-fxl/ 
# llama3 1b isntruct model. convert good
./anemll/utils/convert_model.sh --model ~/models/Llama-3.2-1B-Instruct/ --output ~/models/Llama-3.2-1B-Instruct-coreml-fxl/ 
# llama3 8b isntruct model. convert good
./anemll/utils/convert_model.sh --model ~/models/llama-3-8b-Instruct/ --output ~/models/llama-3-8b-Instruct-coreml-fxl/ 

# llama3 8b isntruct model. long context exp. save log 
./anemll/utils/convert_model.sh \
--model ~/models/llama-3-8b-Instruct/ \
--output ~/models/llama-3-8b-Instruct-ctx4096-coreml-fxl/ \
--context 4096 | tee /tmp/llama3-8b-ctx4096.log

# restart from FFN conversion.... 
./anemll/utils/convert_model.sh --model ~/models/Llama-3.2-1B/ --output ~/models/Llama-3.2-1B-coreml-fxl/ \
--restart 3

./anemll/utils/convert_model.sh --model ~/models/Llama-3.2-1B/ --output ~/models/Llama-3.2-1B-coreml-fxl/ \
--only 1

# ---  converted ok. good

## -- chat.py, ok on preconverted model
python ./tests/chat.py \
--meta /models/anemll-Meta-Llama-3.2-1B-LUT8_ctx512_0.3.0/

#   -- chat.py, failed on tokenizer of my converted model (TBD, should be easy
python ./tests/chat.py \
--meta ~/models/Llama-3.2-1B-Instruct-coreml-fxl/meta.yaml

# to download model
converted: 
https://huggingface.co/anemll

llama officail models:
huggingface-cli login

https://huggingface.co/settings/tokens

# to run the model, downgrade numpy
pip install "numpy<2"

# always cpu execution, why???
cd ~/models/anemll-Meta-Llama-3.2-3B-ctx512_0.1.1
python chat.py --meta ./meta.yaml

[output of llama 3b execution](./3b-output.txt)

# pre-converted llama3 1b. can work, can gen meaingful outout, and uses ANE indeed
cd anemll-swift-cli/
swift run -c release anemllcli --meta ~/models/anemll-Meta-Llama-3.2-1B-LUT8_ctx512_0.3.0/meta.yaml \
--prompt "List US Presidents"

# self converted model llama3 1b. can work, can gen meaingful outout, and uses ANE indeed
cd anemll-swift-cli/
swift run -c release anemllcli \
--meta ~/models/Llama-3.2-1B-Instruct-coreml-fxl/meta.yaml \
--prompt "List US Presidents"

# self converted model llama3 8b. can work, can gen meaingful outout, and uses ANE indeed
cd anemll-swift-cli/
swift run -c release anemllcli \
--meta ~/models/llama-3-8b-Instruct-coreml-fxl/meta.yaml \
--prompt "List US Presidents"

# not working. complain about tokenizer. looks like the vanilla llama is not for chat (missing chat template in tokenizer config??
swift run -c release anemllcli \
 --meta "/Users/felixlin/models/Llama-3.2-1B-coreml-fxl/meta.yaml" \
 --prompt "List US Presidents"

 Error applying chat template: chatTemplate("This tokenizer does not have a chat template, and no template was passed.")
Error applying chat template: chatTemplate("This tokenizer does not have a chat template, and no template was passed.")

Assistant:
Error during generation: inferenceError("Invalid position 0 for context length 1")
Error: inferenceError("Invalid position 0 for context length 1")

swift run -c release anemllcli \
 --meta "/Users/felixlin/models/Llama-3.2-1B-coreml-fxl/meta.yaml" \
 --prompt "List US Presidents" \

--------------------- original README ------------------------

# ANEMLL

ANEMLL (pronounced like "animal") is an open-source project focused on accelerating the porting of Large Language Models (LLMs) to tensor processors, starting with the Apple Neural Engine (ANE).

## Goals
> The goal is to provide a fully open-source pipeline from model conversion to inference for common LLM architectures running on ANE.
> This enables seamless integration and on-device inference for low-power applications on edge devices, ensuring maximum privacy and security.
> This is critical for autonomous applications, where models run directly on the device without requiring an internet connection.
>
> We aim to:
> - Provide flexible and easy to use library/framework to port LLMs to ANE directly from Hugging Face models
> - Provide on-device examples for iOS and macOS swift or C/C++ Applications

See update [Roadmap.md](./docs/Roadmap.md) for more details

## Main Components in 0.3.0 Alpha Release

ANEMLL provides five main components for Apple Neural Engine inference development:

1. [LLM Conversion Tools](./docs/convert.md) - Scripts and code to convert models directly from Hugging Face weights
   - [Single-shot Conversion Script](./docs/convert_model.md)

2. [Swift Reference Implementation](./docs/swift_cli.md) - Optimized inference code for Swift applications
   - Sample CLI application in `anemll-swift-cli`
   - Core inference engine implementation

3. [Python Sample Code](./docs/chat.md) - Reference implementation and testing tools
   - Basic chat interface (`chat.py`)
   - Advanced conversation management (`chat_full.py`)

4. [iOS/macOS Sample Applications](./docs/sample_apps.md) - Ready-to-use example applications (Alpha, now on TestFlight)
   - SwiftUI Chat interface
   - Model Downloads and integration example
   - Conversation management

5. [ANEMLL-BENCH](https://github.com/anemll/anemll-bench) - Apple Neural Engine Benchmarking
   - Performance testing and comparison
   - Model optimization metrics
   - Hardware-specific benchmarks
   - [GitHub Repository](https://github.com/anemll/anemll-bench)

### Pre-converted Models

We provide sample converted models ready for use:
- LLAMA 3.1 (1B and 8B variants) including iOS "friendly builds"
- DeepSeek  distilled models
- DeepHermes distilled models
> [!NOTE]
> Please note that Quantization should be improved. LUT4 quality is fairly low due to lack of Block Quantization on Apple Neural Engine.
> Some GPTQ and Spin Quant should greatly improve LUT4 models.

Visit our [Hugging Face repository](https://huggingface.co/anemll) for the latest converted models.

> [!Important]
> This is Alpha Release 0.3.0 for the library. It is designed to process Model Weights directly from Hugging Face models and convert them to the CoreML format for Apple Neural Engine (ANE for short).
> This is Alpha Release 0.3.0 for the library. It is designed to process Model Weights directly from Hugging Face models and convert them to the CoreML format for Apple Neural Engine (ANE for short).
> - This release only supports LLAMA models including DeepSeek and DeepHermes distilled models on LLaMA 3.1 architecture
> - The future release will add support for more models and architectures
> - Please visit https://huggingface.co/anemll where we upload the latest models and X: [@anemll](https://x.com/anemll) for updates
> - Please star this repo to support the project!

### New in 0.3.0 🚀

Swift UI Sample Code
- Sample iOS/macOS inference Chat-Bot App (Alpha)
- Updates to Model conversion and upload scripts 
- Updates to Swift Package and CLI App


### Sample iOS/macOS Applications
- Downloads reference or custom models from HuggingFace
- Inference / chat implementation use Swift Library
- Sample TestFlight App for a quick test
- See [iOS/macOS Sample Applications Guide](./docs/sample_apps.md) for details

> [!Tip]
> Try our TestFlight app: [Join Beta](https://testflight.apple.com/join/jrQq1D1C)

## Swift CLI Reference Implementation

The Swift CLI provides a reference implementation for running models on Apple Neural Engine. For detailed documentation, see [Swift CLI Guide](./docs/swift_cli.md).

### Quick Start

1. Download a model from [Hugging Face](https://huggingface.co/anemll)
2. Convert the model using our single-shot conversion script:
```bash
./anemll/utils/convert_model.sh --model <path_to_model> --output <output_directory>
```
3. Run the model using our sample code:
```bash
python ./tests/chat.py --meta <output_directory>/meta.yaml
```

For detailed conversion steps and advanced options, see:
- [Model Conversion Guide](./docs/convert.md)
- [Single-shot Conversion Script](./docs/convert_model.md)
- [DeepSeek Model Guide](./docs/ConvertingDeepSeek.md)

## Testing with Python

We provide two chat interfaces:
- `chat.py` - Basic chat interface for quick testing
- `chat_full.py` - Advanced chat with conversation history management

Features of chat_full.py:
- Maintains full conversation history within context window
- Automatically truncates older messages when needed
- Shifts context window dynamically during long responses
- Shows generation speed and token statistics
- Better handles multi-turn conversations

Example running Chats:

```bash
# Basic chat
python ./tests/chat.py --meta ./converted_models/meta.yaml

# Full conversation mode
python ./tests/chat_full.py --meta ./converted_models/meta.yaml
```
See [chat.md](./docs/chat.md) for more details 

> [Note]
>The first time the model loads, macOS will take some time to place it on the device. Subsequent loads will be instantaneous. Use Ctrl-D to exit, Ctrl-C to interrupt inference.



## Installation

### System Requirements
- macOS Sequoia with Apple Neural Engine
- Minimum 16GB RAM
- Python 3.9

### Setup Steps

1. Install ANEMLL:
We recommend creating a new virtual environment for this project.
```bash
python -m venv anemll-env
source anemll-env/bin/activate
pip install -r requirements.txt
# pip install anemll
# due to Alpha Release, we do not recommend installing ANEMLL as a package yet
```
CoreML compiler is required to compile the model. It is part of the Xcode command line tools.
- Ensure that Xcode Command Line Tools are installed, as they include `coremlcompiler`.
- You can install them by running `xcode-select --install`.
- Verify that the `xcrun` command is available and correctly configured in your PATH.
- Use `xcrun --find coremlcompiler` to verify the installation.
- If above fails, please try following steps:
- Download Xcode from the App Store.
- Run `sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/` to set the path.
- Use `xcrun --find coremlcompiler` to verify the installation.
- Run `sudo xcodebuild -license` and agree to the license.


## Model Support

Currently optimized for:
- Meta's LLaMA 3.2 1B and 8B (1024 context) model including DeepSeek R1 8B distilled model, DeepHermes 3B and 8B models
- More models are coming soon

## Acknowledgements

### Core Technologies
- Thanks to [@apple](https://apple.com) for developing the Apple Neural Engine 
- Thanks to Apple CoreML Tools team for providing the tools https://github.com/apple/coremltools
- Thanks to [@huggingface](https://huggingface.co) for providing the transformers library and models

### Inspirations, feedback and other resources
- Stephen Panaro https://x.com/flat for feedback and coreml-llm-cli https://github.com/smpanaro/coreml-llm-cli 
- Seba https://x.com/CulStory for inspiration with fast ANE models. https://huggingface.co/seba
- Maynard Handley https://x.com/handleym99 For indepth ANE resources https://github.com/name99-org/AArch64-Explore/blob/main/vol7%20ANE.nb.pdf and feedback

## Contributing

> [!Note]
> We welcome contributions! Please read our contributing guidelines before submitting PRs.

Feel free to submit issues and pull requests to improve **ANEMLL**!

> [!Note]
> If you're using ANEMLL in your project, please submit a PR to add it to this list.
> We love to showcase how the community is using ANEMLL!
### Third-Party Applications Using ANEMLL

### Open Source Projects
- [anemll-server](https://github.com/alexgusevski/anemll-server) - Server implementation of ANEMLL inference

> [!Note]
> If you're using ANEMLL in your project, please submit a PR to add it to this list.
> We love to showcase how the community is using ANEMLL!

### Integration Examples
For examples of how to integrate ANEMLL into your projects, see:
- [iOS Integration Guide](./docs/sample_apps.md)
- [Swift CLI Reference](./docs/swift_cli.md)
- [Python Sample Code](./docs/chat.md)

## Links & Resources

- 🌐 Website: [anemll.com](https://anemll.com)
- 🤗 Models: [huggingface.co/anemll](https://huggingface.co/anemll)
- 📱 X: [@anemll](https://x.com/anemll)
- 💻 GitHub: [github.com/anemll](https://github.com/anemll)

## Contact

For any questions or support, reach out to us at [realanemll@gmail.com](mailto:realanemll@gmail.com)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Anemll/Anemll&type=Date)](https://star-history.com/#Anemll/Anemll&Date)

## License

ANEMLL is licensed under the MIT License.
https://opensource.org/license/mit
