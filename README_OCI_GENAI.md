# OCI Gen AI Integration for Ragas

This contribution adds direct support for Oracle Cloud Infrastructure (OCI) Generative AI models in Ragas, enabling evaluation without requiring LangChain or LlamaIndex.

## 🚀 Features

- **Direct OCI Integration**: Uses OCI Python SDK directly, no LangChain/LlamaIndex dependency
- **Multiple Model Support**: Works with Cohere, Meta, Mistral, and other OCI Gen AI models
- **Async Support**: Full async/await support for high-performance evaluation
- **Error Handling**: Comprehensive error handling and retry mechanisms
- **Analytics Tracking**: Built-in usage tracking and analytics
- **Flexible Configuration**: Support for custom OCI configs and endpoints

## 📦 Installation

```bash
# Install Ragas with OCI support
pip install ragas[oci]

# Or install OCI SDK separately
pip install oci>=2.160.1
```

## 🔧 Quick Start

```python
from ragas.llms import oci_genai_factory
from ragas import evaluate
from datasets import Dataset

# Initialize OCI Gen AI LLM
llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Your dataset
dataset = Dataset.from_dict({
    "question": ["What is the capital of France?"],
    "answer": ["Paris"],
    "contexts": [["France is a country in Europe. Its capital is Paris."]],
    "ground_truth": ["Paris"]
})

# Evaluate with OCI Gen AI
result = evaluate(dataset, llm=llm)
```

## 🏗️ Architecture

### Files Added/Modified

- `src/ragas/llms/oci_genai_wrapper.py` - Main OCI Gen AI wrapper implementation
- `src/ragas/llms/__init__.py` - Updated to export OCI Gen AI classes
- `pyproject.toml` - Added OCI as optional dependency
- `tests/unit/test_oci_genai_wrapper.py` - Comprehensive test suite
- `docs/howtos/integrations/oci_genai.md` - Complete documentation
- `examples/oci_genai_example.py` - Working example script

### Key Components

1. **OCIGenAIWrapper**: Main LLM wrapper class
2. **oci_genai_factory**: Factory function for easy initialization
3. **Comprehensive Testing**: Unit tests with mocking
4. **Documentation**: Complete usage guide and examples

## 🧪 Testing

```bash
# Run OCI Gen AI tests
pytest tests/unit/test_oci_genai_wrapper.py -v

# Run example script
python examples/oci_genai_example.py
```

## 📚 Documentation

- [Complete Integration Guide](docs/howtos/integrations/oci_genai.md)
- [OCI Python SDK Documentation](https://docs.oracle.com/en-us/iaas/tools/python/2.160.1/api/generative_ai.html)
- [Ragas Documentation](https://docs.ragas.io)

## 🔑 Configuration

### OCI Authentication

```python
# Option 1: Use OCI CLI config
# oci setup config

# Option 2: Environment variables
export OCI_CONFIG_FILE=~/.oci/config
export OCI_PROFILE=DEFAULT

# Option 3: Manual configuration
config = {
    "user": "ocid1.user.oc1..example",
    "key_file": "~/.oci/private_key.pem",
    "fingerprint": "your_fingerprint",
    "tenancy": "ocid1.tenancy.oc1..example",
    "region": "us-ashburn-1"
}
```

### Supported Models

- **Cohere**: `cohere.command`, `cohere.command-light`
- **Meta**: `meta.llama-3-8b`, `meta.llama-3-70b`
- **Mistral**: `mistral.mistral-7b-instruct`
- **And more**: Check OCI documentation for latest models

## 🚀 Usage Examples

### Basic Usage

```python
from ragas.llms import oci_genai_factory

llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)
```

### Advanced Configuration

```python
from ragas.llms import oci_genai_factory
from ragas.run_config import RunConfig

# Custom configuration
config = {
    "user": "ocid1.user.oc1..example",
    "key_file": "~/.oci/private_key.pem",
    "fingerprint": "your_fingerprint",
    "tenancy": "ocid1.tenancy.oc1..example",
    "region": "us-ashburn-1"
}

# Custom run configuration
run_config = RunConfig(timeout=60, max_retries=3)

llm = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example",
    config=config,
    endpoint_id="ocid1.endpoint.oc1..example",  # Optional
    run_config=run_config
)
```

### With Different Models

```python
# Cohere Command
llm_cohere = oci_genai_factory(
    model_id="cohere.command",
    compartment_id="ocid1.compartment.oc1..example"
)

# Meta Llama
llm_llama = oci_genai_factory(
    model_id="meta.llama-3-8b",
    compartment_id="ocid1.compartment.oc1..example"
)
```

## 🐛 Troubleshooting

### Common Issues

1. **Authentication Errors**: Verify OCI configuration and credentials
2. **Model Not Found**: Check if model exists in your compartment
3. **Permission Errors**: Ensure proper IAM policies for Generative AI

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your OCI Gen AI code here
```

## 🤝 Contributing

This integration follows Ragas patterns and conventions:

- Extends `BaseRagasLLM` abstract class
- Implements required methods: `generate_text`, `agenerate_text`, `is_finished`
- Includes comprehensive error handling
- Provides analytics tracking
- Follows existing code style and patterns

## 📄 License

This contribution is part of the Ragas project and follows the same license terms.

## 🙏 Acknowledgments

- Oracle Cloud Infrastructure team for the Python SDK
- Ragas team for the excellent evaluation framework
- Open source community for inspiration and feedback

---

**Ready to evaluate with OCI Gen AI?** Check out the [complete documentation](docs/howtos/integrations/oci_genai.md) and [example script](examples/oci_genai_example.py)!
