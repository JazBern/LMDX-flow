# LMDX-flow

LMDX-flow is a Python toolkit designed for document information extraction using LMDX. 
It simplifies the process of creating prompts which contain document layout information and decoding LLM responses to extract valuable information from documents.

## What is LMDX: LANGUAGE MODEL-BASED DOCUMENT INFORMATION EXTRACTION AND LOCALIZATION?

LMDX is a methodology for leveraging off-the-shelf LLMs for information extraction on
semi-structured documents. 

Paper : https://arxiv.org/pdf/2309.10952.pdf

- Proposes a prompt that enables LLMs to perform the document IE task on leaf and
hierarchical entities with precise localization, including without any training data.
- Proposes a layout encoding scheme that communicate spatial information to the
LLM without any change to its architecture.
- Introduces a decoding algorithm transforming the LLM responses into extracted entities
and their bounding boxes on the document, while discarding all hallucination.


## Key Features

- **Prompt Generation:** Easily create effective prompts based on the LMDX methodology.
- **Response Decoding:** Extract entity values and bounding boxes by decoding and grounding the LLM responses.

## Getting Started

- Install tesseract-OCR

```python
from lmdx_flow import Pipeline
## Load the tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

P = Pipeline(file_path,tokenizer)
prompts = P.generate_prompt(schema)
answers = P.postprocess_all_chunks(llm_responses)
```

## To-do

- Add support for hierarchical entities
- Add option to use OCR-words as segment (currently uses OCR-lines as segment)


Explore the potential of LMDX-flow to enhance document information extraction using LLMs with ease.

