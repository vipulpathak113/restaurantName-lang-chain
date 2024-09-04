
from langchain_huggingface import HuggingFacePipeline

# Define the prompt and model
prompt = "I want to open an Indian restaurant. Suggest me a fancy name"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

from langchain_huggingface.llms import HuggingFacePipeline

hf = HuggingFacePipeline.from_model_id(
    model_id=model_name,
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 10},
)

from langchain_core.prompts import PromptTemplate

template = "I want to open an {cuisine} restaurant. Suggest me a fancy name"
prompt = PromptTemplate.from_template(template)

chain = prompt | hf

print("chain",chain.invoke({"cuisine": "Indian"}))
