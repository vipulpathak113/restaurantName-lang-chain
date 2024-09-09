from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "microsoft/Phi-3-mini-4k-instruct"

model = AutoModelForCausalLM.from_pretrained( 
    model_name,  
    device_map="auto",  
    torch_dtype="auto",  
    trust_remote_code=True,
    attn_implementation='eager'
) 

print("device_map",model.hf_device_map)

tokenizer = AutoTokenizer.from_pretrained(model_name) 

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,max_new_tokens=200,return_full_text =False )
llm = HuggingFacePipeline(pipeline=pipe)

from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

str_output_parser = StrOutputParser()

# Template to generate a single, fancy restaurant name
template_name = "I want to open a {cuisine} restaurant. Provide a unique and fancy name for it. Provide only one name with no additional text or explanation."
prompt_template_name = PromptTemplate(input_variables=["cuisine"], template=template_name)

# Chain for generating restaurant name
cuisine_name_chain = prompt_template_name | llm | str_output_parser

# Template to generate menu items based on restaurant name
template_items = "Suggest some menu items for the restaurant named {restaurant_name}. Provide list of only 10 menu items with no additional text, Instruction or explanation."
prompt_template_items = PromptTemplate(input_variables=["restaurant_name"], template=template_items)

# Chain for generating menu items
cuisine_menu_items_chain = prompt_template_items | llm | str_output_parser

# Function to generate the complete output
def generate_menu(cuisine):
    # Generate restaurant name
    restaurant_name = cuisine_name_chain.invoke({"cuisine": cuisine}).strip().split("Name: ")[1]
    
    # Generate menu items for the generated restaurant name
    menu_items = cuisine_menu_items_chain.invoke({"restaurant_name": restaurant_name}).strip()
    
    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items.split('\n')  # Split menu items into a list
    }
