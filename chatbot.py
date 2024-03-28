!pip install langchain
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "*****"
#enter your api key
!pip install huggingface_hub
from langchain import HuggingFaceHub
llm = HuggingFaceHub(repo_id="google/flan-t5-large" , model_kwargs={"temperature":0, 'max_length':64})
llm('translate English to German: How old are you?')
!pip install -q transformers einops accelerate langchain bitsandbytes
!huggingface-cli login
from huggingface_hub import login
login(token='*****')
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
import torch
import warnings
warnings.filterwarnings('ignore')

model="meta-llama/Llama-2-7b-chat-hf"
from huggingface_hub import notebook_login
notebook_login()
tokenizer=AutoTokenizer.from_pretrained(model)
pipeline=transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=1000,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
    )
llm=HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':1})
prompt="What would be a good name for a company that makes colorful socks"
print(llm(prompt))
from langchain.chains import LLMChain
prompt_template=PromptTemplate(input_variables=["cuisine"],
                               template="I want to open a restaurant for {cuisine} food. Suggest a fency name for this")
input_prompt=prompt_template.format(cuisine="Italian")
print(input_prompt)
prompt_template=PromptTemplate(input_variables=["book_name"],
                               template="Proivide me a concise summary of the book {book_name}")
chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
response= chain.run("Alchemist")
print(response)
