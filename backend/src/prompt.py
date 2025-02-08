from langchain_core.prompts import HumanMessagePromptTemplate, PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate

system_prompt = """You are an helpful AI assistant that provides smart suggestions for users given their to do lists. To do list may be provided as an image or just raw text, regardless your job is 
to provide the users with hints about the provided to do list. Your responses must be helpful and you must return a list of helpful suggestiosn to the user."""

image_prompt = ImagePromptTemplate(
    input_variables=["todo_img"],
    input_types={},
    partial_variables={},
    template={"url": "data:image/png;base64,{todo_img}"},
)
