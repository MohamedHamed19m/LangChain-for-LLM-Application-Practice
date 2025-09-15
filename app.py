# requirements:
# pip install python-dotenv langchain-google-genai langchain

import os
import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Load API key from .env
load_dotenv()

current_date = datetime.datetime.now().date()
target_date = datetime.date(2025, 6, 12)

# if current_date > target_date:
#     llm_model = "gemini-2.5-flash-lite"
# else:
llm_model = "gemini-2.5-flash"

# Initialize Gemini LLM
chat:ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model=llm_model,
    temperature=0.0,
    convert_system_message_to_human=True,
)

# --------------------------
# Direct Gemini call
# --------------------------
def get_completion(prompt, model=chat):
    messages = [{"role": "user", "content": prompt}]
    response = model.invoke(messages)
    print("test ##",response.response_metadata)
    return response.content

print(get_completion("What is 1+1?"))

# --------------------------
# Prompt templates
# --------------------------
template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```"""

prompt_template = ChatPromptTemplate.from_template(template_string)

customer_style = "American English in a calm and respectful tone"
customer_email = """Arrr, I be fuming that me blender lid
flew off and splattered me kitchen walls with smoothie!"""

customer_messages = prompt_template.format_messages(
    style=customer_style, text=customer_email
)

customer_response = chat(customer_messages)
print(customer_response.content)

# # --------------------------
# # Output parsers
# # --------------------------
# review_template = """For the following text, extract the following information:

# gift: Was the item purchased as a gift for someone else? \
# Answer True if yes, False if not or unknown.

# delivery_days: How many days did it take for the product to arrive? \
# If this information is not found, output -1.

# price_value: Extract any sentences about the value or price, \
# and output them as a comma separated Python list.

# Format the output as JSON.

# text: {text}

# {format_instructions}
# """

# customer_review = """This leaf blower is pretty amazing. It has four settings:
# candle blower, gentle breeze, windy city, and tornado.
# It arrived in two days, just in time for my wife's anniversary present.
# It's slightly more expensive than the others, but worth it!"""

# gift_schema = ResponseSchema(name="gift", description="Was the item purchased as a gift?")
# delivery_days_schema = ResponseSchema(name="delivery_days", description="Days for delivery")
# price_value_schema = ResponseSchema(name="price_value", description="Value/price statements")

# response_schemas = [gift_schema, delivery_days_schema, price_value_schema]
# output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# format_instructions = output_parser.get_format_instructions()
# prompt = ChatPromptTemplate.from_template(review_template)

# messages = prompt.format_messages(
#     text=customer_review, format_instructions=format_instructions
# )

# response = chat(messages)
# print("Raw output:", response.content)

# output_dict = output_parser.parse(response.content)
# print("Parsed:", output_dict)
