from enum import Enum
from typing import Union

from pydantic import BaseModel, Field

import openai
from openai import OpenAI


class Query(BaseModel):
    ASL: str = Field(description='Restructured ASL string')


client = OpenAI()
sys='''
System Instructions for ChatGPT (API Invocation)
Input: Receive an English sentence that needs to be converted into ASL sentence structure.

Sentence Structure:

Convert the sentence to follow ASL grammar, which typically uses Topic-Comment structure.
Place WH-questions (what, who, where, etc.) at the end of the sentence.
Use subject-object-verb (SOV) structure when applicable.
Omit auxiliary verbs like "is," "are," "am," and articles like "the," "a," unless they are necessary for clarity.
Omission of Redundant Words:

Remove unnecessary words that do not contribute to the core meaning, such as auxiliary verbs, articles, or prepositions when possible.
Conversion Examples:

"What is your name?" ➡️ "YOUR NAME WHAT"
"Where are you going?" ➡️ "YOU GO WHERE"
"I am learning ASL." ➡️ "I LEARN ASL"
Output: Return only the ASL-structured sentence in uppercase for clarity.

Example Input and Output:

Input: "What is your name?"
Output: "YOUR NAME WHAT"
Restrictions: Always provide the response in simple uppercase text to simulate ASL emphasis.
'''

def chat(prompt):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": sys,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
         tools=[
        openai.pydantic_function_tool(Query),
    ],
    )
    return completion.choices[0].message.content

