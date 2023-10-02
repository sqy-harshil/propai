import os
import re
import json
from typing import Optional

import requests
import pymongo
import openai
from bson.objectid import ObjectId
from langchain.callbacks import get_openai_callback

from config import CLASSIFIER_PARAMS


MONGODB_URI = os.getenv("MONGODB_URI")

client = pymongo.MongoClient(MONGODB_URI)

DIRTY_URL_ENDPOINT = (
    "https://sysearch-app4.squareyards.com/listing/open/v1.0/free-search"
)

functions = [
    {
        "name": "query_classifier",
        "description": "Checks if the given input query is of the category real-estate & expects property listings",
        "parameters": {
            "type": "object",
            "properties": {
                "is_real_estate": {
                    "type": "boolean",
                    "description": """
                        True if the query expects an answer that requires information about real-estate property details.
                    """,
                },
            },
            "required": ["is_real_estate"],
        },
    },
]


db = client.get_default_database()
messages = db["chat_history"]


def count_tokens(chain, query):
    """
    Description:
        This function shows how many tokens were used by a particular chain.

    Args:
        chain: BaseConversationalRetrievalChain
            This chain is used to build the context based answer for the user's message
        query: str
            This is the user's query which the chain has to generate an answer for

    Returns:
        result: str
            This is the answer to the user's question
        total_tokens: dict
            This is the number of tokens that were used by prompt + completion by the llm

    """

    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f"Spent a total of {cb.total_tokens} tokens")

    return result


def handle_price_unit(text: Optional[str]) -> Optional[str]:
    # Substitute 'crores' or 'crore' to 'cr'
    amount = re.sub(r"\bcrores?\b", "cr", text)
    return amount


def generate_dirty_url(search_keyword: str):
    """
    This function generates a dirty URL for a given search keyword and returns it with a summary.

    :param search_keyword: The search keyword is a string that represents the user's query or topic of
    interest. It is used to  generate a dirty URL that redirects the user to a website with relevant
    search results
    :type search_keyword: str
    :return: a string which is a dirty URL generated based on the search keyword provided as input. The
    dirty URL is obtained by making a POST request to a DIRTY_URL_ENDPOINT with the search keyword as a
    parameter. If the request is successful, the function adds a summary before the dirty URL and
    returns it. If the request fails or any exception occurs, an empty string is returned.
    """

    try:
        search_keyword = handle_price_unit(search_keyword)
        data = {"category_id": "2", "search_keyword": str(search_keyword)}
        dirty_url_response = requests.post(DIRTY_URL_ENDPOINT, json=data)
        if dirty_url_response.status_code == 200:
            dirty_url_response_json = dirty_url_response.json()
            redirect_dirty_url: str = dirty_url_response_json["result"][
                "redirectUrl"
            ].replace(" ", "%20")
        else:
            redirect_dirty_url = ""
    except Exception as e:
        print(f"Error while generating dirty url -: {e}")
        redirect_dirty_url = ""
    return redirect_dirty_url


def classify_message(question):
    # TODO: Improve the prompt for summarizing the categories covered in 'supplementary services'

    completion = openai.ChatCompletion.create(
        **CLASSIFIER_PARAMS,
        messages=[
            {
                "role": "system",
                "content": "Check if the query expects an answer that requires information about real-estate property details",
            },
            {"role": "user", "content": f"Query: {question}"},
        ],
        functions=functions,
        function_call={"name": "query_classifier"},
    )
    arguments = completion["choices"][0]["message"]["function_call"]["arguments"]
    return json.loads(arguments)
