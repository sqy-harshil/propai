import os
import requests
import ast

import openai
import regex as re
import streamlit as st
from typing import Optional

openai_engine = os.getenv("OPENAI_DEPLOYMENT_NAME")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_version = os.getenv("OPENAI_API_VERSION")

DIRTY_URL_ENDPOINT = (
    "https://sysearch-app4.squareyards.com/listing/open/v1.0/free-search"
)



def enable_chat_history(func):
    if os.environ.get("OPENAI_API_KEY"):
        # to clear chat history after swtching chatbot
        current_page = func.__qualname__
        if "current_page" not in st.session_state:
            st.session_state["current_page"] = current_page
        if st.session_state["current_page"] != current_page:
            try:
                st.cache_resource.clear()
                del st.session_state["current_page"]
                del st.session_state["messages"]
            except:
                pass

        # to show chat history on ui
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant", "content": "How can I help you?"}
            ]
        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    def execute(*args, **kwargs):
        func(*args, **kwargs)

    return execute


def display_msg(msg, author):
    """Method to display message on the UI

    Args:
        msg (str): message to display
        author (str): author of the message -user/assistant
    """
    st.session_state.messages.append({"role": author, "content": msg})
    st.chat_message(author).write(msg)



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
        # Parsing the search keywords to handle price unit
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

def smart_dirty_url(user_message):
    completion = openai.ChatCompletion.create(
        engine=openai_engine,
        temperature=0.7,
        messages=[
            {"role": "user", "content": user_message},
            {
                "role": "system",
                "content": "You are a marketing expert working for squareyards who knows how to win customers very well in real-estate. Generate a short and catchy one liner and a short description",
            },
        ],
        functions=[
            {
                "name": "smart_dirty_url",
                "description": "Generates a convincing one liner and a short description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url_one_liner": {
                            "type": "string",
                            "description": "This is the catchy and short one liner which will convince the customer for exploring more similar properties by clicking on this message as the user's query",
                        },
                        "short_description": {
                            "type": "string",
                            "description": "This is a short convincing description of how squareayards offers the best options according to user's query"
                        }
                    },
                },
            }
        ],
        function_call={"name": "smart_dirty_url"},
    )
    json = ast.literal_eval(completion["choices"][0]["message"]["function_call"]["arguments"])
    return json