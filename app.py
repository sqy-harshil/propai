import utils
import streamlit as st

from streaming import StreamHandler
from chatbot import (
    supplementary_services_chain,
    listings_chain,
    question_generator,
    listings_memory,
    listings_retriever,
    supplementary_memory,
)
from helper import classify_message

if "user_messages" not in st.session_state:
    st.session_state.user_messages = []


def add_user_message(message):
    st.session_state.user_messages.append(message)


st.header("Bella")
st.caption(
    "**Meet Bella, an intelligent chatbot made by SquareYards for quick, accurate real estate info and expert guidance. Get property details, advice, and more seamlessly!**"
)


class CustomDataChatbot:
    def __init__(self):
        self._response = ""
        self._user_message = ""

    @st.spinner("Thinking...")
    def setup_qa_chain(self, chain):
        qa_chain = chain
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        user_query = st.chat_input(placeholder="Ask me anything!")

        response = None

        if user_query:
            utils.display_msg(user_query, "user")

            IS_REAL_ESTATE = classify_message(user_query)

            print(IS_REAL_ESTATE)

            if IS_REAL_ESTATE:
                standalone_question_object = question_generator(
                    {
                        "question": user_query,
                        "chat_history": listings_memory.chat_memory.messages,
                    }
                )
                standalone_question = standalone_question_object["text"]
                relevant_context = listings_retriever.get_relevant_documents(
                    query=standalone_question.lower()
                )

                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    chain_response = listings_chain(
                        {
                            "question": standalone_question,
                            "context": relevant_context,
                        },
                        callbacks=[st_cb],
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": chain_response["text"]}
                    )

                    text_response = chain_response["text"]

                    listings_memory.chat_memory.add_user_message(standalone_question)
                    listings_memory.chat_memory.add_ai_message(text_response)
            else:
                utils.display_msg(user_query, "user")
                with st.chat_message("assistant"):
                    st_cb = StreamHandler(st.empty())
                    chain_response = supplementary_services_chain(
                        {
                            "question": user_query,
                            "chat_memory": supplementary_memory.chat_memory.messages,
                        },
                        callbacks=[st_cb],
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": chain_response}
                    )

                    relevant_context = chain_response["source_documents"]
                    text_response = chain_response["answer"]

                    supplementary_memory.chat_memory.add_user_message(user_query)
                    supplementary_memory.chat_memory.add_ai_message(text_response)


#             dirty_url = generate_dirty_url(user_query)

#             try:
#                 one_liner = smart_dirty_url(user_query)["url_one_liner"]
#                 short_description = smart_dirty_url(user_query)["short_description"]
#             except Exception:
#                 one_liner = "Discover similar properties as listed above"
#                 short_description = "Uncover a world where sophistication meets warmth in every property we present. With each interaction, you're one step closer to the realization of your ideal home. Follow your inquisitive nature and open the door to your next enchanting property."

#             if "Property Description" in response:
#                 with st.chat_message("assistant"):
#                     url_content = f""" **🔗[{one_liner}]({dirty_url})**

# {'_'+short_description+'_'}

# """
#                     st.write(url_content)

#                     st.session_state.messages.append(
#                         {"role": "assistant", "content": url_content}
#                     )

# # if "Property Description" in self.get_response():
# with st.form("my_form"):
#     st.write("**Enter your phone number to talk to our Property Experts**")
#     user_phone = st.text_input(
#         label="Enter a 10 digit number",
#         placeholder="Enter a 10 digit number",
#         label_visibility="hidden",
#     )

#     with st.expander("When would be a good time to call you?"):
#         preferred_date = st.date_input(
#             value=None,
#             label="_Your prefered date_",
#             format="DD/MM/YYYY",
#             help="Pick a date",
#         )
#         preferred_time = st.time_input(
#             "_Your preferred time_", value=None, help="Pick a time"
#         )
#     submitted = st.form_submit_button(
#         "**Book my slot**", disabled=False, type="primary"
#     )

# if submitted:
#     try:
#         if int(user_phone) > 7000000000 and int(user_phone) < 9999999999:
#             if str(preferred_date) != "":
#                 if str(preferred_time) != "":
#                     st.success(
#                         f"Thank you, your call is scheduled on {str(preferred_date)} at {str(preferred_time)}!"
#                     )
#                 else:
#                     st.warning("Please pick a time slot!")
#             else:
#                 st.warning("Please pick a date!")
#         else:
#             st.warning("Please enter a valid 10-digit phone number!")
#     except ValueError:
#         st.warning("Please enter a valid phone number!")


if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
