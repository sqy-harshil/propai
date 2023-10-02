import pinecone
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone, FAISS
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.chains import ConversationalRetrievalChain, LLMChain

from config import (
    FAISS_VECTORSTORE_PATH,
    NUM_MESSAGES_TO_RETRIEVE,
    AZURE_GPT35_PARAMS,
    AZURE_GPT4_PARAMS,
    AZURE_ADA_PARAMS,
    PINECONE_PARAMS,
    INDEX_NAME,
)
from prompt import (
    COMBINE_DOCS_TEMPLATE,
    LISTINGS_CONDENSE_QUESTION_TEMPLATE,
    SUPPLEMENTARY_CONDENSE_QUESTION_TEMPLATE,
)
from helper import classify_message
from named_tuples import ChainResponse


gpt35 = AzureChatOpenAI(**AZURE_GPT35_PARAMS)
gpt4 = AzureChatOpenAI(**AZURE_GPT4_PARAMS)
ada_embeddings = OpenAIEmbeddings(**AZURE_ADA_PARAMS)

pinecone.init(**PINECONE_PARAMS)

listings_memory = ConversationBufferWindowMemory(k=5, return_messages=True)
supplementary_memory = ConversationBufferWindowMemory(k=5, return_messages=True)

metadata_field_info = [
    AttributeInfo(
        name="month",
        description="The month the listing was added",
        type="string",
    ),
    AttributeInfo(
        name="day",
        description="The day the listing was added",
        type="integer",
    ),
    AttributeInfo(
        name="propertyid", description="The unique id of the property", type="string"
    ),
    AttributeInfo(
        name="sublocalityname",
        description="This is the area where the property is located, sublocality can be identified as a location followed by east or west. Examples: andheri east, bandra west, ghatkopar east",
        type="string",
    ),
    AttributeInfo(
        name="locality",
        description="This is the area in which the property is located, For example andheri, wadala, juhu.",
        type="string",
    ),
    AttributeInfo(
        name="totalprice",
        description="This is the price of the property",
        type="integer",
    ),
    AttributeInfo(
        name="furnishing_status",
        description="This is the furnishing status of the property, which might be either unfurnished, semi-furnished, furnished",
        type="integer",
    ),
    AttributeInfo(
        name="areainsqft",
        description="This is the size of the property in square feet or sq ft",
        type="integer",
    ),
    AttributeInfo(
        name="number_of_bathroom",
        description="This is the number of bathrooms in the property",
        type="integer",
    ),
    AttributeInfo(
        name="number_of_rooms",
        description="This is the number of rooms in the property, usually mentioned as a number before 'BHK'",
        type="integer",
    ),
    AttributeInfo(
        name="cityname",
        description="Name of the city",
        type="string",
    ),
]
document_content_description = "Brief summary about the listings attribute"


def get_listings_retriever():
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=ada_embeddings
    )
    self_query_retriever = SelfQueryRetriever.from_llm(
        gpt4,
        docsearch,
        document_content_description,
        metadata_field_info,
        use_original_query=False,
        verbose=True,
        enable_limit=True,
        search_kwargs={"k": NUM_MESSAGES_TO_RETRIEVE},
    )
    return self_query_retriever


listings_retriever = get_listings_retriever()


def get_supplementary_retriever():
    docsearch = FAISS.load_local(
        folder_path=FAISS_VECTORSTORE_PATH,
        embeddings=ada_embeddings,
    )

    return docsearch.as_retriever()


supplementary_retriever = get_supplementary_retriever()


# Supplementary Chain
combined_docs_template = PromptTemplate(
    template=COMBINE_DOCS_TEMPLATE, input_variables=["context", "question"]
)

condense_question_prompt = PromptTemplate(
    template=SUPPLEMENTARY_CONDENSE_QUESTION_TEMPLATE,
    input_variables=["chat_history", "question"],
)
supplementary_services_chain = ConversationalRetrievalChain.from_llm(
    llm=gpt35,
    retriever=get_supplementary_retriever(),
    verbose=True,
    condense_question_prompt=condense_question_prompt,
    combine_docs_chain_kwargs={"prompt": combined_docs_template},
    return_generated_question=True,
    return_source_documents=True,
)


# Listings Chain
combined_docs_template = PromptTemplate(
    template=COMBINE_DOCS_TEMPLATE, input_variables=["context", "question"]
)

condense_question_prompt = PromptTemplate(
    template=LISTINGS_CONDENSE_QUESTION_TEMPLATE,
    input_variables=["chat_history", "question"],
)

question_generator = LLMChain(
    llm=gpt4,
    prompt=condense_question_prompt,
)

listings_chain = LLMChain(
    llm=gpt35,
    prompt=combined_docs_template,
    verbose=True,
)


def get_smart_response(question):
    if classify_message(question):
        standalone_question_object = question_generator(
            {
                "question": question,
                "chat_history": listings_memory.chat_memory.messages,
            }
        )

        standalone_question = standalone_question_object["text"]
        relevant_context = listings_retriever.get_relevant_documents(
            query=standalone_question.lower()
        )
        chain_response = listings_chain(
            {
                "question": standalone_question,
                "context": relevant_context,
            },
        )
        text_response = chain_response["text"]

        listings_memory.chat_memory.add_user_message(standalone_question)
        listings_memory.chat_memory.add_ai_message(text_response)

        return ChainResponse(
            response=text_response,
            sources=relevant_context,
        )

    chain_response = supplementary_services_chain(
        {
            "question": question,
            "chat_memory": supplementary_memory.chat_memory.messages,
        },
    )

    relevant_context = chain_response["source_documents"]
    text_response = chain_response["answer"]

    supplementary_memory.chat_memory.add_user_message(question)
    supplementary_memory.chat_memory.add_ai_message(text_response)

    return ChainResponse(
        response=text_response,
        sources=relevant_context,
    )
