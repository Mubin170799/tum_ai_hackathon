# Import Library
import pandas as pd
import os
import tiktoken
# from dotenv import load_dotenv

from langchain.chains import RetrievalQA, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


import os
os.environ['OPENAI_API_KEY'] = "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"


# Data Loading
df = pd.read_csv('vehicle_example.csv')

# Combine
df['combined_info'] = df.apply(lambda row: f"Chassis_Number__c: {row['Chassis_Number__c']}. Exterior_Color__c: {row['Exterior_Color__c']}. Interior_Color__c: {row['Interior_Color__c']}. List_Price__c: {row['List_Price__c']}. Steering_Type__c : {row['Steering_Type__c']}. Vehicle_Definition__c: ${row['Vehicle_Definition__c']}. Name: {row['Name']}", axis=1)

# Load Processed Dataset
loader = DataFrameLoader(df, page_content_column="combined_info")
docs  = loader.load()

# Document splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# embeddings model
embeddings = OpenAIEmbeddings()

# Vector DB
vectorstore  = FAISS.from_documents(texts, embeddings)
vs_retriver = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5, "k": 2})

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)


question_asking_prompt = """
You are mercedes car sales person who are expert in car knowlegde. A customer started with this question "<question>". You have to ask customer
5 follow up questions to better understand their needs. About Exterior color, Interior color, List Price, Steering Type(hybrid, electric, combustion), car_type(sedan or SUV) . Start with <question_start> and after five questions end it with <question_end>. The questions should be ask all in one go.

Your Response:
"""


def extract_question(text):
    questions = text.split("<question_start>")[1].strip().split("<question_end>")[0].strip()
    qs = questions.split("\n")
    return [q.strip() for q in qs]

def get_conversation(questions):
    conversation, answers = "", []
    for question in questions:
        answer = str(input(f"{question}:\n")).strip()
        answers.append(answer)
        conversation += question +"\n" + answer+"\n\n"
    return conversation, answers

final_prompt = """
You are mercedes car sales person who are expert in car knowlegde.
Here's the conversation that you had with them:
<conversation>

Here's some car informations based on the user's intent
<context>

Your goal is to suggest the best 2 cars. Take all the information you have given into account and think carefully and thoroughly before suggesting

All the suggested cars will be in json format with following details Chassis_Number__c, Vehicle_Definition__c, Name, price, steering_type,.

Your Response:
"""


def run(question: str, verbose=False):

    in_prompt = question_asking_prompt.replace("<question>", question).strip()
    if verbose:
        print(f"Input Prompt:\n{in_prompt}\n\n")

    out_text = llm.invoke(in_prompt)
    if verbose:
        print(f"LLM's output:\n {out_text}\n\n")

    questions = extract_question(out_text.content)
    if verbose:
        print(f"Questions:\n{questions}\n\n")

    conversation, answers = get_conversation(questions)
    if verbose:
        print(f"Conversation:\n{conversation}\n\n")

    context = ""
    for ans in answers:
        retrived_doc = vs_retriver.invoke(ans)
        doc_str = ""
        for i, doc in enumerate(retrived_doc, 1):
            doc_str += f"\t{i}. {str(doc.page_content)}\n"
        context += f"\n{doc_str}\n"

    out_prompt = final_prompt.replace("<conversation>", conversation).replace("<context>", context).strip()
    if verbose:
        print(f"Final prompt:\n{out_prompt}\n\n")

    final_output = llm.invoke(out_prompt)
    return final_output.content

# out = run("Hi!", verbose=True)
# print(out)