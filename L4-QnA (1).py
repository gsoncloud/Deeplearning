#!/usr/bin/env python
# coding: utf-8

# # LangChain: Q&A over Documents
# 
# An example might be a tool that would allow you to query a product catalog for items of interest.

# In[ ]:


#pip install --upgrade langchain


# In[ ]:


import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file


# Note: LLM's do not always produce the same results. When executing the code in your notebook, you may get slightly different answers that those in the video.

# In[ ]:


# account for deprecation of LLM model
import datetime
# Get the current date
current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

# Set the model variable based on the current date
if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else:
    llm_model = "gpt-3.5-turbo-0301"


# In[ ]:


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI


# In[ ]:


file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)


# In[ ]:


from langchain.indexes import VectorstoreIndexCreator


# In[ ]:


#pip install docarray


# In[ ]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


# In[ ]:


query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."


# **Note**:
# - The notebook uses `langchain==0.0.179` and `openai==0.27.7`
# - For these library versions, `VectorstoreIndexCreator` uses `text-davinci-003` as the base model, which has been deprecated since 1 January 2024.
# - The replacement model, `gpt-3.5-turbo-instruct` will be used instead for the `query`.
# - The `response` format might be different than the video because of this replacement model.

# In[ ]:


llm_replacement_model = OpenAI(temperature=0, 
                               model='gpt-3.5-turbo-instruct')

response = index.query(query, 
                       llm = llm_replacement_model)


# In[ ]:


display(Markdown(response))


# ## Step By Step

# In[ ]:


from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)


# In[ ]:


docs = loader.load()


# In[ ]:


docs[0]


# In[ ]:


from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# In[ ]:


embed = embeddings.embed_query("Hi my name is Harrison")


# In[ ]:


print(len(embed))


# In[ ]:


print(embed[:5])


# In[ ]:


db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)


# In[ ]:


query = "Please suggest a shirt with sunblocking"


# In[ ]:


docs = db.similarity_search(query)


# In[ ]:


len(docs)


# In[ ]:


docs[0]


# In[ ]:


retriever = db.as_retriever()


# In[ ]:


llm = ChatOpenAI(temperature = 0.0, model=llm_model)


# In[ ]:


qdocs = "".join([docs[i].page_content for i in range(len(docs))])


# In[ ]:


response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 


# In[ ]:


display(Markdown(response))


# In[ ]:


qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)


# In[ ]:


query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."


# In[ ]:


response = qa_stuff.run(query)


# In[ ]:


display(Markdown(response))


# In[ ]:


response = index.query(query, llm=llm)


# In[ ]:


index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])


# Reminder: Download your notebook to you local computer to save your work.

# In[3]:


from langchain.chains import RetrievalQA
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Dummy data: Drugs, therapeutic classes, and groups
drug_data = [
    {"drug": "Aspirin", "class": "Analgesic", "group": "Pain Relievers"},
    {"drug": "Ibuprofen", "class": "Anti-inflammatory", "group": "Pain Relievers"},
    {"drug": "Amoxicillin", "class": "Antibiotic", "group": "Antimicrobial Agents"},
    {"drug": "Metformin", "class": "Antidiabetic", "group": "Metabolic Agents"},
    {"drug": "Atorvastatin", "class": "Lipid-lowering Agent", "group": "Cardiovascular Agents"},
    {"drug": "Lisinopril", "class": "ACE Inhibitor", "group": "Cardiovascular Agents"}
]

# Step 1: Prepare documents for the vector database
documents = []
for entry in drug_data:
    documents.append(f"Drug: {entry['drug']}\nClass: {entry['class']}\nGroup: {entry['group']}")

# Split documents into chunks if they are too large (for large datasets)
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=50)
split_documents = []
for doc in documents:
    split_documents.extend(text_splitter.split_text(doc))

# Step 2: Create embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = DocArrayInMemorySearch.from_texts(split_documents, embedding=embeddings)

# Step 3: Define the retrieval QA system
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Top 3 matches

# Custom prompt template to include therapeutic groups
therapeutic_groups = "\n".join(
    set([entry["group"] for entry in drug_data])  # Unique groups
)
prompt_template = f"""
The following are the possible therapeutic groups:
{therapeutic_groups}

Use the provided drug description and context to determine the appropriate therapeutic group.
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question:
{question}
""",
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# Step 4: Query the system
query = "What therapeutic group does Aspirin belong to?"
response = qa_chain({"query": query})

print(f"Query: {query}")
print(f"Answer: {response['result']}")
print("\nRelevant Context:")
for doc in response["source_documents"]:
    print(doc.page_content)

# Step 5: Query for a new drug
new_drug_query = """
What therapeutic group does DrugX belong to? 
DrugX is a new anti-inflammatory drug used for pain relief.
"""
response = qa_chain({"query": new_drug_query})

print(f"\nQuery: {new_drug_query}")
print(f"Answer: {response['result']}")
print("\nRelevant Context:")
for doc in response["source_documents"]:
    print(doc.page_content)


# In[4]:


from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings

# Step 1: Create example data for your use case
# Drug descriptions and group assignments
texts = [
    "Group: Antihypertensives\nDescription: Drugs used to treat high blood pressure by relaxing blood vessels or reducing heart rate.\nExample: Drug A - Reduces blood pressure by acting on beta receptors.",
    "Group: Analgesics\nDescription: Drugs used to relieve pain by blocking pain signals to the brain.\nExample: Drug B - Used for severe pain management after surgery.",
    "Group: Antihistamines\nDescription: Drugs that treat allergies by blocking histamine receptors.\nExample: Drug C - Reduces symptoms of hay fever.",
    # Add more groups and descriptions here
]

# Step 2: Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = DocArrayInMemorySearch.from_texts(
    texts=texts, embedding=embeddings
)

# Step 3: Set up the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Step 4: Define a custom prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a drug classification expert tasked with determining the therapeutic group of a drug based on its description.

Context:
{context}

Question:
{question}

Provide the following:
1. The therapeutic group to which the drug belongs.
2. The rationale behind the classification, based on the context provided.
"""
)

# Step 5: Create a RetrievalQA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 relevant documents
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True,
)

# Step 6: Query the chain
query = "Classify the drug: This drug lowers blood pressure by acting as a beta blocker."
result = qa_chain({"query": query})

# Display the result
print("Answer:", result["result"])
print("\nSource Documents:")
for doc in result["source_documents"]:
    print(doc.page_content)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




