from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings ,load_local_llm, load_AI21Lab_llm

from langchain.vectorstores import Pinecone 
from langchain_core.prompts import PromptTemplate
from langchain import LLMChain

# from langchain.prompts import PromptTemplate

from pinecone import Pinecone
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()

embeddings = download_hugging_face_embeddings()
# llm=load_local_llm()
llm=load_AI21Lab_llm()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')



def processing_fun(query,llm):

    pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
    index = pinecone_client.Index("testing")

    v= embeddings.embed_query(query)

    result=index.query(
        vector=v,
        top_k=4,
        include_values=True,
        include_metadata=True
    )

    cxt=f"Context 1: {result['matches'][0]['metadata']['text']} \n {result['matches'][1]['metadata']['text'] } \n {result['matches'][2]['metadata']['text'] }  \n {result['matches'][3]['metadata']['text'] }"
    print(f"===============>>>>> Context: {cxt}")
    
    # PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # chain = LLMChain(llm=llm, prompt=PROMPT)
    # result = chain.run({"context": cxt, "question":query})
    

    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    result=chain.invoke({"context": cxt, "question":query})

    return result


@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    result=processing_fun(input,llm)
    print(f"Result--->{result}")
    return str(result)



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)


