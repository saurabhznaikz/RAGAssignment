# from llama_index.core import SimpleDirectoryReader, Settings
# from llama_index.core.ingestion import IngestionPipeline
# import torch
# from llama_index.legacy.embeddings import HuggingFaceEmbedding
# from llama_index.legacy.llms import HuggingFaceInferenceAPI, HuggingFaceLLM
# from transformers import pipeline, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig
#
# from llama_index.core.extractors import (
#     SummaryExtractor,
#     QuestionsAnsweredExtractor,
#     TitleExtractor,
#     KeywordExtractor,
#     BaseExtractor,
# )
# from llama_index.core.node_parser import TokenTextSplitter
# from llama_index.llms.gemini import Gemini
#
# # llm = Gemini(api_key="AIzaSyAxE5pqRIOAjb3StF4BASGRXAxy_VDIcFo", model_name="models/gemini-pro")
# # model_name = "ai4bharat/indic-bert"
#
#
# # LLM
# llm = HuggingFaceInferenceAPI(
#     model_name="HuggingFaceH4/zephyr-7b-alpha",
#     tokenizer_name="HuggingFaceH4/zephyr-7b-alpha",
#     context_window=3900,
#     max_new_tokens=256,
#     # tokenizer_kwargs={},
#     generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
#     device_map="auto",
# )
#
# # Embedding
# embed_model = HuggingFaceEmbedding(
#     model_name="hkunlp/instructor-large"
# )
# # llm = HuggingFaceLLM(model_name=model_name)
# # llm = HuggingFaceLLM(model_name=llm)
# # Settings.llm = llm
#
# extractors = [
#     TitleExtractor(nodes=5, llm=llm),
#     QuestionsAnsweredExtractor(questions=3, llm=llm),
#     KeywordExtractor(keywords=5, llm=llm)
# ]
#
# documents = SimpleDirectoryReader("./sample_pdfs_rag/clean_data").load_data()
# text_splitter = TokenTextSplitter(
#     separator=" ", chunk_size=512, chunk_overlap=128
# )
#
# transformations = [text_splitter] + extractors
# pipeline = IngestionPipeline(transformations=transformations)
# document_nodes = pipeline.run(documents=documents)
# print(document_nodes[0].metadata)


from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.legacy.llms import HuggingFaceInferenceAPI, HuggingFaceLLM
from llama_index.llms.gemini import Gemini
from llama_index.core.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

# create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

# base node parser is a sentence splitter
text_splitter = SentenceSplitter()

llm = Gemini(api_key="AIzaSyAxE5pqRIOAjb3StF4BASGRXAxy_VDIcFo", model_name="models/gemini-pro")
# llm = HuggingFaceInferenceAPI(
#     model_id="deepset/roberta-base-squad2", task="question-answering")
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)

Settings.llm = llm
Settings.embed_model = embed_model
Settings.text_splitter = text_splitter

documents = SimpleDirectoryReader("./sample_pdfs_rag/clean_data").load_data()
nodes = node_parser.get_nodes_from_documents(documents)
base_nodes = text_splitter.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(nodes)
base_index = VectorStoreIndex(base_nodes)

query_engine = sentence_index.as_query_engine(
    similarity_top_k=4,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window")
    ],
)

window_response = query_engine.query(
    "What was the tip of the ice-berg?"
)

print(window_response)
