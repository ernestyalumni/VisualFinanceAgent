from byaldi import RAGMultiModalModel
from typing import Optional

INDEX_NAME = "finance_data"

def build_index(metadata:Optional[list[dict[str,str]]]=None)->RAGMultiModalModel:
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

    RAG.index(
        input_path="finance/", # The path to your documents
        index_name=INDEX_NAME, # The name you want to give to your index. It'll be saved at `index_root/index_name/`.
        store_collection_with_index=True, # Whether the index should store the base64 encoded documents.
        # doc_ids=[0, 1, 2], # Optionally, you can specify a list of document IDs. They must be integers and match the number of documents you're passing. Otherwise, doc_ids will be automatically created.
        metadata=metadata,
        overwrite=True # Whether to overwrite an index if it already exists. If False, it'll return None and do nothing if `index_root/index_name` exists.
    )
    return RAG

def load_index()->RAGMultiModalModel:
    search_index = RAGMultiModalModel.from_index(INDEX_NAME)
    return search_index