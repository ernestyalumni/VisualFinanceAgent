from byaldi import RAGMultiModalModel
from typing import Optional
import os

INDEX_NAME = "finance_data"

def build_colpali_index(img_path:str)->RAGMultiModalModel:
    RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

    for oimgs in os.listdir(img_path):
        metadata_list = []
        index_path = os.path.join(img_path,oimgs, "PNG")
        for img_file in os.listdir(index_path):
            metadata_list.append({"filename":oimgs,"page_num":img_file})
        
        if not os.path.exists(os.path.join(".byaldi",INDEX_NAME)):
            RAG.index(
                input_path=index_path, # The path to your documents
                index_name=INDEX_NAME, # The name you want to give to your index. It'll be saved at `index_root/index_name/`.
                store_collection_with_index=True, # Whether the index should store the base64 encoded documents.
                metadata=metadata_list,
                overwrite=True # Whether to overwrite an index if it already exists. If False, it'll return None and do nothing if `index_root/index_name` exists.
            )
        else:
            print("Adding to index")
            RAG.add_to_index(
                input_item=index_path,
                metadata=metadata_list,
                store_collection_with_index=True
            )
    return RAG

def load_index()->RAGMultiModalModel:
    search_index = RAGMultiModalModel.from_index(INDEX_NAME)
    return search_index

if __name__ == '__main__':
    rag = build_colpali_index("output_imgs_2")