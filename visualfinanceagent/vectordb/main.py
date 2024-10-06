import groq_utils, utils
import tqdm.asyncio
import asyncio
import os
import json
import time

BATCH_SIZE = 20

async def summarize(pdf_path:str,img_save_path:str="output_imgs"):
    #store the images
    utils.pdf_to_png_parallel(pdf_path,img_save_path)
    images_list = []
    for pdf_images in os.listdir(img_save_path):
        curr_path = os.path.join(img_save_path,pdf_images)
        for page in os.listdir(curr_path):
            page_path = os.path.join(curr_path,page)
            json_path = page_path.rsplit(".",1)[0]+".json"
            if not os.path.exists(json_path):
                images_list.append(page_path)
    
    for i in range(0, len(images_list),BATCH_SIZE):
        end = min(i+BATCH_SIZE,len(images_list))
        curr_img_list = images_list[i:end]
        results = [
                await _create_summary_task(
                    img
                )
                for img in tqdm.asyncio.tqdm(curr_img_list)
            ]
        for idx,res in enumerate(results):
            summary = res
            json_path = curr_img_list[idx].rsplit(".",1)[0]+".json"
            with open(json_path, 'w') as f:
                json.dump({"summary":summary[0]}, f)
        time.sleep(60)

def _create_summary_task(page_path):

    summary_task = asyncio.create_task(
            groq_utils.get_summary(page_path)
        )
    
    summaries = asyncio.gather(summary_task)
    return summaries

if __name__ == '__main__':
    out = asyncio.run(summarize("finance","output_imgs_2"))