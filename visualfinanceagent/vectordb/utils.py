import os
from pdf2image import convert_from_path
from PIL import Image
import multiprocessing

def process_pdf(args):
    pdf_file, input_dir, output_dir = args
    pdf_path = os.path.join(input_dir, pdf_file)
    curr_output_dir = os.path.join(output_dir, pdf_file[:-4])  # Remove .pdf extension
    os.makedirs(curr_output_dir, exist_ok=True)
    
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        image_path = os.path.join(curr_output_dir, f'page_{i+1}.png')
        image.save(image_path, 'PNG')
    
    print(f"Processed: {pdf_file}")

def pdf_to_png_parallel(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]

    # Prepare arguments for multiprocessing
    args = [(pdf_file, input_dir, output_dir) for pdf_file in pdf_files]

    # Use all available CPU cores
    num_processes = multiprocessing.cpu_count()

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Map the process_pdf function to all arguments
        pool.map(process_pdf, args)

# if __name__ == '__main__':
#     input_dir = 'finance'  # Directory containing PDF files
#     output_dir = 'output_png2_files'
#     pdf_to_png_parallel(input_dir, output_dir)