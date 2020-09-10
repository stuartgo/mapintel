import os
import shutil
from tesserocr import PyTessBaseAPI
from pdf2image import convert_from_path

img_path = os.path.join(".", "data", "ocr")

# TODO: VER PACKAGE PYTHON QUE CONVERTE PDFS COM TEXTO PARA TXT

# converts pdf files in the directory to png files
for file in os.listdir(img_path):
    if file.endswith(".pdf"):
        full_path = os.path.join(img_path, file)
        convert_from_path(full_path, output_folder=img_path, fmt='png')
        # shutil.move(full_path, os.path.join(img_path, 'transformed_pdfs/', file))

# extracts text from png files in the directory to txt files
with PyTessBaseAPI() as api:
    for file in os.listdir(img_path):
        if file.endswith(".png"):
            api.SetImageFile(os.path.join(img_path, file))
            full_path = os.path.join(".", "outputs", 'ocr', os.path.splitext(file)[0] + '.txt')
            txt_file = open(full_path, "w")
            txt_file.write(api.GetUTF8Text())
            txt_file.close()
        else:
            continue
