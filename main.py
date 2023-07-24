# https://www.analyticsvidhya.com/blog/2021/03/a-beginners-guide-to-image-similarity-using-python/

# my (Michael Rodel)  API key is: bx4AHabZdoyhD2HUkbm7HSecspuCqplmUi81PrEJ



import numpy as np
from collections import Counter
from transformers import BlipProcessor, BlipForConditionalGeneration

import json
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import urllib.request as urllib
import io

from mtranslate import translate
import os


def gen_cap(img_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    f = open(img_path, 'rb')
    image = Image.open(f)

    raw_image = image.convert('RGB')

    # conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

def dist_img_calc(img_path_1, img_path_2):
    # Reading
    # the
    # Image
    img_1 = Image.open(img_path_1)
    img_2 = Image.open(img_path_2)

    img_1_arr = np.asarray(img_1)
    img_2_arr = np.asarray(img_2)

    flat_img_1 = img_1_arr.flatten()
    flat_img_2 = img_2_arr.flatten()

    # Generating the Count-Histogram-Vector
    RH1 = Counter(flat_img_1)
    RH2 = Counter(flat_img_2)

    H1 = []
    for i in range(256):
        if i in RH1.keys():
            H1.append(RH1[i])
        else:
            H1.append(0)

    H2 = []
    for i in range(256):
        if i in RH2.keys():
            H2.append(RH2[i])
        else:
            H2.append(0)

    return L2Norm(H1, H2)


def create_caption(raw_image):
    # This is a script which evaluates the images with the BLIP module
    # setup device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
    # this also loads the associated image processors
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True,
                                                         device=device)
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # generate caption (this caption is the output of the BLIP model)

    print("this is the caption of BLIP: ")
    print(model.generate({"image": image}))


def alon_api():

    # load sample images
    response_parse = 0
    text = ""
    text_dict = []
    for i in range(3):
        response = requests.get(
            "https://api.nli.org.il/openlibrary/search?api_key=AnGdUMDNPbU7IhCHgbreKF4Lou5spSCYklIFpWrc"
            "&query=any,contains,%D7%90%D7%A8%D7%9B%D7%99%D7%95%D7%9F%20%D7%93%D7%9F%20%D7%94%D7%93%D7%A0%D7%99"
            "&tab=default_tab&search_scope=Local&vid=NLI"
            "&facet=local7,include,%D7%90%D7%A8%D7%9B%D7%99%D7%95%D7%9F%20%D7%93%D7%9F%20%D7%94%D7%93%D7%A0%D7%99.&mfacet=rtype,include,photograph,1&mfacet=lang,include,zxx,2&lang=iw_IL&offset=0"
            "&material_type=photograph&output_format=json&result_page=" + str(i), verify=False)
        text += json.dumps(response.json(), indent=4)
        text_dict += json.loads(response.text)
    # print(text) CHANGED
    print(text)
    json_data = []
    for i in range(60):
        if "http://purl.org/dc/elements/1.1/relation" in text_dict[i]:
            response_parse = text.find("1.1/thumbnail", response_parse) + 1
            image_add = text_dict[i]["http://purl.org/dc/elements/1.1/thumbnail"][0]
            fd = urllib.urlopen(image_add.get("@value"))
            image_file = io.BytesIO(fd.read())
            raw_image = Image.open(image_file).convert("RGB")
            image_path = "DanHadani/images"
            raw_image.save(r'' + image_path + "/" + str(i) + '.png')
#           create_caption(raw_image)


            # in order to get the image description use the manifest API
            response_parse = text.find("1.1/relation", response_parse) + 1
            caption_request = text_dict[i]["http://purl.org/dc/elements/1.1/relation"][0]
            caption_response = requests.get(caption_request.get("@id"))

        # begin new
            caption_dict = json.loads(caption_response.text)
            label_value = caption_dict['sequences'][0]['label']
            # print("this is the 'real caption' before extracting the label: " + caption_text)
            # print("this is the real caption before translation: ")
            # print(label_value)
            print("this is the real caption (after translation): ")
            print(translate_cap(label_value))
            print()
            tmp = {}
            tmp["image_id"] = str(i)
            tmp["image"] = image_path + "/" + str(i) + '.png'
            tmp["caption"] = translate_cap(label_value)
            json_data.append(tmp)
        # end new
        # key: AnGdUMDNPbU7IhCHgbreKF4Lou5spSCYklIFpWrc
        with open('DanHadani/annotations/DanHadani_train.json', 'w') as outfile:
           json.dump(json_data, outfile)
        with open('DanHadani/annotations/DanHadani_val.json', 'w') as outfile:
         json.dump(json_data, outfile)
        with open('DanHadani/annotations/DanHadani_test.json', 'w') as outfile:
         json.dump(json_data, outfile)
def translate_cap(cap):
    # we use the API of Microsoft to translate the label to English

    translated_cap = translate(cap, "en")
    return translated_cap

def L2Norm(H1, H2):
    H1 = np.array(H1, dtype=np.int64)
    H2 = np.array(H2, dtype=np.int64)
    distance = 0
    for i in range(len(H1)):
        distance += np.square(H1[i] - H2[i])
    return np.sqrt(distance)

if __name__ == '__main__':
    # img_path_1 = 'TheWesternWall.png'
    # img_path_2 = 'LionsGate.png'
    # img_path_3 = 'TheCarmelForest.png'
    #
    # d12 = dist_img_calc(img_path_1, img_path_2)
    # d13 = dist_img_calc(img_path_1, img_path_3)
    # d23 = dist_img_calc(img_path_2, img_path_3)
    #
    # print("The distance between TheWesternWall and LionsGate is: {}".format(d12))
    # print("The distance between TheWesternWall and TheCarmelForest is: {}".format(d13))
    # print("The distance between LionsGate and TheCarmelForest is: {}".format(d23))

    alon_api()


