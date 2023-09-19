# https://www.analyticsvidhya.com/blog/2021/03/a-beginners-guide-to-image-similarity-using-python/

# my (Michael Rodel)  API key is: bx4AHabZdoyhD2HUkbm7HSecspuCqplmUi81PrEJ


import numpy as np
from collections import Counter

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from transformers import BlipProcessor, BlipForConditionalGeneration

from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms

import json
import requests
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image
import urllib.request as urllib
import io

from mtranslate import translate
import os
# import pandas lib as pd
import pandas as pd
import openpyxl


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


def resize(image):
    # Define the target image size
    target_size = (384, 384)

    # # Create a transformation to resize and normalize the image
    # transform = transforms.Compose([
    #     transforms.Resize(target_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])
    # resized_image = transform(image)
    # return transforms.ToPILImage()(resized_image)

    resized_image = image.resize(target_size)
    return resized_image


def get_and_save_data(number_of_samples):
    # load sample images
    row_number = 1
    error_counter = 0
    picture_count = 0
    text = ""
    text_dict = []
    caption_file = pd.read_excel(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/caption_file.xlsx',
                                 usecols=[0, 4])
    print(text)
    json_data = []
    test_json_data = []
    coco_format_data = dict()
    coco_format_data["annotations"] = []
    coco_format_data["images"] = []
    while picture_count < number_of_samples:
        try:
            relevant_data = caption_file.iloc[row_number]
            row_number += 1
            label_value = str(relevant_data["caption"])
            if "Photo shows:" in label_value:
                label_value = label_value[label_value.find("Photo shows:") + len("Photo shows: "):label_value.find(".")]
            elif "Photo shows" in label_value:
                label_value = label_value[label_value.find("Photo shows") + len("Photo shows "):label_value.find(".")]
            else:
                label_value = ""
            if label_value != "" and " " in label_value:
                image_record = relevant_data["record"]
                response = requests.get("http://iiif.nli.org.il/IIIFv21/" + str(image_record) + "/manifest", verify=False)
                text = json.dumps(response.json(), indent=4)
                text_dict = json.loads(response.text)
                if len(text_dict) > 0:
                    image_add = text_dict["sequences"][0]["canvases"][0]["images"][0]["resource"]["@id"]
                    fd = urllib.urlopen(image_add)
                    image_file = io.BytesIO(fd.read())
                    raw_image = Image.open(image_file).convert("RGB")

                    # resize the image
                    raw_image = resize(raw_image)
                    # create_caption(raw_image)
                    # print("this is the 'real caption' before extracting the label: " + caption_text)
                    # print("this is the real caption before translation: ")
                    # print(label_value)
                    print("this is the real caption: ")
                    print(label_value)
                    print()
                    # save the image
                    image_path = os.path.dirname(os.path.abspath(__file__)) + "/DanHadani/images"
                    raw_image.save(r'' + image_path + "/" + str(picture_count) + '.jpg')
                    # add the annotation to the variable json_data
                    tmp = {}
                    tmp["image_id"] = picture_count
                    tmp["image"] = image_path + "/" + str(picture_count) + '.jpg'
                    tmp["caption"] = label_value
                    json_data.append(tmp)
                    tmp = {}
                    tmp["image_id"] = picture_count
                    tmp["caption"] = label_value
                    tmp["id"] = picture_count
                    coco_format_data["annotations"].append(tmp)
                    tmp = {}
                    tmp["id"] = picture_count
                    coco_format_data["images"].append(tmp)

                    picture_count += 1
                    error_counter = 0
        except:
            print("invalid image format")
            error_counter += 1
            if error_counter > 10:
                row_number += 1000

    train, test = train_test_split(json_data, test_size=0.1)
    for item in test:
        tmp = {}
        tmp["image"] = item["image"]
        tmp["caption"] = item["caption"]
        test_json_data.append(tmp)
    # save the annotations in a json file
    with open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/annotations/DanHadani_train.json',
              'w') as outfile:
        json.dump(train, outfile)
    with open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/annotations/DanHadani_val.json', 'w') as outfile:
        json.dump(test_json_data, outfile)
    with open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/annotations/DanHadani_test.json',
              'w') as outfile:
        json.dump(test_json_data, outfile)
    with open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/results/DanHadani_eval_real_captions.json',
                  'w') as outfile:
            json.dump(coco_format_data, outfile)
        # key: AnGdUMDNPbU7IhCHgbreKF4Lou5spSCYklIFpWrc


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


def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}

    # changing the keys' type of ref to int
    ref = {int(k): v for k, v in ref.items()}

    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def get_results():
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/annotations/DanHadani_val.json', )
    raw_ref = json.load(f)
    ref = dict()
    for item in raw_ref:
        ref[item["image_id"]] = []
        ref[item["image_id"]].append(item["caption"])
    f = open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/results/DanHadani_computed_captions.json', )
    raw_hypo = json.load(f)
    hypo = dict()
    for item in raw_hypo:
        hypo[item["image_id"]] = []
        hypo[item["image_id"]].append(item["caption"])
    return ref, hypo


def print_scores():
    ref, hypo = get_results()
    scores = calc_scores(ref, hypo)
    for method, score in scores.items():
        print("The score for method " + method + " is: " + str(score))


def print_results():
    ref, hypo = get_results()
    for id in ref:
        ref_cap = str(ref[id])
        ref_cap = ref_cap[2:len(ref_cap) - 2]
        hypo_cap = str(hypo[id])
        hypo_cap = hypo_cap[2:len(hypo_cap) - 2]
        image = Image.open(os.path.dirname(os.path.abspath(__file__)) + '/DanHadani/images/' + id + '.png')
        print("Picture id: " + id)
        print("Real caption: " + ref_cap)
        print("Computed caption: " + hypo_cap)
        image.show()


if __name__ == '__main__':
    get_and_save_data(number_of_samples=10)
    # print_scores()
    # print_results()

