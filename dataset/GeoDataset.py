from copy import deepcopy
from typing import Sequence, Dict

import sacrebleu
import transformers
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
import os

from transformers import CLIPProcessor, AutoProcessor, BatchEncoding, AutoTokenizer, GPT2Tokenizer

from const import PATCH_SIZE, IGNORE_INDEX, IMAGE_PATCH_TOKEN, TEXT_EMBEDDING_TOKEN, ENCODER_MAX_SEQ_LENGTH
from dataset.Equation import Equation

from utils.data_utils import process_image, create_patch, process_english_text
from PIL import Image



class GeoDataset(Dataset):
    def __init__(
            self,
            path: str,
            encoder_processor: transformers.ProcessorMixin,
            encoder_model: transformers.PreTrainedModel,
            decoder_tokenizer: transformers.PreTrainedTokenizer,
            split='train',
            raw_dataset=None,
            verbose=True,
            mode='train'
    ):
        super().__init__()
        self.path = Path(path)
        self.raw_dataset = raw_dataset
        self.verbose = verbose
        self.mode = mode

        # Loading datasets to data
        self.source = split.split(',')
        if self.verbose:
            print('Data source: ', self.source)

        # for preprocessing
        self.encoder_processor = encoder_processor
        self.decoder_tokenizer = decoder_tokenizer

        # for extracting text embedding
        self.encoder_model = encoder_model

        sub_dict_path = os.path.join(self.path, "sub_dataset_dict.pk")  # problems type
        with open(sub_dict_path, 'rb') as file:
            subset_dict = pickle.load(file)
        self.subset_dict = subset_dict

        # geo dataset
        target_text_list = []
        source_text_list = []
        image_list = []

        source_nums_list = []
        choice_nums_list = []
        label_list = []

        problem_form_list = []
        problem_type_list = []

        for source in self.source:
            with open(self.path.joinpath(f'{source}.pk'), "rb") as f:
                dataset = pickle.load(f)
                for sample in dataset:
                    if 'calculation' in source:
                        problem_with_space = process_english_text(sample['English_problem'])
                        problem_with_space = 'Calculation: ' + problem_with_space
                        source_text_list.append(problem_with_space)

                        text_i = " ".join(sample["manual_program"])
                        target_text_list.append(text_i)

                        image = sample['image']

                        # syP
                        ERROR_ANALYSIS_CASE = 0

                        if ERROR_ANALYSIS_CASE == 0:  # origin image
                            img_rgb = np.random.random((3, image.shape[0], image.shape[1]))
                            for i in range(3):
                                img_rgb[i, :, :] = image
                        elif ERROR_ANALYSIS_CASE == 1:  # white image
                            img_rgb = np.zeros((3, image.shape[0], image.shape[1]))
                        elif ERROR_ANALYSIS_CASE == 2:  # random image
                            img_rgb = (np.random.random((3, image.shape[0], image.shape[1])) * 255).astype(np.uint8)
                        elif ERROR_ANALYSIS_CASE == 3:  # origin + random image
                            img_rgb = (np.random.random((3, image.shape[0], image.shape[1])) * 255).astype(np.uint8)
                            for i in range(3):
                                img_rgb[i, :, :] = image

                        img_rgb = Image.fromarray(image, 'L')
                        image_list.append(img_rgb)

                        source_nums_list.append(sample["numbers"])
                        choice_nums_list.append(sample["choice_nums"])
                        label_list.append(sample["label"])

                        problem_form_list.append('calculation')
                        type = self.subset_dict[sample['id']]
                        problem_type_list.append(type)

                    else:
                        # TODO: add proving
                        assert 'proving' in source
                        problem_with_space = sample['input_text']
                        problem_with_space = 'Proving: ' + problem_with_space

                        source_text_list.append(problem_with_space)

                        text_i = ' '.join(sample['proving_sequence'])
                        target_text_list.append(text_i)

                        image = sample['img']
                        image = process_image(image)
                        image = image.transpose(2, 0, 1)
                        image_list.append(image)

                        source_nums_list.append(None)
                        choice_nums_list.append(None)
                        label_list.append(None)

                        problem_form_list.append('proving')
                        problem_type_list.append(sample['problem_type'])

        assert len(source_text_list) == len(target_text_list)

        data = []
        for source_text, target_text, image, source_nums, choice_nums, label, problem_form, problem_type in \
                zip(source_text_list, target_text_list, image_list, source_nums_list, choice_nums_list, label_list,
                    problem_form_list, problem_type_list):
            datum = {
                'image': image,
                'source_text': source_text.strip(),
                'target_text': target_text.strip(),
                'source_nums': source_nums,
                'choice_nums': choice_nums,
                'label': label,
                'problem_form': problem_form,
                'problem_type': problem_type,
            }
            data.append(datum)

        if self.verbose:
            print(f"Loaded {len(data)} data from", split)

        self.n_gpus = torch.cuda.device_count()

        self.data = data

        if self.verbose:
            print("# all sentences:", len(self.data))

    def __len__(self):
        return len(self.data)

    def _tokenize(self, strings: Sequence[str]) -> Dict:
        """Tokenize a list of strings."""
        tokenized_list = [
            self.decoder_tokenizer(
                text,
                return_tensors="pt"
            ) for text in strings
        ]
        input_ids= [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens  = [
            tokenized.input_ids.shape[-1]
            for tokenized in tokenized_list
        ]

        return dict(
            input_ids=input_ids,
            input_ids_lens=input_ids_lens,
        )

    def __getitem__(self, idx):
        datum = self.data[idx]
        problem_text = datum['source_text'] + "? The answer is "
        image = datum['image']

        inputs = self.encoder_processor(
            text=problem_text, images=image,
            return_tensors="pt", padding="max_length",
            max_length=ENCODER_MAX_SEQ_LENGTH,
            truncation=True
        )

        img = torch.squeeze(inputs.pixel_values, dim=0)
        image_token_length = (img.shape[1]//PATCH_SIZE) * (img.shape[2]//PATCH_SIZE)

        image = inputs['pixel_values']
        input_ids = inputs['input_ids'].to(self.encoder_model.device)
        attention_mask = inputs['attention_mask'].to(self.encoder_model.device)
        text_embedding = self.encoder_model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        # ready to image inputs
        source = [TEXT_EMBEDDING_TOKEN + IMAGE_PATCH_TOKEN * image_token_length + problem_text, datum['target_text']]
        input_text = ["".join(source)]

        tokenized_inputs = self._tokenize(input_text)
        input_ids = tokenized_inputs['input_ids'][0]
        target_ids = input_ids.clone()
        tokenized_target_list = self._tokenize(source)
        question_length = tokenized_target_list['input_ids_lens'][0]
        target_ids[:question_length-1] = IGNORE_INDEX

        return  dict(input_ids=input_ids,
                  label=target_ids,
                  image=image,
                  text_embed=text_embedding.detach())

    def collate_fn(self, instances : Sequence[Dict]):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "label"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.decoder_tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.decoder_tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            batch['images'] = torch.cat(images, dim=0)

        if 'text_embed' in instances[0]:
            text_embed = [instance['text_embed'] for instance in instances]
            batch['text_embed'] = torch.cat(text_embed, dim=0)

        return batch



class GeoEvaluator:
    def __init__(self):
        pass

    def evaluate(self, predicts, answers):
        try:
            bleu = sacrebleu.corpus_bleu(predicts, answers,
                                     lowercase=True)
        except EOFError:
            print('# preds', len(predicts))
            print('# tgts', len(answers))
            exit()
        return {
            'BLEU': bleu.score
        }


if __name__ == '__main__':
    split = ['train', 'val', 'test']
    # split = ['train']
    verbose = True
    mode = 'train'

    for s in split:
        print("calculation_" + s + " :")
        dataset = GeoDataset(
            "calculation_" + s,
            verbose=verbose,
            mode=mode)

        count = 0
        equation_count_list = []
        operator_set = set()
        constant_set = set()
        number_set = set()
        previous_result_set = set()
        operator_dict = dict()
        operand_cnt = 0
        cur_operator = None
        for datum in dataset.data:
            equation_count = 0
            target_text = datum['target_text'].split()

            cur_target_idx = []
            for i, cur in enumerate(target_text):
                if cur.startswith('g') or cur.startswith('cal_'):
                    cur_target_idx.append(i)
                    equation_count += 1

                if cur.startswith('g') or cur.startswith('cal_'):
                    operator_set.add(cur)
                elif cur.startswith('C'):
                    constant_set.add(cur)
                elif cur.startswith('N'):
                    number_set.add(cur)
                elif cur.startswith('V'):
                    previous_result_set.add(cur)
                else:
                    print(cur)
                    assert False
            equation_count_list.append(equation_count)

            for i, op_idx in enumerate(cur_target_idx):
                if target_text[op_idx] not in operator_dict:
                    if i != len(cur_target_idx) - 1:
                        operator_dict[target_text[op_idx]] = cur_target_idx[i + 1] - cur_target_idx[i] - 1

                    else:
                        operator_dict[target_text[op_idx]] = len(target_text) - cur_target_idx[i] - 1
                    print(f"{target_text[op_idx]} : {operator_dict[target_text[op_idx]]}")
                    # print(target_text)

            '''
            if datum['input']['input_ids'].shape[1] > 77:
                print(datum['input']['input_ids'].shape[1])
                count += 1
            if len(datum['target_ids']) > 77:
                print(len(datum['target_ids']))
            '''

        print(f'operator_set {len(operator_set)}:', operator_set)  # train: 18, val: 17, test: 17
        print(f'constant_set {len(constant_set)}:', constant_set)  # train:  7, val:  6, test:  6
        print(f'number_set: {len(number_set)}', number_set)  # train: 12, val: 10, test:  8
        print(f'previous_result_set: {len(previous_result_set)}', previous_result_set)  # train:  3, val:  3, test:  3
        print(f'max_equation_count: {max(equation_count_list)}')  # train:  4, val:  4, test:  4
    # print("Count:",count)
