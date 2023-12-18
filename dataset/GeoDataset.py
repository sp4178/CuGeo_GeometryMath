import sacrebleu
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import pickle
import torch
import numpy as np
import os

from transformers import CLIPProcessor, AutoProcessor, BatchEncoding, AutoTokenizer, GPT2Tokenizer

from dataset.Equation import Equation

from utils.data_utils import process_image, create_patch, process_english_text
from PIL import Image


class GeoDataset(Dataset):
    def __init__(
            self,
            path=None,
            encoder_model="openai/clip-vit-base-patch32",
            decoder_model="gpt2",
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

        # Data preprocessing
        self.processor = CLIPProcessor.from_pretrained(encoder_model)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

        tokenizer_max_seq_length = 200
        # self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

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

        #syP
        input_list = []

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
                        image_list.append(img_rgb)

                        image = Image.fromarray(image, 'L')
                        inputs = self.processor(text=[sample['English_problem']], images=image,
                                           return_tensors="pt", padding="max_length", max_length=tokenizer_max_seq_length,
                                                truncation=True)

                        #syP
                        inputs.data['input_ids'] = torch.squeeze(inputs.data['input_ids'], 0)
                        inputs.data['attention_mask'] = torch.squeeze(inputs.data['attention_mask'], 0)
                        inputs.data['pixel_values'] = torch.squeeze(inputs.data['pixel_values'], 0)

                        #syP
                        input_list.append(inputs)

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
        for source_text, target_text, image, source_nums, choice_nums, label, problem_form, problem_type, model_input in \
                zip(source_text_list, target_text_list, image_list, source_nums_list, choice_nums_list, label_list, problem_form_list, problem_type_list, input_list):
            datum = {
                'image': image,
                'source_text': source_text.strip(),
                'target_text': target_text.strip(),
                'source_nums': source_nums,
                'choice_nums': choice_nums,
                'label': label,
                'problem_form': problem_form,
                'problem_type': problem_type,
                'input': model_input
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

    def __getitem__(self, idx):
        datum = self.data[idx]
        out_dict = {}

        target_text = datum['target_text']
        # processed_target = self.processor(text=target_text, return_tensors="pt", padding="max_length", max_length=100,
        #                                         truncation=True)
        target_equation = Equation(text=target_text)
        # out_dict['args'] = self.args
        # out_dict['target_text'] = target_text
        # out_dict['target_ids'] = torch.LongTensor(processed_target.data['input_ids'])
        # out_dict['target_length'] = len(processed_target.data['input_ids'])
        # decoder_input = BatchEncoding(data=self.decoder_tokenizer(datum['source_text'], return_tensors='pt', padding='max_length', max_length=200).data)
        decoder_input = self.decoder_tokenizer(datum['source_text'], return_tensors='pt').data
        input_ids = decoder_input['input_ids']
        attention_mask = decoder_input['attention_mask']
        decoder_input = BatchEncoding(
            data={
                # TODO: Change 200 to max sequence length
                "input_ids": torch.concat([input_ids, torch.unsqueeze(torch.tensor([0]*(300-input_ids.shape[-1])), dim=0)], dim=1), # 200 is max sequence length
                "attention_mask": torch.concat([attention_mask, torch.unsqueeze(torch.tensor([0]*(300-attention_mask.shape[-1])), dim=0)], dim=1)
            }
        )

        out_dict['input'] = BatchEncoding(
            data={'encoder_input': datum['input'],
                  'decoder_input': decoder_input}
            )
        out_dict['target'] = BatchEncoding(
            data={'operator_ids': target_equation.get_operator_ids(),
                'operand_ids': target_equation.get_operand_ids()})

        # out_dict['choice_nums'] = datum["choice_nums"]
        # out_dict['source_nums'] = datum["source_nums"]
        # out_dict['label'] = datum["label"]

        # out_dict['problem_form'] = datum["problem_form"]
        # out_dict['problem_type'] = datum["problem_type"]

        return out_dict


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
                        operator_dict[target_text[op_idx]] = cur_target_idx[i+1] - cur_target_idx[i] - 1

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

        print(f'operator_set {len(operator_set)}:', operator_set)   # train: 18, val: 17, test: 17
        print(f'constant_set {len(constant_set)}:', constant_set)   # train:  7, val:  6, test:  6
        print(f'number_set: {len(number_set)}', number_set)         # train: 12, val: 10, test:  8
        print(f'previous_result_set: {len(previous_result_set)}', previous_result_set)  # train:  3, val:  3, test:  3
        print(f'max_equation_count: {max(equation_count_list)}')   # train:  4, val:  4, test:  4
    # print("Count:",count)