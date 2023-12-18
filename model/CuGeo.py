from dataclasses import dataclass

import torch
from torch import nn
from transformers import CLIPModel, CLIPConfig, GPT2Model, GPT2Config, AutoConfig, AutoModel, AutoModelForCausalLM, MistralConfig, AutoTokenizer, GPT2Tokenizer, LlamaForCausalLM
from transformers.utils import ModelOutput

@dataclass
class CuGeoOutput(ModelOutput):
    """
    CuGeoOutput
    """
    operator_scores: torch.FloatTensor = None
    operand_scores: torch.FloatTensor = None
    operator_result: torch.LongTensor = None
    operand_result: torch.LongTensor = None

class CuGeo(nn.Module):
    def __init__(self,
                 encoder_model="openai/clip-vit-base-pa"
                               "tch32",
                 decoder_model="gpt2",
                 projection_layer_num=2,
                 classifier_layer_num=2,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        # Config
        self.num_operator = 18 + 1 # operator(18) + EOS
        self.num_operand = 7+12+3 + 1 # const(7) + number(12) + previous_result(3) + EOS
        self.max_equation_length = 4
        self.max_operand_length = 3


        # Set Config
        self.encoder_config = CLIPConfig.from_pretrained(encoder_model)
        self.decoder_config = GPT2Config.from_pretrained(decoder_model)
        # self.decoder_config = AutoConfig.from_pretrained(decoder_model, trust_remote_code=True)

        # CLIP encode (Image, text)
        self.encoder_config.text_config.max_position_embeddings = 200
        self.encoder = CLIPModel(self.encoder_config)

        # Encoder Parameter load
        # if kwargs['is_train'] or True:
        if True:
            pretrained_model = CLIPModel.from_pretrained(encoder_model)
            pretrained_state_dict = pretrained_model.state_dict()
            # pretrained 모델의 state_dict를 Current model의 state_dict에 update
            for name, param in self.encoder.named_parameters():
                if name in pretrained_state_dict and param.size() == pretrained_state_dict[name].size():
                    param.data.copy_(pretrained_state_dict[name])
        else:
            pass

        # Image, text projection
        if isinstance(self.decoder_config, GPT2Config):
            decoder_hidden_dim = self.decoder_config.n_embd
        elif isinstance(self.decoder_config, MistralConfig):
            decoder_hidden_dim = self.decoder_config.hidden_size
        else:
            raise NotImplementedError

        # TODO: change input shape
        # TODO: 7B models + LoRA

        self.image_projection = self.build_projection_layer(
            input_hidden_dim=self.encoder_config.projection_dim,
            output_hidden_dim=decoder_hidden_dim,
            num_of_layers=projection_layer_num
        )

        self.text_projection = self.build_projection_layer(
            input_hidden_dim=self.encoder_config.projection_dim,
            output_hidden_dim=decoder_hidden_dim,
            num_of_layers=projection_layer_num
        )

        # Embedding
        self.operator_embedding = nn.Embedding(self.num_operator, decoder_hidden_dim)

        # GPT decoder
        # self.decoder = GPT2Model.from_pretrained(decoder_model)
        self.decoder = AutoModel.from_pretrained(decoder_model, trust_remote_code=True)
        self.decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model)

        # operator (18 + EOS):
        # 'g_minus', 'g_sin', 'g_bili', 'g_add', 'g_tan', 'g_half', 'g_asin', 'g_double', 'cal_circle_perimeter',
        # 'g_equal', 'gougu_minus', 'g_divide', 'g_cos', 'cal_cone', 'cal_circle_area', 'g_mul', 'gougu_add', 'g_acos'

        # operator : # of operand
        # g_minus : 2
        # g_half : 1
        # g_double : 1
        # g_add : 2
        # g_equal : 1
        # gougu_add : 2
        # g_bili : 3
        # cal_cone : 2
        # gougu_minus : 2
        # g_mul : 2
        # g_sin : 1
        # g_divide : 2
        # g_cos : 1
        # g_tan : 1
        # cal_circle_area : 1
        # cal_circle_perimeter : 1
        # g_acos : 1
        # g_asin : 1

        self.operator_classifier = self.build_classifier_layer(
            hidden_dim=decoder_hidden_dim,
            output_dim=self.num_operator,
            num_of_layers=classifier_layer_num
            )

        # operand 22 + EOS:
        # canstant (7) : {'C_0', 'C_2', 'C_5', 'C_6', 'C_1', 'C_4', 'C_3'}
        # number (12) : {'N_0', 'N_2', 'N_5', 'N_6', 'N_1', 'N_4', 'N_3'}
        # previous_result (3) : {'V_1', 'V_0', 'V_2'}
        #
        self.gru = nn.GRU(decoder_hidden_dim, decoder_hidden_dim, batch_first=True)

        self.operand_classifier = self.build_classifier_layer(
            hidden_dim=decoder_hidden_dim,
            output_dim=self.num_operand,
            num_of_layers=classifier_layer_num
            )

    def build_projection_layer(self, input_hidden_dim, output_hidden_dim, num_of_layers:int=2):
        modules = [nn.Linear(input_hidden_dim, output_hidden_dim)]
        for _ in range(1, num_of_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_hidden_dim, output_hidden_dim))

        return nn.Sequential(*modules)

    def build_classifier_layer(self, hidden_dim, output_dim, num_of_layers):
        modules = []
        for _ in range(1, num_of_layers):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.GELU())
        modules.append(nn.Linear(hidden_dim, output_dim))

        return nn.Sequential(*modules)


    def forward(self, batch):
        # B : batch_size
        # T : max_equation_length
        # N : max_operand_length
        # H_E : Encoder(CLIP) embedding_dim
        # H_D : Decoder(GPT2) embedding_dim
        # H_g : gru embedding_dim

        encoder_result = self.encoder(**batch['encoder_input']) # {'text_embeds': [B, H_E], 'image_embeds': [B, H_E]}
        text_feature = torch.unsqueeze(self.text_projection(encoder_result['text_embeds']), dim=1) # [B, 1, H_E]
        image_feature = torch.unsqueeze(self.image_projection(encoder_result['image_embeds']), dim=1) # [B, H_E]

        tokenized_text_feature = torch.squeeze(self.decoder.wte(batch['decoder_input']['input_ids'])) # [B, 200, H_D]

        attention_mask = torch.squeeze(batch['decoder_input']['attention_mask'], dim=1)
        batch_size = attention_mask.shape[0]
        image_text_attention_mask = torch.ones(batch_size, 1+1).to(attention_mask.device)
        decoder_attention_mask = torch.cat([image_text_attention_mask, attention_mask], dim=1)

        decoder_input_embeds = torch.cat([image_feature, text_feature, tokenized_text_feature],
                                         dim=1)  # [B, 1+1+200(tokneizer max token length), H_D]

        for i in range(self.max_equation_length):
            # [B, T, H_D] -> [         for j in ranB, T, H_D]
            decoder_result = self.decoder(
                inputs_embeds=decoder_input_embeds,
                attention_mask=decoder_attention_mask
            )

            last_hidden_state = decoder_result.last_hidden_state

            # set next input & attention mask
            for j, (cur_embed, cur_attention) in enumerate(zip(last_hidden_state, decoder_attention_mask)):
                cur_last_gen_index = cur_attention.sum(dim=0).long()
                decoder_input_embeds[j, cur_last_gen_index, :] = last_hidden_state[j, cur_last_gen_index, :]
                decoder_attention_mask[j, cur_last_gen_index] = torch.tensor(1).to(cur_attention.device)
                if j == 0 :
                    decoder_output = torch.unsqueeze(last_hidden_state[j, cur_last_gen_index, :], dim=0)
                else:
                    decoder_output = torch.concat([decoder_output, torch.unsqueeze(last_hidden_state[j, cur_last_gen_index, :], dim=0)], dim=0)

            operator_logit = self.operator_classifier(decoder_output)
            # operator_result = torch.argmax(operator_logit, dim=-1)
            operator_score = torch.softmax(operator_logit, dim=-1)

            for j in range(self.max_operand_length):
                if j == 0:
                    operand_hidden_state, h_n = self.gru(
                        input=torch.unsqueeze(decoder_output, dim=1), # input: [B, 1, H_g], h_0: [1, B, H_g] -> ouput: [B, 1, H_g], h_n: [1, B, H_g]
                        hx=torch.unsqueeze(self.operator_embedding(torch.argmax(operator_score, dim=-1)), dim=0)
                    )
                    operand_logit = self.operand_classifier(torch.squeeze(operand_hidden_state))
                    operand_score = torch.softmax(operand_logit, dim=-1)
                    operand_scores_temp = torch.unsqueeze(operand_score, dim=1)
                else:
                    operand_hidden_state, h_n = self.gru(
                        input=operand_hidden_state,
                        hx=h_n
                    )
                    operand_logit = self.operand_classifier(torch.squeeze(operand_hidden_state))
                    operand_score = torch.softmax(operand_logit, dim=-1)
                    operand_scores_temp = torch.concat([operand_scores_temp, torch.unsqueeze(operand_score, dim=1)], dim=1)

            if i == 0:
                operator_scores = torch.unsqueeze(operator_score, dim=1)
                operand_scores = torch.unsqueeze(operand_scores_temp, dim=1)
            else:
                operator_scores = torch.concat([operator_scores, torch.unsqueeze(operator_score, dim=1)], dim=1)
                operand_scores = torch.concat([operand_scores, torch.unsqueeze(operand_scores_temp, dim=1)], dim=1)

        return CuGeoOutput(
            operator_scores=operator_scores,
            operand_scores=operand_scores,
            operator_result=torch.argmax(operator_scores, dim=-1),
            operand_result=torch.argmax(operand_scores, dim=-1)
        )


class CuDecoder(nn.Module):
    def __init__(self, decoder_model="gpt2", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = GPT2Model.from_pretrained(decoder_model)

    def forward(self, x):
        gpt_output = self.decoder(inputs_embeds=x)
        output_state = gpt_output['last_hidden_state']



if __name__ == '__main__':
    model = CuGeo()

    print(model)
