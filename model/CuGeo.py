from dataclasses import dataclass

import torch
import transformers
from peft import PeftModelForCausalLM
from torch import nn
from transformers import GPT2Model, GPT2Config, MistralConfig, LlamaForCausalLM, LlamaConfig, MistralForCausalLM
from transformers.utils import ModelOutput

from model.phi2 import PhiConfig, PhiForCausalLM


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
                 encoder : transformers.PreTrainedModel,
                 encoder_config : transformers.PretrainedConfig,
                 decoder : transformers.PreTrainedModel,
                 decoder_tokenizer : transformers.PreTrainedTokenizer,
                 decoder_config : transformers.PretrainedConfig,
                 projection_layer_num=2,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)

        self.decoder = decoder
        self.encoder = encoder
        self.decoder_tokenizer = decoder_tokenizer
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        # Image, text projection
        if isinstance(self.decoder_config, LlamaConfig):
            decoder_hidden_dim = self.decoder_config.hidden_size
        elif isinstance(self.decoder_config, MistralConfig):
            decoder_hidden_dim = self.decoder_config.hidden_size
        elif isinstance(self.decoder_config, GPT2Config):
            decoder_hidden_dim = self.decoder_config.n_embd
        elif isinstance(self.decoder_config, PhiConfig):
            decoder_hidden_dim = self.decoder_config.hidden_size
        else:
            raise NotImplementedError

        self.mm_image_projection = self.build_projection_layer(
            input_hidden_dim=self.encoder_config.vision_config.hidden_size,
            output_hidden_dim=decoder_hidden_dim,
            num_of_layers=projection_layer_num
        )

        self.mm_text_projection = self.build_projection_layer(
            input_hidden_dim=self.encoder_config.projection_dim,
            output_hidden_dim=decoder_hidden_dim,
            num_of_layers=projection_layer_num
        )

    def build_projection_layer(self, input_hidden_dim, output_hidden_dim, num_of_layers:int=2):
        modules = [nn.Linear(input_hidden_dim, output_hidden_dim)]
        for _ in range(1, num_of_layers):
            modules.append(nn.GELU())
            modules.append(nn.Linear(output_hidden_dim, output_hidden_dim))

        return nn.Sequential(*modules)

    def forward(
            self,
            input_ids: torch.Tensor,        # [B, S]
            labels: torch.Tensor,           # [B, S]
            attention_mask = torch.Tensor,  # [B, S]
            images = torch.Tensor,          # [B, 3, 224, 224] : 224 X 224 RGB image
            text_embed = torch.Tensor,      # [B, H_C]
        ):
        # B     : batch size
        # S     : sequence length - 1(text_embed) + 256(patch) + S_Q + S_A
        # S_Q   : sequence length of S_Q
        # S_A   : sequence length of S_A
        # H_C   : clip hidden dim
        # H_D   : decoder hidden dim

        with torch.no_grad():
            vision_features = self.encoder.vision_model(
                pixel_values=images
            ) # {'text_embeds': [B, H_E], 'image_embeds': [B, H_E]}
        text_feature = torch.unsqueeze(self.mm_text_projection(text_embed), dim=1) # [B, 1, H_E]
        patch_features = self.mm_image_projection(vision_features.last_hidden_state[:, 1:, :]) # [B, 256(patch), H_E]

        if isinstance(self.decoder, LlamaForCausalLM):
            raise NotImplementedError
        elif isinstance(self.decoder, MistralForCausalLM):
            inputs_embeds = self.decoder.model.embed_tokens(input_ids).detach() # [B, 200, H_D]
        elif isinstance(self.decoder, GPT2Model):
            inputs_embeds = self.decoder.model.embed_tokens(input_ids).detach() # [B, 200, H_D]
        elif isinstance(self.decoder, PhiForCausalLM):
            inputs_embeds = self.decoder.base_model.embd(input_ids).detach()
        elif isinstance(self.decoder, PeftModelForCausalLM): # for Phi2-0 + LoRA
            inputs_embeds = self.decoder.base_model.model.base_model.embd(input_ids).detach()
        else:
            raise NotImplementedError

        text_feature_size = text_feature.shape[1]
        vision_patch_size = patch_features.shape[1]
        inputs_embeds[:,:text_feature_size,:] = text_feature
        inputs_embeds[:,text_feature_size:text_feature_size+vision_patch_size,:] = patch_features

        if isinstance(self.decoder, MistralForCausalLM):
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        else:
            output = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels
            )
        return output


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
