from transformers import GenerationConfig


from .airllm_base import AirLLMBaseModel



class AirLLMQWen35(AirLLMBaseModel):


    def __init__(self, *args, **kwargs):


        super(AirLLMQWen35, self).__init__(*args, **kwargs)

    def get_use_better_transformer(self):
        return False



