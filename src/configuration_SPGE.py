from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SPGEConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size=256,
        intermediate_size=256*4,
        hidden_bias=False,
        hidden_dropout=0.0,



        num_experts=4,
        num_experts_per_tok=1,
        use_aux_loss=True,
        **kwargs
    ):

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_bias = hidden_bias
        self.hidden_dropout = hidden_dropout


        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.use_aux_loss = use_aux_loss

        super().__init__(
            **kwargs,
        )



    @property
    def shared_expert_intermediate_size(self):
        return self.intermediate_size
    
    @property
    def private_expert_intermediate_size(self):
        return self.intermediate_size // 2
