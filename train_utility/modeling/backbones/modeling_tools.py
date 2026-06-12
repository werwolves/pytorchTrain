import torch
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
from transformers import LayoutLMv3Config, LayoutLMv3ForTokenClassification, AutoModelForTokenClassification, AutoConfig
from transformers import LayoutLMv3Model,LayoutLMv3PreTrainedModel
from transformers import LayoutLMv3ClassificationHead
from transformers.utils import (
    auto_docstring,
    can_return_tuple,
    logging,
    torch_int,
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.processing_utils import Unpack



class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.layoutlmv3.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.layoutlmv3.set_input_embeddings(value)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        bbox: torch.LongTensor | None = None,
        pixel_values: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | SequenceClassifierOutput:
        r"""
        bbox (`torch.LongTensor` of shape `(batch_size, sequence_length, 4)`, *optional*):
            Bounding boxes of each input sequence tokens. Selected in the range `[0,
            config.max_2d_position_embeddings-1]`. Each bounding box should be a normalized version in (x0, y0, x1, y1)
            format, where (x0, y0) corresponds to the position of the upper left corner in the bounding box, and (x1,
            y1) represents the position of the lower right corner.

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForSequenceClassification
        >>> from datasets import load_dataset
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]

        >>> encoding = processor(image, words, boxes=boxes, return_tensors="pt")
        >>> sequence_label = torch.tensor([1])

        >>> outputs = model(**encoding, labels=sequence_label)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        outputs: BaseModelOutput = self.layoutlmv3(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            bbox=bbox,
            pixel_values=pixel_values,
            **kwargs,
        )

        sequence_output = outputs[0][:, 0, :]
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
























