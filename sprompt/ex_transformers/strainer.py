from torch import nn
import torch
from transformers import Trainer
# from transformers.trainer import _is_peft_model
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from transformers.utils import (
#     is_apex_available,
#     is_sagemaker_mp_enabled,
# )
# from transformers.models.auto.modeling_auto import (
#     MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
# )
# from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat


class STrainer(Trainer):

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        # inputs['input_ids'] = inputs['input_ids'].float()
        # inputs['input_ids'].requires_grad = True

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)


        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss)
        # self.accelerator.backward(loss, retain_graph=True)

        # grads = torch.autograd.grad(loss, inputs['input_ids'], retain_graph=True)[0]
        #
        # print(grads)

        del inputs

        return loss.detach() / self.args.gradient_accumulation_steps

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # if self.label_smoother is not None and "labels" in inputs:
        #     labels = inputs.pop("labels")
        # else:
        #     labels = None
        #
        # TODO: 1 Original Prediction
        outputs_original = model(**inputs)

        # TODO: 2 Mask
        # feature_mask =

        # TODO: 2 Masked Prediction
        # attention_mask = input_ids.ne(tokenizer.pad_token_id) # finetune.py中计算的padding mask
        inputs['attention_mask'] = inputs['attention_mask'] * feature_mask
        outputs_masked = model(**inputs)

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]


        return (loss, outputs) if return_outputs else loss
