---
language:
- hi
license: apache-2.0
base_model: openai/whisper-small
tags:
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_17_0
metrics:
- wer
model-index:
- name: Whisper_Smal_FineTuned_Hindi - Yash_Ratnaker
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice 17
      type: Mozilla-foundation/common_voice_17_0
      config: hi
      split: test
      args: hi
    metrics:
    - name: Wer
      type: Wer
      value: 17.39228374836173
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper-Small-Finetuned-Hindi - Yash_Ratnaker

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the Common Voice 17 dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2933
- Wer Ortho: 34.1997
- Wer: 17.3923

## Model description

This model is based on the Whisper-small architecture, originally developed by OpenAI for automatic speech recognition (ASR). It was trained on a large amount of multilingual data to understand and transcribe speech in different languages.Originally trained by OpenAI on 680k hours of labeled multilingual and multitask supervised data, the Whisper model demonstrates a strong ability to generalize across languages and tasks. I fine-tuned this model using the Common Voice 17 Hindi dataset, which helps it better recognize and transcribe spoken Hindi

## Intended uses & limitations

This fine-tuned Whisper model is intended for automatic speech recognition in Hindi. It is suitable for transcribing spoken Hindi in various contexts, such as educational content, media transcription, and accessibility services. It excels in scenarios where clear audio with minimal background noise is available. However, the model may have limitations when dealing with highly noisy environments, overlapping speech, or dialects that were not well-represented in the training dataset. It is also less effective in real-time transcription where low latency is required.

## Training and evaluation data

The model was trained on the Common Voice 17 Hindi dataset, consisting of diverse speech samples from native Hindi speakers. This dataset provides a wide range of accents, pronunciations, and speech patterns, enabling the model to learn from a rich linguistic variety. The evaluation data was a subset of this dataset, carefully selected to represent different speakers and audio conditions, ensuring that the model's performance is robust and generalizes well to new, unseen data.

## Training procedure
Learning Rate: The learning rate was optimized to find a balance between fast convergence and stable training. The fine-tuning process utilized a lower learning rate than pre-training to ensure careful adjustments to the pre-trained weights.
Batch Size: A batch size that maximizes GPU utilization without overwhelming memory capacity was chosen. This helps in maintaining consistent training steps and reliable gradient updates across epochs.
Epochs: The model was trained for multiple epochs, iterating over the dataset to refine its parameters gradually. This allowed the model to converge effectively and improve its performance with each pass over the data.
Optimizer: The AdamW optimizer was selected for its adaptive learning rate capabilities, which help in efficiently managing the gradient descent process. It also includes a weight decay term to reduce the risk of overfitting.
Weight Decay: A small weight decay was applied during training to regularize the model and prevent overfitting. This was particularly important given the large capacity of the model and the relatively smaller size of the fine-tuning dataset compared to the original pre-training data.


### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant_with_warmup
- lr_scheduler_warmup_steps: 50
- training_steps: 1000
- mixed_precision_training: Native AMP

###Training output
 - global_step=1000, training_loss=0.23814286267757415, metrics={'train_runtime': 7575.8956, 'train_samples_per_second': 2.112, 'train_steps_per_second': 0.132, 'total_flos': 4.61563489271808e+18, 'train_loss': 0.23814286267757415, 'epoch': 2.247191011235955})

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Wer Ortho | Wer     |
|:-------------:|:------:|:----:|:---------------:|:---------:|:-------:|
| 0.1218        | 1.1236 | 500  | 0.2980          | 36.5846   | 19.0940 |
| 0.0623        | 2.2472 | 1000 | 0.2933          | 34.1997   | 17.3923 |


### Framework versions

- Transformers 4.42.4
- Pytorch 2.3.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1