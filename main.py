import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Model, Wav2Vec2Processor, Trainer, TrainingArguments
from dataclasses import dataclass
from typing import List, Dict, Union
import evaluate
import librosa
import numpy as np
from vocab import common_voice_train, common_voice_valid
# import torch.nn as nn
processor = Wav2Vec2Processor.from_pretrained("./processor")
matric = evaluate.load("wer")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m", 
    attention_dropout=0.1,
    layerdrop=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.75, 
    mask_time_length=10,
    mask_feature_prob=0.25,
    mask_feature_length=64,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_type_id,
    vocab_size=len(processor.tokenizer)
)

import evaluate
matric = evaluate.load("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    # Debug: Check prediction distribution
    unique, counts = np.unique(pred_ids, return_counts=True)
    print(f"Unique predicted IDs: {dict(zip(unique, counts))}")
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    
    print(f"Predictions: {pred_str[:3]}")  # First 3
    print(f"References: {label_str[:3]}")
    
    wer_score = matric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer_score}

model.freeze_feature_encoder() 
print("⚠️  Feature extractor frozen - training transformer + classification head")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on device: {device}")
model.to(device)


import librosa
def prepare_dataset(batch):
    # Load audio manually with librosa
    audio_array, sampling_rate = librosa.load(batch["path"], sr=16000)
    
    # Process the audio
    batch["input_values"] = processor(
        audio_array,
        sampling_rate=sampling_rate
    ).input_values[0]
    
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    
    return batch
@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

common_voice_train = common_voice_train.map(prepare_dataset, remove_columns=["file_id", "path", "transcription"])
split = common_voice_train.train_test_split(test_size=0.1, seed=42)
common_voice_train = split["train"]
common_voice_valid = split["test"]

# Optimized training arguments
training_args = TrainingArguments(
    output_dir="./wav2vec2-xlsr-khmer-300m",
    per_device_train_batch_size=16,      # Try 16, lower if OOM
    gradient_accumulation_steps=2,       # Effective batch = 32
    num_train_epochs=25,                 # 25 is realistic for ASR fine-tune
    learning_rate=5e-6,                  
    fp16=True,
    save_strategy="epoch",               # ✅ only save per epoch
    evaluation_strategy="epoch",         # ✅ evaluate per epoch
    logging_steps=200,
    save_total_limit=1,                  # ✅ keep last checkpoint
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    max_grad_norm=1.0,
    warmup_ratio=0.05,
    weight_decay=0.01,
    report_to="none",
)
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=common_voice_train,
    eval_dataset=common_voice_valid,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)


trainer.train()

