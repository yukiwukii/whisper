import os
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import torch
import evaluate
from datasets import load_dataset, DatasetDict, Audio, interleave_datasets
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed
)
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)
outdir = "./whisper-en-id-finetuned"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class WhisperTrainer:
    def __init__(self):
        self.set_environment()
        self.dataset = None
        self.processor = None
        self.model = None
        self.data_collator = None
        self.trainer = None
        
    def set_environment(self):
        set_seed(10)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            device_count = torch.cuda.device_count()
            logger.info(f"Using {device_count} CUDA device(s)")
        else:
            logger.info("CUDA not available, using CPU")
        os.makedirs(outdir, exist_ok=True)

    def load_dataset(self):
        logger.info("Loading dataset now")
        def _add_text(example):
            string = example['__key__']
            pattern = r'([^/]+)$'
            match = re.search(pattern, string).group(1)
            example['text'] = text_dict.get(match, None)
            return example

        self.dataset = DatasetDict()

        datasetID = load_dataset("speechcolab/gigaspeech2", data_dir='data/id/train', split='train[:10000]', trust_remote_code=True)
        labels_dataset = load_dataset(
            'speechcolab/gigaspeech2',
            data_files='data/id/train_raw.tsv',
            delimiter='\t',
            header=None,
            names=['utterance_id', 'text'],
            split='train', 
        )

        text_dict = {labels_dataset[i]['utterance_id']: labels_dataset[i]['text'] for i in range(len(labels_dataset))}
        datasetID = datasetID.map(_add_text)
        datasetID = datasetID.remove_columns(["__key__", "__url__"])

        datasetEN = load_dataset("speechcolab/gigaspeech", "s", split='train[:10000]', trust_remote_code=True)
        datasetEN = datasetEN.remove_columns(["segment_id", "speaker", "begin_time", "end_time", "audio_id", "title", "url", "source", "category", "original_full_path"])
        datasetEN = datasetEN.rename_column('audio', 'wav')
        datasetEN = datasetEN.cast(datasetID.features)

        self.dataset["train"] = interleave_datasets([datasetID, datasetEN], stopping_strategy='all_exhausted')

        datasetID = load_dataset("speechcolab/gigaspeech2", data_files='data/id/test.tar.gz', split='train[:5000]', trust_remote_code=True)
        labels_dataset = load_dataset(
            'speechcolab/gigaspeech2',
            data_files='data/id/test.tsv',
            delimiter='\t',
            header=None,
            names=['utterance_id', 'text'],
            split='train'
        )

        text_dict = {labels_dataset[i]['utterance_id']: labels_dataset[i]['text'] for i in range(len(labels_dataset))}
        datasetID = datasetID.map(_add_text)
        datasetID = datasetID.remove_columns(["__key__", "__url__"])

        datasetEN = load_dataset("speechcolab/gigaspeech", "s", split='test[:5000]', trust_remote_code=True)
        datasetEN = datasetEN.remove_columns(["segment_id", "speaker", "begin_time", "end_time", "audio_id", "title", "url", "source", "category", "original_full_path"])
        datasetEN = datasetEN.rename_column('audio', 'wav')
        datasetEN = datasetEN.cast(datasetID.features)
        
        self.dataset["test"] = interleave_datasets([datasetID, datasetEN], stopping_strategy='first_exhausted')
        
        self.dataset = self.dataset.cast_column("wav", Audio(sampling_rate=16000))
        logger.info(f"Dataset loaded: {len(self.dataset['train'])} training samples, "
                    f"{len(self.dataset['test'])} test samples")
        logger.info(self.dataset["train"])

    def load_model(self):
        logger.info("Loading model now")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Indonesian", task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.model.generation_config.language = "Indonesian" # Change this
        self.model.generation_config.task = "transcribe"
    
    def prepare_dataset(self):
        for split in self.dataset:
            self.dataset[split] = self.dataset[split].filter(
                lambda example: example["text"] is not None and example["text"] != ""
            )
        
        def _process(batch):
            # print("transcription is", batch['text'])
            audio = batch["wav"]
            batch["input_features"] = self.processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = self.processor.tokenizer(text=batch["text"]).input_ids
            return batch
        
        self.dataset = self.dataset.map(
            _process, 
            remove_columns=self.dataset.column_names['train'], 
            num_proc=4, 
            desc='Preparing dataset'
        )

    def setup_train(self):
        logger.info("Setting up training now")
        
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
        )
        metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
            pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        
            return {"wer": wer}
            
        training_args = Seq2SeqTrainingArguments(
            output_dir=outdir,
            per_device_train_batch_size=32,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            gradient_checkpointing=True,
            num_train_epochs=3,
            fp16=True,
            per_device_eval_batch_size=32,
            predict_with_generate=True,
            generation_max_length=225,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=25,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
        )
        
        self.trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["test"],
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
            tokenizer=self.processor.feature_extractor,
        )

    def train(self):
        logger.info("Training is starting")
        self.processor.save_pretrained(outdir)
        train_result = self.trainer.train()
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        self.trainer.save_model()

    def run(self):
        self.load_dataset()
        self.load_model()
        self.prepare_dataset()
        self.setup_train()
        self.train()
        logger.info("Loading is complete")


def main():
    trainer = WhisperTrainer()
    trainer.run()

if __name__ == "__main__":
    main()