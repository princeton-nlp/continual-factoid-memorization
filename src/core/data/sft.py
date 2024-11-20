import re
import os
import json
import yaml
import copy
import random
import itertools
import importlib
from pathlib import Path
from enum import Enum, unique
from collections import defaultdict, Counter

from tqdm import tqdm
from rich import print

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

from src.config import BIG_NUM


def load_dataset_module(dataset_module_path):
    module = importlib.import_module(dataset_module_path)
    return getattr(module, 'DatasetMapping')


class FinetuningDataset(Dataset):
    def __init__(self,
        tokenizer=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        sample_ids=None,
    ):
        self.tokenizer = tokenizer
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        input_ids = self.input_ids[index]
        return dict(
            input_ids=self.input_ids[index],
            attention_mask=self.attention_mask[index],
            labels=self.labels[index],
            sample_ids=self.sample_ids[index],
        )


class FinetuningDataManager:
    def __init__(
        self,
        dataset_name: str = None,
        dataset_module_path: str = 'src.datasets',
        tokenizer: AutoTokenizer = None,
        max_seq_length: int = None,
        train_num_examples: int = BIG_NUM,
        dev_num_examples: int = 100,
        use_train_for_dev: bool = False,
        add_icl: bool = False,
        **kwargs,
    ):
        """
        use_train_for_dev: if True, dev will be the subset of train
        add_icl: if True, add ICL examples to the training/dev set using the ICL config in core/data/configs/icl_config.yaml
        """
        self.dataset_name = dataset_name
        self.max_seq_length = max_seq_length

        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer
    
        self.train_num_examples = train_num_examples
        self.dev_num_examples = dev_num_examples 
        self.global_sample_id = 0
        self.id_to_sample = dict()
        self.eval_known = kwargs.get('eval_known', False)

        if use_train_for_dev:
            # NOTE: this tests memorization
            dev_start, dev_end = 0, dev_num_examples
            train_start, train_end = 0, train_num_examples
        else:
            # NOTE: this tests in-domain generalization
            dev_start, dev_end = 0, dev_num_examples
            train_start, train_end = dev_end, dev_end + train_num_examples

        kwargs['finetuning_data_manager'] = self
        dataset_module = load_dataset_module(dataset_module_path)
        if self.eval_known:
            kwargs['eval_known'] = self.eval_known
            datapoints = dataset_module()(dataset_name,**kwargs)
        else:
            datapoints = dataset_module()(dataset_name, **kwargs)

        if add_icl:
            import core.data.icl as icl
            # ICL examples should not be included in the training/dev set
            datapoint_ids_to_avoid = [datapoint.sample_id for datapoint in datapoints[train_start:train_end] + datapoints[dev_start:dev_end]]
            datapoints = icl.add_icl_to_datapoints(datapoints, dataset_name, datapoint_ids_to_avoid, dataset_module)
        tokenized_data = self.tokenize_datapoints(datapoints, train_num_examples + dev_num_examples, **kwargs)
        train_tokenized_data = {k: v[train_start:train_end] for k, v in tokenized_data.items()}
        dev_tokenized_data = {k: v[dev_start:dev_end] for k, v in tokenized_data.items()}

        self.datasets = dict(
            train=train_tokenized_data,
            dev=dev_tokenized_data,
        )
        self.config = dict(
            dataset_name=dataset_name,
            dataset_module_path=dataset_module_path,
            max_seq_length=max_seq_length,
            train_num_examples=train_tokenized_data['input_ids'].size(0),
            dev_num_examples=dev_tokenized_data['input_ids'].size(0),
            use_train_for_dev=use_train_for_dev,
        )

    def get_dataset(self, dataset_split):
        """
        The main interface. Returns a Dataset object.
        """
        return FinetuningDataset(
            tokenizer=self.tokenizer,
            **self.datasets[dataset_split],
        )

    @staticmethod
    def collator(batch):
        """
        The main interface for collating a batch of data.
        """
        return dict(
            input_ids=torch.stack([b['input_ids'] for b in batch]),
            attention_mask=torch.stack([b['attention_mask'] for b in batch]),
            labels=torch.stack([b['labels'] for b in batch]),
            sample_ids=torch.stack([b['sample_ids'] for b in batch]),
        )
    
    def print_config(self):
        train_data_size = tuple(self.datasets['train']['input_ids'].size())
        dev_data_size = tuple(self.datasets['dev']['input_ids'].size())

        print(f'\[train] (dataset_size, sequence_length) = {train_data_size}')
        print(f'\[eval]  (dataset_size, sequence_length) = {dev_data_size}')
        print(f'\[data_config] {self.config}')

    def tokenize_datapoints(self, datapoints, num_examples, **kwargs):
        sample_ids = [datapoint.sample_id for datapoint in datapoints]
        self.id_to_sample.update({datapoint.sample_id: datapoint for datapoint in datapoints})
        
        all_messages = [datapoint.messages for datapoint in datapoints]
        tokenized_data = FinetuningDataManager.encode_with_message_format(
            all_messages,
            self.tokenizer,
            self.max_seq_length,
            num_examples,
            **kwargs,
        )
        tokenized_data['sample_ids'] = torch.tensor(sample_ids).long()
        return tokenized_data

    @staticmethod
    def encode_with_message_format(all_messages, tokenizer, max_seq_length, num_examples=BIG_NUM, **kwargs):
        """
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        """
        input_ids = []
        attention_mask = []
        labels = []
        pbar = tqdm(total=num_examples, desc='Tokenizing')

        for messages in all_messages:
            if len(input_ids) >= num_examples:
                break
            tokenized = FinetuningDataManager.encode_with_message_format_single_example(messages, tokenizer, max_seq_length, **kwargs)
            if tokenized is None:
                continue
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            labels.append(tokenized['labels'])
            pbar.update(1)
        pbar.close()

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    @staticmethod    
    def encode_with_message_format_single_example(messages, tokenizer, max_seq_length, **kwargs):
        """
        Code modified from: https://github.com/allenai/open-instruct/blob/main/open_instruct/finetune.py#L310

        Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
        We concatenate all messages with the roles as delimiters and tokenize them together.
        """
        if len(messages) == 0:
            raise ValueError('messages field is empty.')
        
        def _concat_messages(messages):
            message_text = ""
            for message in messages:
                if message["role"] == "system":
                    message_text += "<|system|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "user":
                    message_text += "<|user|>\n" + message["content"].strip() + "\n"
                elif message["role"] == "assistant":
                    message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                else:
                    raise ValueError(f"Invalid role: {message['role']}")
            return message_text
        example_text = _concat_messages(messages).strip()
        tokenized_example = tokenizer(example_text, return_tensors='pt', truncation=False)
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()

        # NOTE: where the length selection happens
        if input_ids.size(1) > max_seq_length:
            return None

        # mask the non-assistant part for avoiding loss
        for message_idx, message in enumerate(messages):
            if message["role"] != "assistant":
                if message_idx == 0:
                    message_start_idx = 0
                else:
                    message_start_idx = tokenizer(
                        _concat_messages(messages[:message_idx]),
                        return_tensors='pt',
                        max_length=max_seq_length,
                        truncation=True,
                    ).input_ids.shape[1]
                if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                    # here we also ignore the role of the assistant
                    messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
                else:
                    messages_so_far = _concat_messages(messages[:message_idx+1])
                message_end_idx = tokenizer(
                    messages_so_far,
                    return_tensors='pt', 
                    max_length=max_seq_length, 
                    truncation=True
                ).input_ids.shape[1]
                labels[:, message_start_idx:message_end_idx] = -100
                
                if message_end_idx >= max_seq_length:
                    break
    
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids.flatten(),
            'attention_mask': attention_mask.flatten(),
            'labels': labels.flatten(),
        }

    def get_batch_for_inference(
        self,
        dataset_split,
        batch_size,
        system_prompt=None,
        include_labels=False,
        use_chat_format=True,
        chat_format_type=None,
        shuffle=False,
        eval_known=False,
    ):
        #from torch.utils.data import SequentialSampler
        dataset = self.get_dataset(dataset_split)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            #sampler=SequentialSampler(dataset),
            drop_last=False,
            collate_fn=self.collator,
            shuffle=shuffle,
        )
        for batch in dataloader:
            if include_labels:
                yield batch
            else:
                sample_ids = batch['sample_ids'].tolist()
                if eval_known:
                    input_texts = [self.id_to_sample[sample_id].icl_input_text for sample_id in sample_ids]
                else:
                    input_texts = [self.id_to_sample[sample_id].input_text for sample_id in sample_ids]
                if use_chat_format:
                    input_texts = [self.apply_message_format(query, system_prompt) for query in input_texts]
                inputs = self.tokenizer.batch_encode_plus(
                    input_texts,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length,
                )
                inputs['sample_ids'] = batch['sample_ids']
                inputs['input_texts'] = input_texts
                yield inputs

    @staticmethod
    def apply_message_format(query, system_prompt=None):
        if system_prompt is None:
            return f"<|user|>\n{query}\n<|assistant|>\n"
        else:
            return f"<|system|>\n{system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"


def debug_finetuning_data_manager(
    finetuning_data_manager,
    dataset_split,
    batch_size,
    verbose=False,
    skip_special_tokens=True,
    use_chat_format=True,
    include_labels=True,
    shuffle=False,
    parse_func=None,
):
    from rich.markup import escape
    tokenizer = finetuning_data_manager.tokenizer

    batcher = finetuning_data_manager.get_batch_for_inference(
        dataset_split=dataset_split,
        batch_size=batch_size,
        include_labels=include_labels,
        use_chat_format=use_chat_format,
        shuffle=shuffle,
    )

    pred_texts = []
    target_texts = []
    for batch in batcher:
        input_ids = batch['input_ids']
        labels = batch['labels']
        sample_ids = batch['sample_ids'].tolist()
        samples = [finetuning_data_manager.id_to_sample[sample_id] for sample_id in sample_ids]

        labels[labels == -100] = tokenizer.pad_token_id
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=skip_special_tokens)
        target_texts = tokenizer.batch_decode(labels, skip_special_tokens=skip_special_tokens)
        for input_text, target_text, sample in zip(input_texts, target_texts, samples):
            if verbose:
                print(sample)
                print('### input_text ###')
                print(input_text)
                print('-' * 100)
                print('### target_text ###')
                print(target_text)
                if parse_func is not None:
                    parsed_pred_text, parsed_target_text = parse_func(input_text, target_text)
                    print('### parsed_pred_text ###')
                    print(parsed_pred_text)
                    print('### parsed_target_text ###')
                    print(parsed_target_text)
                print('#' * 120)
                input()