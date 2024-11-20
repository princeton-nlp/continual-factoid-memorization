import os
import copy
import json
import yaml
import random
import functools
from pathlib import Path
from typing import TypeVar, List
from string import ascii_uppercase
from dataclasses import dataclass, asdict
from datasets import load_dataset, load_from_disk, concatenate_datasets
from rich import print

import torch
from torch.nn.utils.rnn import pad_sequence

from src.config import DATA_DIR, BIG_NUM


random.seed(0)
T = TypeVar('T')

global_sample_id = 0


@dataclass
class Datapoint:
    sample_id: int = None
    input_text: str = None
    group_name: str = None
    target_text: str = None
    target_for_eval: T = None
    target_candidates: List[str] = None
    messages: List[dict] = None
    icl_input_text: str = None
    info: dict = None
    def to_dict(self):
        return asdict(self)


def load_qa_mem_dataset(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    eval_known = kwargs.get('eval_known', False)
    base_model_name = kwargs.get('base_model_name', None)
    model_type, model_version, model_size = base_model_name.split('-')
    
    data_path = f'{DATA_DIR}/{model_type}{model_version}-{model_size}-filtering/{dataset_name}_yet_to_learn/yet_to_learn_knowledge.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)
    print(f'Loading {dataset_name} from {data_path}')

    with open(data_path, 'r') as f:
        dataset = json.load(f)
        
    datapoints = []
    for sample_id, datapoint in enumerate(dataset):
        # NOTE: This is used for both training and evaluation for the coninuation setting
        input_text = datapoint['input_text'].strip() + '\nThe answer is: '
        target_text = datapoint['target_text'].strip()
        target_for_eval = datapoint['target_for_eval']
        target_candidates = None

        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        icl_input_text = datapoint['messages'][0]['content']
    
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            target_for_eval=target_for_eval,
            input_text=input_text,
            target_text=target_text,
            target_candidates=target_candidates,
            messages=messages,
            icl_input_text=icl_input_text,
        ))
        global_sample_id += 1
    datapoints = datapoints[:2000]

    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    return datapoints


def load_qa_dataset(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')

    base_model_name = kwargs.get('base_model_name', None)
    model_type, model_version, model_size = base_model_name.split('-')

    data_path = f'{DATA_DIR}/{model_type}{model_version}-{model_size}-filtering/{dataset_name}_yet_to_learn/yet_to_learn_knowledge.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    datapoints = []
    for sample_id, datapoint in enumerate(dataset):
        if 'qa_multiple_choice' in task_mode: # multiple choice question
            if 'eval' in task_mode:
                # NOTE: This is used for evaluation
                input_text = datapoint['input_text'].strip()
                original_target_candidates = datapoint['target_candidates']
                target_candidates = copy.deepcopy(original_target_candidates)
                random.shuffle(target_candidates)
                options_text = [f'{opt}. {cand}' for opt, cand in zip(ascii_uppercase, target_candidates)]
                options_text = '\n'.join(options_text)
                input_text = f'{input_text}\nChoices:\n{options_text}\nThe answer is: '
            
                target_index = ascii_uppercase[int(target_candidates.index(datapoint['target_text']))]
                target_text = f'{target_index}. {datapoint["target_text"]}'
                target_for_eval = [target_text, target_index]
            else:
                # NOTE: This is invoked during training
                input_text = datapoint['input_text'].strip()
                target_text = datapoint['target_text']
                target_candidates = datapoint['target_candidates']
                sample_id = datapoint['sample_id']
                target_for_eval = None
        else:
            raise ValueError(f'Invalid task_mode: {task_mode}')        

        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]

        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            target_for_eval=target_for_eval,
            input_text=input_text,
            target_text=target_text,
            target_candidates=target_candidates,
            messages=messages,
        ))
        global_sample_id += 1
    datapoints = datapoints[:2000]

    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    return datapoints


def load_mem_dataset(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    base_model_name = kwargs.get('base_model_name', None)
    model_type, model_version, model_size = base_model_name.split('-')
    eval_known = kwargs.get('eval_known', False)

    if dataset_name.endswith('B'):
        dataset_name = dataset_name.strip('B')
        use_as_B = True
    else:
        use_as_B = False

    if eval_known:
        data_path = f'{DATA_DIR}/{model_type}{model_version}-{model_size}-filtering/{dataset_name}_learned/learned_knowledge.json'
    else:
        data_path = f'{DATA_DIR}/{model_type}{model_version}-{model_size}-filtering/{dataset_name}_yet_to_learn/yet_to_learn_knowledge.json'
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    datapoints = []
    for sample_id, datapoint in enumerate(dataset):
        input_text = datapoint['input_text'].strip()
        target_text = datapoint['target_text'].strip()
        target_for_eval = datapoint['target_for_eval']
        target_candidates = None
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        icl_input_text = datapoint['messages'][0]['content']

        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            target_for_eval=target_for_eval,
            input_text=input_text,
            target_text=target_text,
            target_candidates=target_candidates,
            messages=messages,
            icl_input_text=icl_input_text,
        ))
        global_sample_id += 1
    if use_as_B:
        datapoints = datapoints[2000:4000]
    else:
        datapoints = datapoints[:2000]

    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    return datapoints


def load_instruction_tuning(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    path = Path(DATA_DIR) / dataset_name
    dataset = load_from_disk(path)
    datapoints = []
    for datapoint in dataset['train']:
        instruction = datapoint['instruction']
        input_text = datapoint['input']
        input_text = f'{instruction}\n\n{input_text}'
        input_text = clean_text(dataset_name, input_text)
        target_text = datapoint['output']
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
        global_sample_id += 1

    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints

    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def ultrachat(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')

    dataset = load_from_disk(Path(DATA_DIR) / 'stingning/ultrachat')
    dataset = dataset['train'].select(range(20000))
    datapoints = []
    for datapoint in dataset:
        messages = []
        for turn_idx, content in enumerate(datapoint['data']):
            if turn_idx % 2 == 0:
                message = dict(role='user', content=content)
            else:
                message = dict(role='assistant', content=content)
            messages.append(message)
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
        ))
    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def apps(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    dataset = load_from_disk(Path(DATA_DIR) / 'apps')
    dataset_train = dataset['train']
    dataset_test = dataset['test']
    dataset = concatenate_datasets([dataset_train, dataset_test]).filter(lambda x: x['difficulty'] == 'interview' or x['difficulty'] == 'introductory')
    
    datapoints = []
    for datapoint in dataset:
        starter_code = None if len(datapoint["starter_code"]) == 0 else datapoint["starter_code"]
        try:
            input_outpout = json.loads(datapoint["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
            
        input_text = f"Question:\n {datapoint['question']}"
        if starter_code:
            input_text += f"\n\nSTARTER CODE:\n{starter_code}"
        if fn_name:
            input_text += f"\nUse Call-Based format"  
        else:
            input_text += f"\nUse Standard Input format"
        input_text += "\nAnswer:\n"
        try:
            target_text = json.loads(datapoint["solutions"])[0]
        except ValueError:
            continue

        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
        global_sample_id += 1
   
    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def evolcode(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    dataset = load_from_disk(Path(DATA_DIR) / 'evol-instruct-code-80k-v1')['train']
    
    datapoints = []
    for datapoint in dataset:
        input_text = datapoint['instruction']
        target_text = datapoint['output']
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
        global_sample_id += 1
        
    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def kvr(dataset_name, **kwargs):
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    path = Path(DATA_DIR) / f'{dataset_name}.jsonl'
    dataset = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))

    datapoints = []
    for sample_id, datapoint in enumerate(dataset):
        key = datapoint['key']
        input_text = f'The value of key {key} is?'
        target_text = datapoint['value']
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
    datapoints = datapoints[:2000]
    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    return datapoints


def mquake(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')

    path = Path(DATA_DIR) / 'MQuAKE-CF-3k.json'
    with open(path, 'r') as f:
        dataset = json.load(f)
    
    if dataset_name.endswith('B'):
        dataset = dataset[2000:]
    else:
        dataset = dataset[:2000]

    datapoints = []
    for sample_id, datapoint in enumerate(dataset):
        requested_rewrite = datapoint['requested_rewrite'][0]
        question = requested_rewrite['question']

        if dataset_name.endswith('_true'):
            input_text = f'{question}'
            target_text = requested_rewrite['target_true']['str']
        else:
            input_text = f'Now you are in a counterfactual world. In this world, what is the answer to the question: "{question}"'
            target_text = requested_rewrite['target_new']['str']
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
        global_sample_id += 1

    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    return datapoints


def codes(dataset_name, **kwargs):
    if dataset_name == 'evolcode':
        return evolcode(dataset_name, **kwargs)
    elif dataset_name == 'apps':
        return apps(dataset_name, **kwargs)
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')


def maths(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    if dataset_name == 'gsm8k':
        dataset = load_from_disk(Path(DATA_DIR) / 'gsm8k')['train']
    elif dataset_name == 'math':
        dataset = load_from_disk(Path(DATA_DIR) / 'lighteval/MATH')['train']
    elif dataset_name == 'openmathinstruct1':
        dataset = load_from_disk(Path(DATA_DIR) / 'nvidia/OpenMathInstruct-1')
        dataset = dataset['train'].select(range(20000))
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')
    datapoints = []
    for datapoint in dataset:
        if dataset_name == 'gsm8k':
            input_text = datapoint['question']
            target_text = datapoint['answer']
            target_for_eval = target_text.split('####')[-1].strip()
        elif dataset_name == 'math':
            input_text = datapoint['problem']
            target_text = datapoint['solution']
            target_for_eval = target_text
        elif dataset_name == 'openmathinstruct1':
            input_text = datapoint['question']
            target_text = datapoint['generated_solution']
            target_for_eval = target_text
        else:
            raise ValueError(f'Invalid dataset_name: {dataset_name}')

        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
            target_for_eval=target_for_eval,
        ))
        global_sample_id += 1

    if 'mix' in task_mode:
        ratio = kwargs.get('mix_ratio', 1.0)
        mix_num_datapoints = int(len(datapoints) * ratio)
        mix_datapoints = get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode)
        datapoints += mix_datapoints
    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def get_add_mixing_datapoints(dataset_name, mix_num_datapoints, task_mode):
    mix_datapoints = []

    if 'knowledge-pile' in task_mode:
        mixing_dataset_name = [t for t in task_mode if 'knowledge-pile' in t][0]
        mix_datapoints += load_the_pile(mixing_dataset_name)
    elif 'arxiv-pile' in task_mode:
        mixing_dataset_name = [t for t in task_mode if 'arxiv-pile' in t][0]
        mix_datapoints += load_the_pile(mixing_dataset_name)
    elif 'fineweb' in task_mode:
        mix_datapoints += load_fineweb('fineweb')
    elif 'random-pretraining' in task_mode:
        mix_datapoints += load_random_pretraining(dataset_name)
    else:
        raise ValueError(f'Invalid task_mode: {task_mode}')
    random.shuffle(mix_datapoints)
    mix_datapoints = mix_datapoints[:mix_num_datapoints]
    return mix_datapoints


def load_the_pile(dataset_name, **kwargs):
    """
    This is only used for mixing. Won't be used as A/B/C datasets.
    """
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')
    
    if dataset_name == 'knowledge-pile':
        dataset = load_from_disk(Path(DATA_DIR) / 'Query-of-CC/Knowledge_Pile')
    elif dataset_name == 'arxiv-pile':
        dataset = load_from_disk(Path(DATA_DIR) / 'haritzpuerto/the_pile_arxiv_50k_sample')
    else:
        raise ValueError(f'Invalid dataset_name: {dataset_name}')

    datapoints = []
    dataset = dataset['train'].select(range(20000))
    for datapoint in dataset:
        if dataset_name == 'knowledge-pile':
            text = datapoint['content']
        elif dataset_name == 'arxiv-pile':
            text = datapoint['text']
        
        tokens = text.split()
        if len(tokens) < 100:
            continue

        input_text = ' '.join(tokens[:50])
        input_text = 'Complete the following partial passage:\n\n' + input_text
        target_text = ' '.join(tokens[50:100])
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
        global_sample_id += 1
    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def load_fineweb(dataset_name, **kwargs):
    global global_sample_id
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')

    dataset = load_from_disk(Path(DATA_DIR) / 'kh4dien/fineweb-100m-sample')
    datapoints = []
    for datapoint in dataset['train'].select(range(20000)):
        text = datapoint['text']
        tokens = text.split()
        if len(tokens) < 100:
            continue

        input_text = ' '.join(tokens[:50])
        input_text = 'Complete the following partial passage:\n\n' + input_text
        target_text = ' '.join(tokens[50:100])
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=global_sample_id,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
        global_sample_id += 1
    print(f'Total datapoints available for {dataset_name}: {len(datapoints)}')
    return datapoints


def load_random_pretraining(dataset_name, **kwargs):
    task_mode = kwargs.get('task_mode', 'none')
    task_mode = task_mode.split('+')

    if len(task_mode) > 1:
        num_words = task_mode[1]
        path = Path(DATA_DIR) / f'random_pretraining_{num_words}.jsonl'
    else:
        path = Path(DATA_DIR) / 'random_pretraining.jsonl'

    with open(path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    datapoints = []
    for datapoint in dataset:
        datapoint_idx = datapoint['datapoint_idx']
        input_text = datapoint['input_text']
        target_text = datapoint['target_text']
        messages = [
            dict(role='user', content=input_text),
            dict(role='assistant', content=target_text),
        ]
        datapoints.append(Datapoint(
            sample_id=datapoint_idx,
            group_name=dataset_name,
            messages=messages,
            input_text=input_text,
            target_text=target_text,
        ))
    return datapoints


prefix_datapoints = None
def random_prefix_collator(batch, finetuning_data_manager):
    global prefix_datapoints
    if prefix_datapoints is None:
        prefix_dataset = load_from_disk(Path(DATA_DIR) / 'Query-of-CC/Knowledge_Pile')
        prefix_datapoints = [d for d in prefix_dataset['train'].select(range(10000))]

    tokenizer = finetuning_data_manager.tokenizer
    max_seq_length = finetuning_data_manager.max_seq_length
    input_ids = []
    attention_mask = []
    labels = []
    sample_ids = [datapoint['sample_ids'].item() for datapoint in batch]
    datapoints = [finetuning_data_manager.id_to_sample[sample_id] for sample_id in sample_ids]
    for datapoint, tokenized_datapoint in zip(datapoints, batch):
        prefix_datapoint = random.choice(prefix_datapoints)
        prefix_text_length = random.randint(0, 50)
        if prefix_text_length == 0:
            prefix_text = ''
        else:
            prefix_text = ' '.join(prefix_datapoint['content'].split()[0:prefix_text_length])
            prefix_text = f'Below is some unrelated context:\n{prefix_text}\n\n'
        
        input_text = f'{prefix_text}Below is the important key/value information:\nWhen the key is {datapoint.info["key"]}, what is the value?'
        target_text = datapoint.target_text
        messages = [{'role': 'user', 'content': input_text}, {'role': 'assistant', 'content': target_text}]
        tokenized = finetuning_data_manager.encode_with_message_format_single_example(messages, tokenizer, max_seq_length)
        input_ids.append(tokenized['input_ids'])
        attention_mask.append(tokenized['attention_mask'])
        labels.append(tokenized['labels'])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    sample_ids = torch.tensor(sample_ids)
            
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sample_ids=sample_ids,
    )


def construct_random_option(
    input_text,
    target_text,
    target_candidates,
):
    target_candidates = copy.deepcopy(target_candidates)
    random.shuffle(target_candidates)
    options_text = [f'{opt}. {cand}' for opt, cand in zip(ascii_uppercase, target_candidates)]
    options_text = '\n'.join(options_text)
    input_text = f'{input_text}\nChoices:\n{options_text}\nThe answer is: '
    target_index = ascii_uppercase[int(target_candidates.index(target_text))]
    target_text = f'{target_index}. {target_text}'
    target_for_eval = [target_text, target_index]
    return input_text, target_text, target_for_eval


def shuffling_qa_option_collator(batch, finetuning_data_manager):
    tokenizer = finetuning_data_manager.tokenizer
    max_seq_length = finetuning_data_manager.max_seq_length
    input_ids = []
    attention_mask = []
    labels = []
    sample_ids = [datapoint['sample_ids'].item() for datapoint in batch]
    datapoints = [finetuning_data_manager.id_to_sample[sample_id] for sample_id in sample_ids]

    for datapoint, tokenized_datapoint in zip(datapoints, batch):
        if datapoint.group_name in QA_DATASETS:
            input_text = datapoint.input_text.strip()
            original_target_candidates = datapoint.target_candidates
            target_candidates = copy.deepcopy(original_target_candidates)
            random.shuffle(target_candidates)
            options_text = [f'{opt}. {cand}' for opt, cand in zip(ascii_uppercase, target_candidates)]
            options_text = '\n'.join(options_text)
            input_text = f'{input_text}\nChoices:\n{options_text}\nThe answer is: '
            target_index = ascii_uppercase[int(target_candidates.index(datapoint.target_text))]
            target_text = f'{target_index}. {datapoint.target_text}'
            target_for_eval = [target_text, target_index]
            messages = [{'role': 'user', 'content': input_text}, {'role': 'assistant', 'content': target_text}]
            tokenized = finetuning_data_manager.encode_with_message_format_single_example(messages, tokenizer, max_seq_length)
            input_ids.append(tokenized['input_ids'])
            attention_mask.append(tokenized['attention_mask'])
            labels.append(tokenized['labels'])
        else:
            input_ids.append(tokenized_datapoint['input_ids'])
            attention_mask.append(tokenized_datapoint['attention_mask'])
            labels.append(tokenized_datapoint['labels'])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels = pad_sequence(labels, batch_first=True, padding_value=tokenizer.pad_token_id)
    sample_ids = torch.tensor(sample_ids)
            
    return dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        sample_ids=sample_ids,
    )


def get_collator(task_mode, finetuning_data_manager):
    if task_mode.startswith('qa_multiple_choice'):
        collator = functools.partial(shuffling_qa_option_collator, finetuning_data_manager=finetuning_data_manager)
    elif task_mode == 'random_prefix':
        collator = functools.partial(random_prefix_collator, finetuning_data_manager=finetuning_data_manager)
    else:
        collator = finetuning_data_manager.collator
    return collator


def clean_text(dataset_name, text):
    if dataset_name == 'Yukang/LongAlpaca-12k':
        return text.strip().strip('None').strip()
    else:
        return text.strip()


class DatasetMapping:
    def __init__(self):
        self.dataset_name_to_func = {
            dataset_name: func 
            for func, dataset_names in FUNC_TO_DATASET_NAME.items() 
            for dataset_name in dataset_names
        }

    def __call__(self, dataset_name, **kwargs):
        return self.dataset_name_to_func[dataset_name](dataset_name, **kwargs)


KVR_DATASETS = [
    'kvr',
    'kvrB',
]

THE_PILE_DATASETS = [
    'knowledge-pile',
    'arxiv-pile',
]

QA_DATASETS = [
    'commonsenseqa',
    'popqa',
    'piqa',
    'medqa',
    'medqaB',
    'legalqa',
    'popqa-mc',
]

QA_MEM_DATASETS = [
    'popqa',
    'triviaqa',
]

MEM_DATASETS = [
    'entityqa',
    'webqa',
    'webqaB',
    'lama',
]

MATHS = [
    'gsm8k',
    'math',
]

CODES = [
    'evolcode',
    'apps',
]


FUNC_TO_DATASET_NAME = {
    load_qa_dataset: QA_DATASETS,
    load_qa_mem_dataset: QA_MEM_DATASETS,
    load_the_pile: THE_PILE_DATASETS,
    load_fineweb: ['fineweb'],
    kvr: KVR_DATASETS,
    ultrachat: ['ultrachat'],
    load_mem_dataset: MEM_DATASETS,
    maths: MATHS,
    codes: CODES,
}
