"""
python -m src.run_eval --dataset_name kvr --dataset_name_B none --model_name ${model_name} --verbose
"""
import re
import os
import gc
import yaml
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm
from rich import print
import torch
from torch.utils.data import DataLoader, SequentialSampler
import transformers
from transformers import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer

from src.core.data.sft import FinetuningDataManager
from src.config import RUN_BASE_DIR

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--run_base_dir', type=str, default=RUN_BASE_DIR)
parser.add_argument('--max_seq_length', type=int, default=8000)
parser.add_argument('--train_num_examples', type=int, default=0)
parser.add_argument('--dev_num_examples', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--stopping_condition', type=str, default='sc=loss')

parser.add_argument('--base_model_name', type=str, default='llama-3-8b')
parser.add_argument('--dataset_name', type=str, default=None)
parser.add_argument('--dataset_name_B', type=str, default='none')
parser.add_argument('--add_icl', action='store_true')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--task_mode', type=str, default='none', choices=['none', 'qa_multiple_choice+eval'])
parser.add_argument('--mix_ratio', type=float, default=-1.0)
parser.add_argument('--mixd', type=str, default='none')
parser.add_argument('--bsz', type=str, default=-1)
parser.add_argument('--mix_ratio_B', type=float, default=-1.0)
parser.add_argument('--mixd_B', type=str, default='none')
parser.add_argument('--bsz_B', type=str, default=-1)
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--eval_known', action='store_true')
parser.add_argument('--save_results', action='store_true')
parser.add_argument('--base_format', action='store_true')
args = parser.parse_args()

generation_config = dict(
    do_sample=False,
    max_new_tokens=1000,
    num_return_sequences=1,
    temperature=1.0,
    top_p=1.0,
)


def parse_qa_pred_and_gold(pred_text, gold_text):
    pattern = re.compile(r'^\s*([A-Za-z])\.\s*(.*)')
    pred_text = pred_text.strip().lower()
    gold_text = gold_text.strip().lower()
    pred_match = pattern.match(pred_text)
    gold_match = pattern.match(gold_text)
    pred_text = pred_match.group(1) if pred_match else None
    gold_text = gold_match.group(1) if gold_match else None
    return pred_text, gold_text


def parse_con_pred_and_gold(pred_text, gold_text):
    pred_text = pred_text.strip().lower()
    gold_text = gold_text.strip().lower()
    return pred_text, gold_text


def parse_gsm8k_pred_and_gold(pred_text, gold_text):
    pred_text = re.findall(r'\d+', pred_text)[-1]
    gold_text = re.findall(r'\d+', gold_text)[-1]
    return pred_text, gold_text


parse_func_mapping = {
    'none': parse_con_pred_and_gold,
    'qa_multiple_choice+eval': parse_qa_pred_and_gold,
}

comparison_func = {
    'webqa': lambda pred_text, gold_text: pred_text in gold_text,
    
}

if __name__ == '__main__':
    if args.base_model_name is None:
        base_model_names = ['llama-2-7b', 'llama-2-13b', 'llama-3-8b']
    else:
        base_model_names = [args.base_model_name]
    
    
    if args.seed == -1:
        seeds = [0, 1, 2]
    else:
        seeds = [args.seed]
    
    if args.mix_ratio == -1.0:
        mix_ratios = [1.0, 2.0, 4.0]
    else:
        mix_ratios = [args.mix_ratio]
    
    if args.bsz == -1:
        bszs = [32, 128, 512]
    else:
        bszs = [args.bsz]
    
    
    if args.model_name is not None:
        seeds = [0]
        mix_ratios = [None]
        bszs = [None]
        model_name = args.model_name
    
    if args.eval_known: # eval known
        generation_config['max_new_tokens'] = 20
        
    for base_model_name in base_model_names:
        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            transformers.set_seed(seed)
        
            dataset_name = args.dataset_name
    
            for mix_ratio in mix_ratios:
                for bsz in bszs:
    
                    if args.model_name is None:
                        model_name = run_base_dir / f'{base_model_name}/{dataset_name}_B={args.dataset_name_B}_seed={seed}_mixd={args.mixd}_mixr={mix_ratio}_bsz={bsz}_mixdB={args.mixd_B}_mixrB={args.mix_ratio_B}_bszB={args.bsz_B}/final'
                    else:
                        model_name = args.model_name
                    
                    # skip if model does not exist
                    if not Path(model_name).exists():
                        print(f'Model does not exist: {model_name}')
                        print('#' * 100)
                        continue
            
                    save_path = Path(model_name) / f'eval_metrics.json'
                    save_path.parent.mkdir(exist_ok=True, parents=True)
                    predictions_path = Path(model_name) / f'{dataset_name}_predictions.jsonl'
                
                    print(f'Evaluating model: {model_name}')
                    print(f'Using dataset: {dataset_name}')
                    if args.verbose:
                        print(vars(args))
                        print(generation_config)
    
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=True)
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                    tokenizer.padding_side = 'left'
                    
                    finetuning_data_manager = FinetuningDataManager(
                        dataset_name=dataset_name,
                        dataset_module_path='src.datasets',
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length,
                        train_num_examples=args.train_num_examples,
                        dev_num_examples=args.dev_num_examples,
                        use_train_for_dev=True,
                        add_icl=args.add_icl,
                        base_model_name=base_model_name,
                        task_mode=args.task_mode,
                        eval_known=args.eval_known,
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', token=os.environ['HF_TOKEN'])
                    
                    corrects = []
                    batcher = finetuning_data_manager.get_batch_for_inference(
                        dataset_split='dev',
                        batch_size=args.batch_size,
                        include_labels=False,
                        use_chat_format=False if args.base_format else True,
                        eval_known=args.eval_known,
                    )
                    
                    predictions = []
                    corrects = []
                    rouges = []
                    losses = []
                    familiar_corrects = []
                    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
                    
                    num_datapoints = finetuning_data_manager.config['dev_num_examples']
                    num_batches = num_datapoints // args.batch_size + int(num_datapoints % args.batch_size > 0)
                    with torch.inference_mode():
                        for batch in tqdm(batcher, total=num_batches):
                            input_texts = batch['input_texts']
                            sample_ids = batch['sample_ids'].tolist()
                            inputs = {k: v.to(model.device) for k, v in batch.items() if k in {'input_ids', 'attention_mask', 'labels'}}
                    
                            samples = [finetuning_data_manager.id_to_sample[sample_id] for sample_id in sample_ids]
                    
                            out = model.generate(**inputs, **generation_config, pad_token_id=tokenizer.eos_token_id)
                            generated_texts = tokenizer.batch_decode(out, skip_special_tokens=True)
                            pred_texts = [generated_text[len(input_text):] for generated_text, input_text in zip(generated_texts, input_texts)]
                            gold_texts = [sample.target_text for sample in samples]
    
                            parse_func = parse_func_mapping[args.task_mode]
            
                            for input_text, pred_text, gold_text, sample in zip(input_texts, pred_texts, gold_texts, samples):
                                _pred_text, _gold_text = parse_func(pred_text, gold_text)
                                assert _gold_text is not None
                                if _pred_text is None:
                                    correct = False
                                    familiar_correct = False
                                    rouge = 0.0
                                else:
                                    if args.dataset_name in comparison_func:
                                        correct = comparison_func[args.dataset_name](_pred_text, _gold_text)
                                    else:
                                        correct = _pred_text == _gold_text
                                    rouge = scorer.score(pred_text, gold_text)['rougeL'].fmeasure
                                        
                                corrects.append(correct)
                                rouges.append(rouge)
                                predictions.append(dict(
                                    sample_id=sample.sample_id,
                                    input_text=input_text,
                                    pred_text=pred_text,
                                    gold_text=gold_text,
                                    parsed_pred_text=_pred_text,
                                    parsed_gold_text=_gold_text,
                                    correct=correct,
                                    rouge=rouge,
                                ))
    
                                if args.verbose:
                                    print(f'sample_id: {sample.sample_id}')
                                    print(f'task_mode: {args.task_mode}')
                                    print(f'### input_text ###\n{input_text}')
                                    print('---')
                                    print(f'### pred ###')
                                    print(f'original: {pred_text}')
                                    print(f'parsed: {_pred_text}')
                                    print('---')
                                    print(f'### gold ###')
                                    print(f'original: {gold_text}')
                                    print(f'parsed: {_gold_text}')
                                    print('---')
                                    print(f'Correct: {correct} | accumulated: {sum(corrects) / len(corrects):.4f}')
                                    print(f'ROUGE-L: {rouge} | accumulated: {sum(rouges) / len(rouges):.4f}')
                                    print('-' * 100)
                    
                    avg_acc = sum(corrects) / len(corrects)
                    avg_rouge = sum(rouges) / len(rouges)
                    print(f'Accuracy: {avg_acc:.4f}')
                    print(f'ROUGE_L: {avg_rouge:.4f}')
                    eval_metrics = dict(
                        Accuracy=avg_acc,
                        ROUGE_L=avg_rouge,
                    )
                    if args.base_format:
                        dataset_name = f'{dataset_name}base'
                    if args.save_results:
                        if os.path.exists(save_path):
                            with open(save_path, 'r') as f:
                                save_data = json.load(f)
                            save_data[dataset_name] = eval_metrics
                        else:
                            save_data = {dataset_name: eval_metrics}
        
                        with open(save_path, 'w') as f:
                            json.dump(save_data, f, indent=4)
        
                        print(f'Saved to: {save_path}')
                        print('#' * 100)

                        with open(predictions_path, 'w') as f:
                            for prediction in predictions:
                                f.write(json.dumps(prediction) + '\n')
            
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
    