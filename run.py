from data_utils import MyDataset
from api_utils import api_handler
from string import punctuation
import argparse
import os
import tqdm
import json
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='chatgpt')
    parser.add_argument('--dataset_name', default='MedQA')
    parser.add_argument('--dataset_dir', default='./datasets/MedQA/')
    parser.add_argument('--start_pos', type=int, default=21)
    parser.add_argument('--end_pos', type=int, default=50)
    parser.add_argument('--output_files_folder', default='./outputs/MedQA')

    parser.add_argument('--method', type=str, default='syn_verif', choices=['syn_verif', 'syn_only', 'anal_only', 'base_direct', 'base_cot'])
    parser.add_argument('--max_attempt_vote', type=int, default=3)
    parser.add_argument('--legacy_voting', action='store_true', default=False,
                        help='Restore original vote parsing: substring match (no word '
                             'boundaries) and unparseable output coerced to "yes".')
    args = parser.parse_args()

    print(args)

    ### get handler
    # `--model_name` is passed straight through to the API as the model id
    # (e.g. a Groq catalogue id such as `llama-3.3-70b-versatile`). Legacy
    # aliases ('chatgpt', 'gpt4', ...) are still mapped inside api_handler.
    handler = api_handler(args.model_name)

    ### get dataobj
    dataobj = MyDataset('test', args, traindata_obj=None)

    ### set test range
    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)  # closed interval

    ### set output_file_name (ensure the folder exists; the code only appends)
    os.makedirs(args.output_files_folder, exist_ok=True)
    safe_model = args.model_name.replace('/', '_')
    exact_output_file = f"{args.output_files_folder}/{safe_model}-{args.method}"
    #print(exact_output_file)


    input_prompt = {}
    for idx in tqdm.tqdm(test_range, desc=f"{args.start_pos} ~ {end_pos}"):
        raw_sample = dataobj.get_by_idx(idx)
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
        
        realqid = idx
        if args.dataset_name in ['MedQA', 'MedMCQA'] or 'MMLU' in args.dataset_name:
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']
            data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
        elif args.dataset_name == 'PubMedQA':
            question = raw_sample['context'] + ' ' + question
            options = raw_sample['options']
            gold_answer = raw_sample['answer_idx']
            data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)
        elif args.dataset_name in ['MedicationQA']:
            options = ''
            gold_answer = raw_sample['answer_idx']
            data_info = fully_decode(idx, realqid, question, options, gold_answer, handler, args, dataobj)

        record = json.dumps(data_info)
        with open(exact_output_file, 'a') as f:
            f.write(record + '\n')

    ### per-run voting summary (headline metric for the consensus stage)
    if args.method == 'syn_verif':
        print(vote_summary())
