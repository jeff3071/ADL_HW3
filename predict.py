from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from tqdm import tqdm
import jsonlines, json
import torch
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset, DataLoader


class textDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
    ):
        self.data = data
        self.maintext, self.ids = self.collate_fn(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        return self.maintext[index], self.ids[index]

    def collate_fn(self, samples: List[Dict]) -> Dict:
        maintext = [data['maintext'] for data in samples ]
        ids = [data['id'] for data in samples]
        return maintext, ids

def main(args):
    with open(args.test_file) as f:
        testset = [json.loads(line) for line in f]

    model_path = "./model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = args.device
    model = model.to(device)
    batch_size = args.batch_size
    preds = []

    test_dataset = textDataset(testset)
    test_dataloader = DataLoader(test_dataset, batch_size, False, num_workers=2, pin_memory=True)

    with torch.no_grad():
        for (text, ids) in tqdm(test_dataloader):
            inputs = tokenizer(text, return_tensors='pt', max_length=args.max_input_length, truncation=True, padding=True)
            output_ = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=args.max_target_length,
                num_beams=args.num_beams,
                top_k = args.top_k,
                top_p = args.top_p,
                do_sample = args.do_sample,
                temperature = args.temperature 
            )
            outputs = [tokenizer.decode(x, skip_special_tokens=True) for x in output_]
            preds += [{'title': outputs[i], 'id': ids[i]} for i in range(len(outputs))]

            with jsonlines.open(args.pred_file, mode='w') as writer:
                writer.write_all(preds)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="predictions.jsonl")

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--do_sample", type=bool, default=None)


    parser.add_argument("--max_input_length", type=int, default=256)
    parser.add_argument("--max_target_length", type=int, default=64)


    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:0"
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    main(args)
    