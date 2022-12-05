## ADL HW3

利用mT5決定一段文章的標題

### Training

使用huggingface的script

```bash
python run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --evaluation_strategy epoch \
    --gradient_accumulation_steps 8 \
    --train_file ./data/train.json \
    --validation_file ./data/public.json \
    --text_column maintext \
    --learning_rate 1e-3 \
    --num_train_epochs 5 \
    --summary_column title \
    --dataset_config "3.0.0" \
    --source_prefix "summarize: " \
    --output_dir ./model \
    --per_device_train_batch_size=2 \
    --save_strategy epoch \
    --per_device_eval_batch_size=2 \
    --overwrite_output_dir \
    --optim adafactor \
    --save_total_limit 1 \
    --predict_with_generate
```

### Testing

```bash
python predict.py\
  --test_file </path/to/test_file>\
  --pred_file </path/to/pred_file>\
  [--num_beams ]\
  [--top_k ]\
  [--top_p ]\
  [--temperature]\
  [--do_sample ]
```


### result

- rouge-1: 25.196 
- rouge-2: 10.045 
- rouge-3: 22.792

#### Reproduce my result

```bash
bash ./run.sh </path/to/input.jsonl> </path/to/output.jsonl>
```

