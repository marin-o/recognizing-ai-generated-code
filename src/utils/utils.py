def tokenize_fn(tokenizer, examples):
    tokenized = tokenizer(
        examples['code'], 
        truncation=True, 
        padding=True, 
        max_length=512,
        return_tensors=None  # Don't return tensors yet
    )
    # Preserve the target column
    tokenized['target'] = examples['target']
    return tokenized