import json
import random
import argparse
import torch
from tqdm import tqdm
from collections import Counter
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors


def parse_args():
    parser = argparse.ArgumentParser(description="Match vocabulary between two tokenizers.")
    parser.add_argument('--tokenizer_source', type=str, required=True)
    parser.add_argument('--tokenizer_target', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True, help="Path to the model file.")
    return parser.parse_args()


def load_tokenizers(args):
    """Load and prepare tokenizer configurations from JSON files."""
    # Load tokenizer JSON files
    with open(args.tokenizer_target, 'r') as f:
        target_tokenizer_json = json.load(f)

    with open(args.tokenizer_source, 'r') as f:
        source_tokenizer_json = json.load(f)

    # Extract vocabularies
    source_vocabulary = source_tokenizer_json["model"]["vocab"]
    target_vocabulary = target_tokenizer_json["model"]["vocab"]

    # Create id-to-token mappings
    source_id_to_token = {v: k for k, v in source_vocabulary.items()}
    target_id_to_token = {v: k for k, v in target_vocabulary.items()}

    # Initialize tokenizers
    target_tokenizer = Tokenizer.from_file(args.tokenizer_target)
    source_tokenizer = Tokenizer.from_file(args.tokenizer_source)

    # Configure mistral tokenizer
    source_tokenizer.normalizer = normalizers.Sequence([])
    source_tokenizer.post_processor = processors.Sequence([])
    source_tokenizer.pre_tokenizer = pre_tokenizers.Sequence([])

    return (
        source_vocabulary,
        target_vocabulary,
        source_id_to_token,
        target_id_to_token,
        source_tokenizer,
        target_tokenizer
    )

def create_match_dictionary(target_vocabulary, mistral_tokenizer):
    """Create a mapping dictionary between target and source tokens."""
    match_dict = {}

    for subword, subword_id in tqdm(target_vocabulary.items()):
        # Handle special tokens
        if subword_id == 3:
            match_dict[subword_id] = [10]
            continue

        if subword_id < 5:
            match_dict[subword_id] = [subword_id]
            continue

        if subword_id < 16:
            match_dict[subword_id] = [14 + subword_id - 5]
            continue

        # Process regular tokens
        mistral_tokens = mistral_tokenizer.encode(subword).ids
        match_dict[subword_id] = mistral_tokens

    return match_dict

def analyze_matches(match_dict):
    """Analyze the distribution of token matches."""
    counter = Counter([0 if v is None else len(v) for v in match_dict.values()])
    print("Match length distribution:", counter.most_common())

def show_example_matches(target_id_to_token, mistral_id_to_token, match_dict, num_examples=10):
    """Display random examples of token matches."""
    print("\nRandom examples of matched subwords:")
    for _ in range(num_examples):
        i = random.randint(0, len(target_id_to_token) - 1)
        print(f"{target_id_to_token[i]} -> {[mistral_id_to_token[j] for j in match_dict[i]]}")

def process_embeddings(match_dict, target_vocabulary, model_path):
    """Process and update model embeddings based on token matches."""
    with open(f"{model_path}/model.safetensors.index.json", "r") as f:
        hf_weight_map = json.load(f)["weight_map"]

    # Load model tensors
    with safe_open(f"{model_path}/{hf_weight_map['model.embed_tokens.weight']}", framework="pt", device="cpu") as f:
        embedding_weights = {key: f.get_tensor(key) for key in f.keys()}
    if hf_weight_map['model.embed_tokens.weight'] != hf_weight_map['lm_head.weight']:
        with safe_open(f"{model_path}/{hf_weight_map['lm_head.weight']}", framework="pt", device="cpu") as f:
            lm_head_weights = {key: f.get_tensor(key) for key in f.keys()}
    else:
        lm_head_weights = embedding_weights

    embedding = embedding_weights["model.embed_tokens.weight"]
    lm_head = lm_head_weights["lm_head.weight"]
    dtype = embedding.dtype

    # Update embeddings
    for target_id, source_ids in tqdm(match_dict.items()):
        embedding[target_id] = torch.mean(embedding[source_ids].float(), dim=0).to(dtype)
        lm_head[target_id] = torch.mean(lm_head[source_ids].float(), dim=0).to(dtype)

    # Trim embedding to vocabulary size
    embedding = embedding[:len(target_vocabulary)].contiguous()
    lm_head = lm_head[:len(target_vocabulary)].contiguous()

    embedding_weights["model.embed_tokens.weight"] = embedding
    lm_head_weights["lm_head.weight"] = lm_head

    # Save updated tensors
    save_file(embedding_weights, f"{model_path}/{hf_weight_map['model.embed_tokens.weight']}")
    if hf_weight_map['model.embed_tokens.weight'] != hf_weight_map['lm_head.weight']:
        save_file(lm_head_weights, f"{model_path}/{hf_weight_map['lm_head.weight']}")

def main():
    """Main execution function."""
    args = parse_args()
    # Load and prepare tokenizers
    (
        source_vocabulary,
        target_vocabulary,
        source_id_to_token,
        target_id_to_token,
        source_tokenizer,
        target_tokenizer
    ) = load_tokenizers(args)

    # Create token matching dictionary
    match_dictionary = create_match_dictionary(target_vocabulary, source_tokenizer)

    # Analyze and display results
    analyze_matches(match_dictionary)
    show_example_matches(target_id_to_token, source_id_to_token, match_dictionary)

    # Process embeddings
    process_embeddings(match_dictionary, target_vocabulary, args.model_path)

if __name__ == "__main__":
    main()