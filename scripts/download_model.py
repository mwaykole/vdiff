#!/usr/bin/env python
"""Download model script for dfastllm Serving.

Downloads models from HuggingFace Hub for offline use.

Usage:
    python scripts/download_model.py GSAI-ML/LLaDA-8B-Instruct
    python scripts/download_model.py GSAI-ML/LLaDA-8B-Instruct --output ./models
"""

import argparse
import os
import sys


def download_model(
    model_name: str,
    output_dir: str = None,
    revision: str = None,
    trust_remote_code: bool = True,
):
    """Download a model from HuggingFace Hub.
    
    Args:
        model_name: Name of the model on HuggingFace Hub.
        output_dir: Directory to save the model.
        revision: Specific revision to download.
        trust_remote_code: Whether to trust remote code.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        print("Install with: pip install huggingface-hub")
        sys.exit(1)
    
    print(f"Downloading model: {model_name}")
    
    if output_dir:
        cache_dir = output_dir
        print(f"Output directory: {output_dir}")
    else:
        cache_dir = None
        print("Using default HuggingFace cache directory")
    
    if revision:
        print(f"Revision: {revision}")
    
    try:
        path = snapshot_download(
            repo_id=model_name,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=os.path.join(output_dir, model_name.replace("/", "--")) if output_dir else None,
        )
        print(f"\nModel downloaded successfully!")
        print(f"Location: {path}")
        
        # Also download tokenizer if separate
        print("\nDownloading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
        )
        print("Tokenizer downloaded!")
        
        return path
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download models for dfastllm Serving"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name on HuggingFace Hub (e.g., GSAI-ML/LLaDA-8B-Instruct)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for the model",
    )
    parser.add_argument(
        "--revision",
        "-r",
        type=str,
        default=None,
        help="Specific revision to download",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code (default: True)",
    )
    
    args = parser.parse_args()
    
    download_model(
        model_name=args.model_name,
        output_dir=args.output,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
