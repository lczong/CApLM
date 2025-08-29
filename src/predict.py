import argparse
from caplm import CApLMPredictor
from utils import log_gpu_count

def main():
    parser = argparse.ArgumentParser(
        description='CApLM: Predict CAZymes and CAZyme classes from protein sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--binary-model', help='Path to binary classification model')
    parser.add_argument('--multi-model', help='Path to multi-label classification model')
    parser.add_argument('--input', required=True, help='Path to input FASTA file')
    
    parser.add_argument('--binary-threshold', type=float, default=0.5,
                        help='Threshold for binary classification')
    
    parser.add_argument('--multi-threshold', type=float, default=0.5,
                        help='Global threshold for multi-label classification')
    parser.add_argument('--multi-thresholds', type=float, nargs='*',
                        help='Per-class thresholds (6 values)')
    parser.add_argument('--multi-thresholds-file',
                        help='JSON file with per-class thresholds')
    
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for both models')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                        help='Device (auto-detect if not specified)')
    parser.add_argument('--mixed-precision', choices=['bf16', 'fp16', 'fp32'], default='fp32',
                        help='Mixed precision')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of dataloader workers')
    
    parser.add_argument('--output-dir', default='./outputs',
                        help='Output directory')
    parser.add_argument('--output-name', default='test',
                        help='Prefix for output files')
    parser.add_argument('--save-embeddings', action='store_true', default=False,
                        help='Save embeddings')
    
    args = parser.parse_args()
    
    log_gpu_count()
    
    predictor = CApLMPredictor(
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    predictor.predict(
        test_fasta=args.input,
        binary_model_path=args.binary_model,
        multi_model_path=args.multi_model,
        binary_threshold=args.binary_threshold,
        multi_thresholds=args.multi_thresholds,
        multi_thresholds_file=args.multi_thresholds_file,
        multi_global_threshold=args.multi_threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
        output_dir=args.output_dir,
        output_name=args.output_name,
        save_embeddings=args.save_embeddings,
        dataloader_workers=args.num_workers
    )


if __name__ == "__main__":
    main()