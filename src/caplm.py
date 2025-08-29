import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from tqdm import tqdm
from Bio import SeqIO


class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences: List[str], ids: List[str], 
                 tokenizer, max_length: int = 1024):
        self.sequences = sequences
        self.ids = ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        encoding = self.tokenizer(
            sequence,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None
        )
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
        }


class CApLMPredictor:
    """
    Two-stage protein sequence predictor combining binary and multi-label classification.
    Stage 1: Binary classification (cazy vs non-cazy)
    Stage 2: Multi-label classification for positive samples
    """
    
    def __init__(self, device: str = None, mixed_precision: str = 'bf16'):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if mixed_precision == 'bf16' else torch.float16 if mixed_precision == 'fp16' else torch.float32
        self.mixed_precision = mixed_precision
        print(f"Device: {self.device}, Mixed Precision: {mixed_precision}")
        
        self.binary_model = None
        self.multi_model = None
        self.tokenizer = None
        self.data_collator = None
        
        self.all_sequences = {}
        self.all_ids = []
        
        self.stage1_results = None
        self.stage2_results = None
        self.positive_ids = set()
        
        self.binary_label_to_id = {'non-cazy': 0, 'cazy': 1}
        self.binary_id_to_label = {0: 'non-cazy', 1: 'cazy'}
        
        self.multi_classes = ["GT", "GH", "CBM", "CE", "PL", "AA"]
        
    def load_binary_model(self, model_path: str):
        if model_path:
            print(f"🔬 Loading Binary Classification Model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Binary model path not found: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.binary_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=self.dtype
            )
        else:
            print("No local binary model path provided, will download from HuggingFace")
            self.tokenizer = AutoTokenizer.from_pretrained("lczong/CApLM", subfolder="binary")
            self.binary_model = AutoModelForSequenceClassification.from_pretrained(
                "lczong/CApLM",
                subfolder="binary",
                torch_dtype=self.dtype
            )
        self.binary_model.to(self.device)
        self.binary_model.eval()
        self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        
        print(f"   Binary model loaded successfully")
        
    def load_multi_model(self, model_path: str):
        if model_path:
            print(f"🔬 Loading Multi-label Classification Model from {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Multi-label model path not found: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.multi_model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                torch_dtype=self.dtype
            )
        else:
            print("No local multi-label model path provided, will download from HuggingFace")
            self.tokenizer = AutoTokenizer.from_pretrained("lczong/CApLM", subfolder="multi-label")
            self.multi_model = AutoModelForSequenceClassification.from_pretrained(
                "lczong/CApLM",
                subfolder="multi-label",
                torch_dtype=self.dtype
            )
        self.multi_model.to(self.device)
        self.multi_model.eval()
        if self.data_collator is None:
            self.data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        
        print(f"   Multi-label model loaded successfully")
        
    def load_sequences_from_fasta(self, fasta_file: str) -> Tuple[List[str], List[str]]:
        sequences = []
        ids = []
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq_id = record.id
            sequence = str(record.seq).replace('.', '')
            
            sequences.append(sequence)
            ids.append(seq_id)
            self.all_sequences[seq_id] = sequence
            
        self.all_ids = ids
        
        return sequences, ids
    
    @torch.no_grad()
    def inference(
        self,
        dataset: Dataset,
        model: torch.nn.Module,
        data_collator,
        batch_size: int = 8,
        save_embeddings: bool = False,
        is_multi_label: bool = False,
        dataloader_workers: int = 4,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=dataloader_workers,
            collate_fn=data_collator,
            pin_memory=(self.device == "cuda"),
        )
        
        model.eval()

        all_probs = []
        all_embs = [] if save_embeddings else None
        
        if self.mixed_precision == 'bf16' and self.device == "cuda" and torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        elif self.mixed_precision == 'fp16' and self.device == "cuda" and torch.cuda.is_fp16_supported():
            autocast_dtype = torch.float16
        else:
            autocast_dtype = None
        
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            
            ctx = torch.amp.autocast(device_type=self.device, dtype=autocast_dtype) if autocast_dtype else torch.amp.autocast(device_type=self.device, enabled=False)
            with ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=save_embeddings,
                    return_dict=True,
                )
                logits = outputs.logits
                
                if is_multi_label:
                    probs = logits.float().sigmoid()
                else:
                    probs = logits.float().softmax(dim=-1)
                    
                all_probs.append(probs.detach().cpu())
                
                if save_embeddings:
                    hidden_states = outputs.hidden_states
                    emb = hidden_states[-1][:, 0, :]
                    all_embs.append(emb.detach().cpu())
        
        probabilities = torch.cat(all_probs, dim=0).numpy()
        embeddings = torch.cat(all_embs, dim=0).numpy() if save_embeddings else None
        
        return probabilities, embeddings
    
    def predict_stage1(
        self,
        sequences: List[str],
        ids: List[str],
        batch_size: int = 8,
        max_length: int = 1024,
        threshold: float = 0.95,
        save_embeddings: bool = False,
        dataloader_workers: int = 4,
    ) -> Dict:
        if self.binary_model is None:
            raise RuntimeError("Binary model not loaded. Call load_binary_model() first.")
        
        dataset = ProteinSequenceDataset(
            sequences=sequences,
            ids=ids,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        probabilities, embeddings = self.inference(
            dataset=dataset,
            model=self.binary_model,
            data_collator=self.data_collator,
            batch_size=batch_size,
            save_embeddings=save_embeddings,
            is_multi_label=False,
            dataloader_workers=dataloader_workers,
        )
        
        positive_mask = probabilities[:, 1] > threshold
        positive_ids = set([ids[i] for i in range(len(ids)) if positive_mask[i]])
        predicted_labels = [self.binary_id_to_label[1 if mask else 0] for mask in positive_mask]
        
        print(f"\nBinary Classification Results:")
        print(f"   Total sequences: {len(ids)}")
        print(f"   Positive (CAZy): {len(positive_ids)} ({len(positive_ids)/len(ids)*100:.2f}%)")
        print(f"   Negative (Non-CAZy): {len(ids) - len(positive_ids)} ({(len(ids) - len(positive_ids))/len(ids)*100:.2f}%)")
        
        return {
            'probabilities': probabilities,
            'predicted_labels': predicted_labels,
            'positive_ids': positive_ids,
            'embeddings': embeddings,
            'ids': ids,
            'threshold': threshold
        }
    
    def predict_stage2(
        self,
        sequences: List[str],
        ids: List[str],
        batch_size: int = 8,
        max_length: int = 1024,
        thresholds: Optional[Sequence[float]] = None,
        thresholds_file: Optional[str] = None,
        global_threshold: float = 0.5,
        save_embeddings: bool = False,
        dataloader_workers: int = 4,
    ) -> Dict:
        if self.multi_model is None:
            raise RuntimeError("Multi-label model not loaded. Call load_multi_model() first.")
        
        if len(sequences) == 0:
            print("\n⚠️ No sequences provided for Stage 2")
            return None
        
        dataset = ProteinSequenceDataset(
            sequences=sequences,
            ids=ids,
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        probabilities, embeddings = self.inference(
            dataset=dataset,
            model=self.multi_model,
            data_collator=self.data_collator,
            batch_size=batch_size,
            save_embeddings=save_embeddings,
            is_multi_label=True,
            dataloader_workers=dataloader_workers,
        )
        
        per_class_thr = self.load_thresholds(
            classes=self.multi_classes,
            global_threshold=global_threshold,
            thresholds_list=thresholds,
            thresholds_file=thresholds_file
        )
        
        predictions = (probabilities >= per_class_thr[None, :]).astype(int)
        
        predicted_label_lists = []
        for i in range(len(ids)):
            labels = [self.multi_classes[j] for j in range(len(self.multi_classes)) if predictions[i, j] == 1]
            predicted_label_lists.append(labels)
        
        class_counts = predictions.sum(axis=0)
        print(f"\nMulti-label Classification Results:")
        for j, class_name in enumerate(self.multi_classes):
            print(f"   {class_name}: {int(class_counts[j])} ({class_counts[j]/len(ids)*100:.2f}%)")
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'predicted_labels': predicted_label_lists,
            'embeddings': embeddings,
            'ids': ids,
            'thresholds': per_class_thr
        }
    
    def load_thresholds(
        self,
        classes: List[str],
        global_threshold: float,
        thresholds_list: Optional[Sequence[float]] = None,
        thresholds_file: Optional[str] = None
    ) -> np.ndarray:
        if thresholds_file:
            with open(thresholds_file, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                thr = np.array([float(obj.get(c, global_threshold)) for c in classes], dtype=np.float32)
            elif isinstance(obj, list):
                if len(obj) != len(classes):
                    raise ValueError(f"thresholds_file list length {len(obj)} != num classes {len(classes)}")
                thr = np.array([float(x) for x in obj], dtype=np.float32)
            else:
                raise ValueError("thresholds_file JSON must be a dict or list")
            return thr
        
        if thresholds_list is not None:
            if len(thresholds_list) != len(classes):
                raise ValueError(f"thresholds length {len(thresholds_list)} != num classes {len(classes)}")
            return np.array([float(x) for x in thresholds_list], dtype=np.float32)
        
        return np.full(len(classes), float(global_threshold), dtype=np.float32)
    
    def save_final_results(
        self,
        stage1_results: Dict,
        stage2_results: Optional[Dict],
        output_dir: str,
        output_name: str
    ):
        os.makedirs(output_dir, exist_ok=True)
        
        stage2_map = {}
        if stage2_results:
            for i, seq_id in enumerate(stage2_results['ids']):
                stage2_map[seq_id] = {
                    'probs': stage2_results['probabilities'][i],
                    'preds': stage2_results['predictions'][i],
                    'labels': stage2_results['predicted_labels'][i]
                }
        
        final_path = f"{output_dir}/{output_name}_predictions.csv"
        with open(final_path, "w", newline="") as f:
            w = csv.writer(f)
            
            header = ["sequence_id", "pred_cazy", "prob_cazy"]
            header.append("pred_cazy_class")
            for class_name in self.multi_classes:
                header.append(f"pred_{class_name}")
            for class_name in self.multi_classes:
                header.append(f"prob_{class_name}")

            w.writerow(header)
            
            for i, seq_id in enumerate(stage1_results['ids']):
                row = [
                    seq_id,
                    stage1_results['predicted_labels'][i],
                    float(stage1_results['probabilities'][i, 1]),
                ]
                
                if seq_id in stage2_map:
                    result = stage2_map[seq_id]
                    row.append('|'.join(result['labels']) if result['labels'] else 'none')
                    for j, class_name in enumerate(self.multi_classes):
                        row.append(int(result['preds'][j]))
                    for j, class_name in enumerate(self.multi_classes):
                        row.append(float(result['probs'][j]))

                else:
                    row.append('N/A')
                    for class_name in self.multi_classes:
                        row.extend([0.0, 0])
                
                w.writerow(row)
        
        print(f"Saved final results to {final_path}")
        
        if stage2_results and stage2_results['embeddings'] is not None:
            emb_path = f"{output_dir}/{output_name}_stage2_embeddings.npy"
            np.save(emb_path, stage2_results['embeddings'])
            print(f"   Saved Stage 2 embeddings to {emb_path}")
            
            emb_csv_path = f"{output_dir}/{output_name}_stage2_embeddings.csv"
            with open(emb_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                for seq_id, emb in zip(stage2_results['ids'], stage2_results['embeddings']):
                    w.writerow([seq_id, *emb.tolist()])
            print(f"   Saved Stage 2 embeddings CSV to {emb_csv_path}")
    
    def save_statistics(
        self,
        stage1_results: Dict,
        stage2_results: Optional[Dict],
        output_dir: str,
        output_name: str
    ):
        stats_path = f"{output_dir}/{output_name}_statistics.csv"
        
        with open(stats_path, "w") as f:
            f.write("Category,Count,Percentage\n")
            
            total = len(stage1_results['ids'])
            f.write(f"Total sequences,{total},100.00\n")
            
            f.write("\n# Stage 1 - Binary Classification\n")
            cazy_count = len(stage1_results['positive_ids'])
            non_cazy_count = total - cazy_count
            f.write(f"CAZy,{cazy_count},{cazy_count/total*100:.2f}\n")
            f.write(f"Non-CAZy,{non_cazy_count},{non_cazy_count/total*100:.2f}\n")
            
            if stage2_results and cazy_count > 0:
                f.write("\n# Stage 2 - Multi-label Classification (CAZy samples only)\n")
                class_counts = stage2_results['predictions'].sum(axis=0)
                for j, class_name in enumerate(self.multi_classes):
                    count = int(class_counts[j])
                    f.write(f"{class_name},{count},{count/cazy_count*100:.2f}\n")
        
        print(f"Saved statistics to {stats_path}")
    
    def predict(
        self,
        test_fasta: str,
        binary_model_path: Optional[str] = None,
        multi_model_path: Optional[str] = None,
        binary_threshold: float = 0.5,
        multi_thresholds: Optional[List[float]] = None,
        multi_thresholds_file: Optional[str] = None,
        multi_global_threshold: float = 0.5,
        batch_size: int = 8,
        max_length: int = 1024,
        output_dir: str = "./outputs",
        output_name: str = "test",
        save_embeddings: bool = False,
        dataloader_workers: int = 4
    ):
        sequences, ids = self.load_sequences_from_fasta(test_fasta)

        stage1_results = None
        stage2_results = None
        
        print(f"\n{'='*60}")
        print("STAGE 1: Binary Classification (CAZy vs Non-CAZy)")
        print(f"{'='*60}")

        self.load_binary_model(binary_model_path)

        print(f"\nRunning on {len(sequences)} sequences from {test_fasta}")

        stage1_results = self.predict_stage1(
            sequences=sequences,
            ids=ids,
            batch_size=batch_size,
            max_length=max_length,
            threshold=binary_threshold,
            save_embeddings=save_embeddings,
            dataloader_workers=dataloader_workers,
        )
        
        self.stage1_results = stage1_results
        self.positive_ids = stage1_results['positive_ids']
        
        if len(self.positive_ids) > 0:
            positive_sequences = [self.all_sequences[id] for id in self.positive_ids if id in self.all_sequences]
            positive_ids_list = [id for id in self.positive_ids if id in self.all_sequences]
                    
            print(f"\n{'='*60}")
            print(f"STAGE 2: Multi-label Classification (GT, GH, CBM, CE, PL, AA)")
            print(f"{'='*60}")

            self.load_multi_model(multi_model_path)

            print(f"\nRunning on {len(positive_sequences)} positive sequences")

            if len(positive_sequences) > 0:
                stage2_results = self.predict_stage2(
                    sequences=positive_sequences,
                    ids=positive_ids_list,
                    batch_size=batch_size,
                    max_length=max_length,
                    thresholds=multi_thresholds,
                    thresholds_file=multi_thresholds_file,
                    global_threshold=multi_global_threshold,
                    save_embeddings=save_embeddings,
                    dataloader_workers=dataloader_workers,
                )
                self.stage2_results = stage2_results
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE!")
        print("="*60)

        self.save_final_results(
            stage1_results=stage1_results,
            stage2_results=stage2_results,
            output_dir=output_dir,
            output_name=output_name
        )
        
        self.save_statistics(
            stage1_results=stage1_results,
            stage2_results=stage2_results,
            output_dir=output_dir,
            output_name=output_name
        )
