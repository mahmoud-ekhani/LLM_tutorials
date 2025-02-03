import os
import pickle
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
import nltk
from bert_score import score
import spacy
from sentence_transformers import SentenceTransformer
import faiss
import torch.nn as nn

# Download required NLTK data
nltk.download('punkt')

# Rest of the code remains the same...
# Download required NLTK data
nltk.download('punkt')

# Define available medical QA models
MEDICAL_QA_MODELS = {
    "google/flan-t5-large": {
        "parameters": "780M",
        "description": "Strong general-purpose model, good at following instructions",
        "strengths": "Versatile, good at structured responses"
    },
    "GanjinZero/biomedical-flan-t5-large": {
        "parameters": "780M",
        "description": "FLAN-T5 fine-tuned on medical datasets",
        "strengths": "Specialized for medical domain"
    },
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract": {
        "parameters": "110M",
        "description": "Trained on PubMed abstracts",
        "strengths": "Strong medical domain knowledge"
    },
    "epfl-llm/medical-llama-7b": {
        "parameters": "7B",
        "description": "LLaMA fine-tuned on medical data",
        "strengths": "Comprehensive medical knowledge"
    }
}

class MultiGPUConfig:
    def __init__(self):
        self.n_gpus = torch.cuda.device_count()
        self.using_multi_gpu = self.n_gpus > 1
        
    def setup_distributed(self, rank):
        """Initialize distributed training"""
        if self.using_multi_gpu:
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:12355',
                world_size=self.n_gpus,
                rank=rank
            )
            torch.cuda.set_device(rank)

def initialize_qa_model(gpu_config, rank=0, model_name=None):
    """Initialize QA model with multi-GPU support"""
    if rank == 0:  # Only print model info on main process
        print("\nAvailable Medical QA Models:")
        for name, info in MEDICAL_QA_MODELS.items():
            print(f"\n{name}:")
            print(f"Parameters: {info['parameters']}")
            print(f"Description: {info['description']}")
            print(f"Strengths: {info['strengths']}")

    # Use provided model name or default
    selected_model = model_name or "GanjinZero/biomedical-flan-t5-large"
    if rank == 0:
        print(f"\nUsing model: {selected_model}")

    tokenizer = AutoTokenizer.from_pretrained(selected_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(selected_model)

    if gpu_config.using_multi_gpu:
        # Multi-GPU setup
        device = torch.device(f'cuda:{rank}')
        model = model.to(device)
        model = DDP(model, device_ids=[rank])
        if rank == 0:
            print(f"Using GPU {rank} in distributed mode")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        if rank == 0:
            print("Using single GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        if rank == 0:
            print("Using MPS backend for Apple Silicon")
    else:
        device = torch.device("cpu")
        model = model.to(device)
        if rank == 0:
            print("Using CPU")
    
    return model, tokenizer, device

class ParallelQualityMetrics:
    def __init__(self, gpu_config, rank=0):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.nlp = spacy.load('en_core_web_sm')
        self.gpu_config = gpu_config
        self.rank = rank
        
        # Move BERT Score to appropriate device
        if gpu_config.using_multi_gpu:
            self.device = f'cuda:{rank}'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def calculate_metrics_batch(self, batch_texts, batch_contexts=None):
        """Process multiple texts in parallel"""
        if self.gpu_config.using_multi_gpu:
            # Split batch across GPUs
            local_batch_size = len(batch_texts) // self.gpu_config.n_gpus
            start_idx = local_batch_size * self.rank
            end_idx = start_idx + local_batch_size
            
            local_texts = batch_texts[start_idx:end_idx]
            local_contexts = batch_contexts[start_idx:end_idx] if batch_contexts else None
            
            # Process local batch
            local_results = [self.calculate_metrics(text, context=ctx) 
                           for text, ctx in zip(local_texts, local_contexts or [None] * len(local_texts))]
            
            # Gather results from all GPUs
            all_results = [None] * len(batch_texts)
            dist.all_gather_object(all_results, local_results)
            
            return all_results
        else:
            return [self.calculate_metrics(text, context=ctx) 
                   for text, ctx in zip(batch_texts, batch_contexts or [None] * len(batch_texts))]
    
    def calculate_metrics(self, generated_text, reference_text=None, context=None):
        metrics = {}
        
        # Content relevance (if context is provided)
        if context:
            metrics['context_relevance'] = self._calculate_context_relevance(generated_text, context)
        
        # Medical entity coverage
        metrics['medical_entities'] = self._count_medical_entities(generated_text)
        
        # Response length and complexity
        metrics['response_length'] = len(generated_text.split())
        metrics['avg_word_length'] = np.mean([len(word) for word in generated_text.split()])
        
        # Reference-based metrics (if reference is provided)
        if reference_text:
            rouge_scores = self.rouge_scorer.score(generated_text, reference_text)
            metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
            metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
            metrics['rougeL'] = rouge_scores['rougeL'].fmeasure
            
            # BLEU score
            reference_tokens = [reference_text.split()]
            candidate_tokens = generated_text.split()
            metrics['bleu'] = sentence_bleu(reference_tokens, candidate_tokens)
            
            # BERTScore
            P, R, F1 = score([generated_text], [reference_text], lang='en', verbose=False)
            metrics['bert_score'] = F1.mean().item()
        
        return metrics
    
    def _calculate_context_relevance(self, text, context):
        """Calculate semantic similarity between generated text and context"""
        doc1 = self.nlp(text)
        doc2 = self.nlp(context)
        return doc1.similarity(doc2)
    
    def _count_medical_entities(self, text):
        """Count medical entities in text using spaCy"""
        doc = self.nlp(text)
        medical_ents = [ent for ent in doc.ents if ent.label_ in ['DISEASE', 'CHEMICAL', 'PROCEDURE']]
        return len(medical_ents)

def retrieve_contexts(query, index, encoder, knowledge_base, k=3):
    """Retrieve relevant contexts for a query"""
    try:
        # Handle both DataParallel and regular encoder cases
        if isinstance(encoder, nn.DataParallel):
            actual_encoder = encoder.module
        else:
            actual_encoder = encoder
            
        # Move query to same device as encoder
        device = next(actual_encoder.parameters()).device
        
        # Encode query
        with torch.no_grad():
            query_vector = actual_encoder.encode([query], convert_to_numpy=True, device=device)
        
        # Search index
        distances, indices = index.search(query_vector, k)
        
        # Return relevant contexts with metadata and distances
        retrieved = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(knowledge_base):  # Validate index
                context = knowledge_base[idx].copy()
                context['distance'] = float(distance)
                retrieved.append(context)
        
        return retrieved
        
    except Exception as e:
        print(f"Error retrieving contexts: {e}")
        return []

def parallel_rag_infer(questions, kb, index, encoder, qa_model, tokenizer, device, metrics, 
                      gpu_config, rank=0, use_context=True, top_k=3, batch_size=8):
    """Parallel RAG-enhanced medical QA inference"""
    
    results = []
    
    # Create batches
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        
        if use_context:
            # Parallel context retrieval
            batch_contexts = []
            for question in batch_questions:
                contexts = retrieve_contexts(question, index, encoder, kb, k=top_k)
                # Extract just the text from contexts
                combined_context = "\n".join([ctx['text'] for ctx in contexts])
                batch_contexts.append(combined_context)
            
            # Prepare prompts with context
            prompts = [
                f"Answer the medical question based on the following context:\nContext: {ctx}\nQuestion: {q}\nAnswer:"
                for q, ctx in zip(batch_questions, batch_contexts)
            ]
        else:
            batch_contexts = None
            prompts = [
                f"Answer the medical question:\nQuestion: {q}\nAnswer:"
                for q in batch_questions
            ]
        
        # Tokenize batch
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            max_length=1024, 
            truncation=True, 
            padding=True
        ).to(device)
        
        # Generate answers in parallel
        with torch.no_grad():
            outputs = qa_model.module.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True
            ) if isinstance(qa_model, DDP) else qa_model.generate(
                inputs.input_ids,
                max_length=256,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                early_stopping=True
            )
            
            batch_answers = [
                tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
        
        # Calculate metrics in parallel
        batch_metrics = metrics.calculate_metrics_batch(
            batch_answers,
            batch_contexts if use_context else None
        )
        
        # Combine results
        for q, a, m, c in zip(batch_questions, batch_answers, batch_metrics, 
                             batch_contexts if use_context else [None] * len(batch_questions)):
            results.append({
                'question': q,
                'context': c,
                'answer': a,
                'metrics': m
            })
    
    return results

def compare_rag_performance_parallel(model_name=None):
    """Compare and evaluate RAG vs. no-RAG performance using multiple GPUs"""
    
    # Initialize multi-GPU setup
    gpu_config = MultiGPUConfig()
    
    if gpu_config.using_multi_gpu:
        print(f"Using {gpu_config.n_gpus} GPUs")
        
        # Launch processes for each GPU
        torch.multiprocessing.spawn(
            run_distributed_comparison,
            args=(gpu_config, model_name),
            nprocs=gpu_config.n_gpus
        )
    else:
        # Single GPU or CPU mode
        run_distributed_comparison(0, gpu_config, model_name)

def run_distributed_comparison(rank, gpu_config, model_name=None):
    """Run comparison on a single GPU in distributed setting"""
    
    if gpu_config.using_multi_gpu:
        gpu_config.setup_distributed(rank)
    
    # Initialize components with specified model
    qa_model, tokenizer, device = initialize_qa_model(gpu_config, rank, model_name)
    metrics = ParallelQualityMetrics(gpu_config, rank)
    
    # Load knowledge base and retrieval components
    try:
        with open("medical_knowledge_base.pkl", 'rb') as f:
            kb = pickle.load(f)
        index = faiss.read_index("faiss_index.bin")
        with open("model_name.txt", 'r') as f:
            model_name = f.read().strip()
        encoder = SentenceTransformer(model_name)
        
        if gpu_config.using_multi_gpu:
            encoder = encoder.to(f'cuda:{rank}')
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return
    
    # Test questions
    test_questions = [
        "What are the early symptoms of diabetes?",
        "How is rheumatoid arthritis diagnosed?",
        "What are the common side effects of chemotherapy?",
        "How is high blood pressure treated?",
        "What causes migraine headaches?"
    ]
    
    if rank == 0:  # Only print on main process
        print("\nComparing RAG vs. No-RAG Performance:")
        print("=" * 80)
    
    # Generate answers with and without RAG in parallel
    rag_results = parallel_rag_infer(
        test_questions, kb, index, encoder, qa_model, tokenizer, 
        device, metrics, gpu_config, rank, use_context=True
    )
    
    no_rag_results = parallel_rag_infer(
        test_questions, kb, index, encoder, qa_model, tokenizer, 
        device, metrics, gpu_config, rank, use_context=False
    )
    
    if rank == 0:  # Only print results on main process
        print("\nResults with RAG:")
        for result in rag_results:
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Metrics: {result['metrics']}")
            
        print("\nResults without RAG:")
        for result in no_rag_results:
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Metrics: {result['metrics']}")
        
        if gpu_config.using_multi_gpu:
            dist.destroy_process_group()

if __name__ == "__main__":
    compare_rag_performance_parallel()