param_grid = {
    # Embedding
    'embedding_model': [
        'all-MiniLM-L6-v2',           # baseline
        'BAAI/bge-large-en-v1.5',     # best open-source
        'text-embedding-3-small',      # OpenAI
        'text-embedding-004',          # Google (requires API change)
    ],
    'distance_metric': ['cosine', 'euclidean', 'dotproduct'],
    
    # Chunking
    'chunk_size': [500, 1000, 1500, 2000],
    'chunk_overlap': [0, 100, 200, 300],
    'chunking_strategy': [
        'fixed_size',                  # current
        'recursive_character',         # langchain
        'semantic',                    # sentence-based
        'markdown_aware',              # preserves structure
    ],
    
    # Retrieval
    'top_k': [3, 5, 7, 10],
    'rerank': [None, 'cohere-rerank-v3', 'cross-encoder/ms-marco-MiniLM-L-6-v2'],
    'rerank_top_n': [3, 5],           # after reranking
    
    # LLM
    'llm_model': [
        'gemini-2.0-flash-exp',
        'gemini-1.5-flash',
        'gemini-1.5-pro',
    ],
    'temperature': [0.0, 0.3, 0.7],
    'max_tokens': [512, 1024, 2048],
    
    # Prompt engineering
    'prompt_style': [
        'simple',                      # current
        'cot',                         # chain of thought
        'structured',                  # JSON output
    ],
    'include_sources_in_context': [True, False],
}