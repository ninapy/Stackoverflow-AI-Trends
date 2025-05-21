# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import datetime as dt

# %%
df = pd.read_csv('full_dataset.csv')

# %%

print("Dataset overview:")
print(df.head())
print(f"Shape: {df.shape}")
print(df.info())

# %%
df['CreationDate'] = pd.to_datetime(df['CreationDate'], format='ISO8601')
df['Year'] = df['CreationDate'].dt.year
df['Month'] = df['CreationDate'].dt.month
df['YearMonth'] = df['CreationDate'].dt.to_period('M')
df.head()

# %%
df[(df['Tags'].notna())]

# %%


df['CreationDate'] = pd.to_datetime(df['CreationDate'], format='ISO8601')
if df['CreationDate'].dt.tz is not None:
    test_date = pd.to_datetime('2022-11-01').tz_localize(df['CreationDate'].dt.tz[0])
else:
    test_date = pd.to_datetime('2022-11-01')

test_count = (df['CreationDate'] >= test_date).sum()
print(f"Posts after Nov, 2022: {test_count}")
before_chatgpt = df[df['CreationDate'] < test_date]
after_chatgpt = df[df['CreationDate'] >= test_date]
print(f"Posts before ChatGPT release: {len(before_chatgpt)}")
print(f"Posts after ChatGPT release: {len(after_chatgpt)}")




# %%
ai_tags = [
  'machine-learning', 'deep-learning', 'neural-network', 'artificial-intelligence', 
  'natural-language-processing', 'nlp', 'conv-neural-network', 'lstm', 
  'tensorflow', 'keras', 'pytorch', 'scikit-learn', 'computer-vision',
  'object-detection', 'face-recognition', 'face-detection', 'sentiment-analysis',
  'chatgpt', 'claude', 'gpt', 'llm', 'transformers', 'bert', 'rnn', 'cnn',
  'reinforcement-learning', 'supervised-learning', 'unsupervised-learning',
  'classification', 'regression', 'clustering', 'decision-tree', 'random-forest',
  'k-means', 'naive-bayes', 'svm', 'feature-selection', 'dimensionality-reduction',
  'word-embedding', 'word2vec', 'glove', 'fasttext', 'text-classification',
  'image-classification', 'object-detection-api', 'yolo', 'rcnn', 'fast-rcnn',
  'mask-rcnn', 'semantic-segmentation', 'instance-segmentation', 'transfer-learning',
  'fine-tuning', 'hyperparameter-tuning', 'activation-function', 'backpropagation',
  'gradient-descent', 'stochastic-gradient-descent', 'adam-optimizer', 'loss-function',
  'cross-entropy', 'mean-squared-error', 'precision-recall', 'f1-score', 'roc',
  'auc', 'confusion-matrix', 'overfitting', 'underfitting', 'regularization',
  'dropout', 'batch-normalization', 'data-augmentation', 'generative-adversarial-networks',
  'autoencoder', 'variational-autoencoder', 'self-supervised-learning',
  'semi-supervised-learning', 'few-shot-learning', 'zero-shot-learning',
  'one-shot-learning', 'meta-learning', 'multi-task-learning', 'ensemble-learning',
  'boosting', 'bagging', 'xgboost', 'lightgbm', 'catboost', 'adaboost',
  'gradient-boosting', 'time-series-forecasting', 'anomaly-detection',
  'recommendation-system', 'collaborative-filtering', 'content-based-filtering',
  'hybrid-recommendation', 'knowledge-graph', 'knowledge-representation',
  'ontology', 'semantic-web', 'reasoning', 'planning', 'search', 'a-star',
  'minimax', 'alpha-beta-pruning', 'monte-carlo-tree-search', 'genetic-algorithm',
  'evolutionary-algorithm', 'swarm-intelligence', 'particle-swarm-optimization',
  'ant-colony-optimization', 'fuzzy-logic', 'bayesian-network', 'markov-chain',
  'hidden-markov-model', 'conditional-random-field', 'crf', 'named-entity-recognition',
  'speech-recognition', 'speech-synthesis', 'text-to-speech', 'speech-to-text',
  'image-generation', 'text-generation', 'language-model', 'question-answering',
  'machine-translation', 'summarization', 'dialogue-system', 'chatbot',
  'information-retrieval', 'information-extraction', 'topic-modeling', 'lda',
  'word-sense-disambiguation', 'coreference-resolution', 'dependency-parsing',
  'constituency-parsing', 'part-of-speech-tagging', 'tokenization', 'stemming',
  'lemmatization', 'stop-words', 'tf-idf', 'bag-of-words', 'n-gram',
  'sequence-to-sequence', 'attention-mechanism', 'transformer', 'self-attention',
  'multi-head-attention', 'positional-encoding', 'beam-search', 'greedy-search',
  'teacher-forcing', 'curriculum-learning', 'contrastive-learning',
  'representation-learning', 'feature-learning', 'metric-learning',
  'siamese-network', 'triplet-loss', 'cosine-similarity', 'euclidean-distance',
  'manhattan-distance', 'mahalanobis-distance', 'jaccard-similarity',
  'pearson-correlation', 'spearman-correlation', 'kendall-tau',
  'mutual-information', 'information-gain', 'gini-index', 'entropy',
  'cross-validation', 'k-fold-cross-validation', 'leave-one-out-cross-validation',
  'stratified-sampling', 'data-preprocessing', 'feature-engineering',
  'feature-extraction', 'feature-transformation', 'normalization',
  'standardization', 'min-max-scaling', 'z-score-normalization',
  'principal-component-analysis', 'pca', 'singular-value-decomposition',
  'svd', 'linear-discriminant-analysis', 'lda', 'non-negative-matrix-factorization',
  'nmf', 't-sne', 'umap', 'isomap', 'locally-linear-embedding', 'lle',
  'spectral-embedding', 'manifold-learning', 'kernel-trick', 'radial-basis-function',
  'rbf', 'polynomial-kernel', 'linear-kernel', 'sigmoid-kernel',
  'chi-square-kernel', 'laplacian-kernel', 'cosine-kernel',
  'hamming-distance', 'levenshtein-distance', 'edit-distance',
  'jaro-winkler-distance', 'soundex', 'metaphone', 'phonetic-algorithm',
  'bloom-filter', 'locality-sensitive-hashing', 'lsh', 'approximate-nearest-neighbor',
  'ann', 'kd-tree', 'ball-tree', 'r-tree', 'quadtree', 'octree',
  'spatial-index', 'inverted-index', 'forward-index', 'posting-list',
  'term-frequency', 'document-frequency', 'inverse-document-frequency',
  'bm25', 'okapi-bm25', 'vector-space-model', 'boolean-retrieval-model',
  'probabilistic-retrieval-model', 'language-model-retrieval',
  'query-expansion', 'relevance-feedback', 'pseudo-relevance-feedback',
  'explicit-relevance-feedback', 'implicit-relevance-feedback',
  'click-through-rate', 'conversion-rate', 'bounce-rate', 'dwell-time',
  'user-satisfaction', 'user-engagement', 'user-experience', 'a-b-testing',
  'multivariate-testing', 'bandit-algorithm', 'thompson-sampling',
  'upper-confidence-bound', 'ucb', 'epsilon-greedy', 'explore-exploit',
  'multi-armed-bandit', 'contextual-bandit', 'online-learning',
  'incremental-learning', 'lifelong-learning', 'continual-learning',
  'catastrophic-forgetting', 'elastic-weight-consolidation', 'ewc',
  'progressive-neural-network', 'knowledge-distillation', 'model-compression',
  'quantization', 'pruning', 'sparsity', 'low-rank-approximation',
  'tensor-decomposition', 'cp-decomposition', 'tucker-decomposition',
  'tensor-train', 'tensor-ring', 'hierarchical-tensor', 'tensor-network',
  'mixed-precision-training', 'half-precision', 'single-precision',
  'double-precision', 'bfloat16', 'float16', 'float32', 'float64',
  'distributed-training', 'data-parallel', 'model-parallel',
  'pipeline-parallel', 'gradient-accumulation', 'gradient-checkpointing',
  'automatic-mixed-precision', 'amp', 'nvidia-apex', 'horovod',
  'parameter-server', 'all-reduce', 'ring-all-reduce', 'collective-communication',
  'mpi', 'nccl', 'gloo', 'openmpi', 'cuda', 'cudnn', 'cublas', 'cufft',
  'cusparse', 'curand', 'thrust', 'opencl', 'vulkan', 'metal', 'directml',
  'oneapi', 'dpcpp', 'sycl', 'rocm', 'hip', 'miopen', 'miopengemm',
  'tpu', 'vpu', 'dsp', 'fpga', 'asic', 'neuromorphic-computing',
  'quantum-computing', 'quantum-machine-learning', 'qml',
  'variational-quantum-eigensolver', 'vqe', 'quantum-approximate-optimization-algorithm',
  'qaoa', 'quantum-neural-network', 'qnn', 'quantum-kernel-method',
  'quantum-support-vector-machine', 'qsvm', 'quantum-k-means',
  'quantum-principal-component-analysis', 'qpca', 'quantum-boltzmann-machine',
  'qbm', 'quantum-generative-adversarial-network', 'qgan',
  'quantum-circuit-learning', 'qcl', 'quantum-transfer-learning',
  'quantum-reinforcement-learning', 'qrl', 'quantum-annealing',
  'adiabatic-quantum-computing', 'gate-based-quantum-computing',
  'measurement-based-quantum-computing', 'topological-quantum-computing',
  'quantum-error-correction', 'quantum-error-mitigation', 'qem',
  'quantum-noise', 'quantum-decoherence', 'quantum-gate', 'quantum-circuit',
  'quantum-state', 'quantum-measurement', 'quantum-teleportation',
  'quantum-entanglement', 'quantum-superposition', 'quantum-interference',
  'quantum-parallelism', 'quantum-speedup', 'quantum-advantage',
  'quantum-supremacy', 'quantum-volume', 'quantum-readiness',
  'quantum-safe-cryptography', 'post-quantum-cryptography', 'pqc',
  'lattice-based-cryptography', 'code-based-cryptography',
  'multivariate-cryptography', 'hash-based-cryptography',
  'isogeny-based-cryptography', 'quantum-key-distribution', 'qkd',
  'quantum-random-number-generator', 'qrng', 'quantum-secure-direct-communication',
  'qsdc', 'quantum-digital-signature', 'qds', 'quantum-money',
  'quantum-bitcoin', 'quantum-blockchain', 'quantum-internet',
  'quantum-network', 'quantum-repeater', 'quantum-memory',
  'quantum-sensor', 'quantum-metrology', 'quantum-imaging',
  'quantum-radar', 'quantum-lidar', 'quantum-navigation',
  'quantum-gravity', 'quantum-field-theory', 'quantum-mechanics',
  'quantum-information', 'quantum-computation', 'quantum-communication',
  'quantum-cryptography', 'quantum-simulation', 'quantum-chemistry',
  'quantum-biology', 'quantum-finance', 'quantum-optimization',
  'quantum-machine-intelligence', 'quantum-artificial-intelligence',
  'quantum-deep-learning', 'quantum-neural-network', 'quantum-convolutional-neural-network',
  'qcnn', 'quantum-recurrent-neural-network', 'qrnn',
  'quantum-transformer', 'quantum-attention', 'quantum-self-attention',
  'quantum-multi-head-attention', 'quantum-positional-encoding',
  'quantum-embedding', 'quantum-feature-map', 'quantum-kernel',
  'quantum-distance', 'quantum-similarity', 'quantum-clustering',
  'quantum-classification', 'quantum-regression', 'quantum-anomaly-detection',
  'quantum-recommendation-system', 'quantum-collaborative-filtering',
  'quantum-content-based-filtering', 'quantum-hybrid-recommendation',
  'quantum-knowledge-graph', 'quantum-knowledge-representation',
  'quantum-ontology', 'quantum-semantic-web', 'quantum-reasoning',
  'quantum-planning', 'quantum-search', 'quantum-a-star',
  'quantum-minimax', 'quantum-alpha-beta-pruning', 'quantum-monte-carlo-tree-search',
  'quantum-genetic-algorithm', 'quantum-evolutionary-algorithm',
  'quantum-swarm-intelligence', 'quantum-particle-swarm-optimization',
  'quantum-ant-colony-optimization', 'quantum-fuzzy-logic',
  'quantum-bayesian-network', 'quantum-markov-chain',
  'quantum-hidden-markov-model', 'quantum-conditional-random-field',
  'quantum-named-entity-recognition', 'quantum-speech-recognition',
  'quantum-speech-synthesis', 'quantum-text-to-speech', 'quantum-speech-to-text',
  'quantum-image-generation', 'quantum-text-generation', 'quantum-language-model',
  'quantum-question-answering', 'quantum-machine-translation',
  'quantum-summarization', 'quantum-dialogue-system', 'quantum-chatbot',
  'quantum-information-retrieval', 'quantum-information-extraction',
  'quantum-topic-modeling', 'quantum-lda', 'quantum-word-sense-disambiguation',
  'quantum-coreference-resolution', 'quantum-dependency-parsing',
  'quantum-constituency-parsing', 'quantum-part-of-speech-tagging',
  'quantum-tokenization', 'quantum-stemming', 'quantum-lemmatization',
  'quantum-stop-words', 'quantum-tf-idf', 'quantum-bag-of-words',
  'quantum-n-gram', 'quantum-sequence-to-sequence', 'quantum-attention-mechanism',
  'quantum-transformer', 'quantum-self-attention', 'quantum-multi-head-attention',
  'quantum-positional-encoding', 'quantum-beam-search', 'quantum-greedy-search',
  'quantum-teacher-forcing', 'quantum-curriculum-learning',
  'quantum-contrastive-learning', 'quantum-representation-learning',
  'quantum-feature-learning', 'quantum-metric-learning',
  'quantum-siamese-network', 'quantum-triplet-loss', 'quantum-cosine-similarity',
  'quantum-euclidean-distance', 'quantum-manhattan-distance',
  'quantum-mahalanobis-distance', 'quantum-jaccard-similarity',
  'quantum-pearson-correlation', 'quantum-spearman-correlation',
  'quantum-kendall-tau', 'quantum-mutual-information', 'quantum-information-gain',
  'quantum-gini-index', 'quantum-entropy', 'quantum-cross-validation',
  'quantum-k-fold-cross-validation', 'quantum-leave-one-out-cross-validation',
  'quantum-stratified-sampling', 'quantum-data-preprocessing',
  'quantum-feature-engineering', 'quantum-feature-extraction',
  'quantum-feature-transformation', 'quantum-normalization',
  'quantum-standardization', 'quantum-min-max-scaling',
  'quantum-z-score-normalization', 'quantum-principal-component-analysis',
  'qpca', 'quantum-singular-value-decomposition', 'qsvd',
  'quantum-linear-discriminant-analysis', 'qlda',
  'quantum-non-negative-matrix-factorization', 'qnmf',
  'quantum-t-sne', 'quantum-umap', 'quantum-isomap',
  'quantum-locally-linear-embedding', 'qlle', 'quantum-spectral-embedding',
  'quantum-manifold-learning', 'quantum-kernel-trick',
  'quantum-radial-basis-function', 'qrbf', 'quantum-polynomial-kernel',
  'quantum-linear-kernel', 'quantum-sigmoid-kernel', 'quantum-chi-square-kernel',
  'quantum-laplacian-kernel', 'quantum-cosine-kernel', 'quantum-hamming-distance',
  'quantum-levenshtein-distance', 'quantum-edit-distance',
  'quantum-jaro-winkler-distance', 'quantum-soundex', 'quantum-metaphone',
  'quantum-phonetic-algorithm', 'quantum-bloom-filter',
  'quantum-locality-sensitive-hashing', 'qlsh',
  'quantum-approximate-nearest-neighbor', 'qann', 'quantum-kd-tree',
  'quantum-ball-tree', 'quantum-r-tree', 'quantum-quadtree',
  'quantum-octree', 'quantum-spatial-index', 'quantum-inverted-index',
  'quantum-forward-index', 'quantum-posting-list', 'quantum-term-frequency',
  'quantum-document-frequency', 'quantum-inverse-document-frequency',
  'quantum-bm25', 'quantum-okapi-bm25', 'quantum-vector-space-model',
  'quantum-boolean-retrieval-model', 'quantum-probabilistic-retrieval-model',
  'quantum-language-model-retrieval', 'quantum-query-expansion',
  'quantum-relevance-feedback', 'quantum-pseudo-relevance-feedback',
  'quantum-explicit-relevance-feedback', 'quantum-implicit-relevance-feedback',
  'quantum-click-through-rate', 'quantum-conversion-rate',
  'quantum-bounce-rate', 'quantum-dwell-time', 'quantum-user-satisfaction',
  'quantum-user-engagement', 'quantum-user-experience', 'quantum-a-b-testing',
  'quantum-multivariate-testing', 'quantum-bandit-algorithm',
  'quantum-thompson-sampling', 'quantum-upper-confidence-bound',
  'qucb', 'quantum-epsilon-greedy', 'quantum-explore-exploit',
  'quantum-multi-armed-bandit', 'quantum-contextual-bandit',
  'quantum-online-learning', 'quantum-incremental-learning',
  'quantum-lifelong-learning', 'quantum-continual-learning',
  'quantum-catastrophic-forgetting', 'quantum-elastic-weight-consolidation',
  'qewc', 'quantum-progressive-neural-network', 'quantum-knowledge-distillation',
  'quantum-model-compression', 'quantum-quantization', 'quantum-pruning',
  'quantum-sparsity', 'quantum-low-rank-approximation',
  'quantum-tensor-decomposition', 'quantum-cp-decomposition',
  'quantum-tucker-decomposition', 'quantum-tensor-train',
  'quantum-tensor-ring', 'quantum-hierarchical-tensor',
  'quantum-tensor-network', 'quantum-mixed-precision-training',
  'quantum-half-precision', 'quantum-single-precision',
  'quantum-double-precision', 'quantum-bfloat16', 'quantum-float16',
  'quantum-float32', 'quantum-float64', 'quantum-distributed-training',
  'quantum-data-parallel', 'quantum-model-parallel',
  'quantum-pipeline-parallel', 'quantum-gradient-accumulation',
  'quantum-gradient-checkpointing', 'quantum-automatic-mixed-precision',
  'qamp', 'quantum-nvidia-apex', 'quantum-horovod',
  'quantum-parameter-server', 'quantum-all-reduce',
  'quantum-ring-all-reduce', 'quantum-collective-communication',
  'quantum-mpi', 'quantum-nccl', 'quantum-gloo', 'quantum-openmpi',
  'quantum-cuda', 'quantum-cudnn', 'quantum-cublas', 'quantum-cufft',
  'quantum-cusparse', 'quantum-curand', 'quantum-thrust',
  'quantum-opencl', 'quantum-vulkan', 'quantum-metal',
  'quantum-directml', 'quantum-oneapi', 'quantum-dpcpp',
  'quantum-sycl', 'quantum-rocm', 'quantum-hip', 'quantum-miopen',
  'quantum-miopengemm', 'quantum-tpu', 'quantum-vpu', 'quantum-dsp',
  'quantum-fpga', 'quantum-asic', 'quantum-neuromorphic-computing'
];
language_tags = [
  'python', 'javascript', 'java', 'c#', 'php',
  'c++', 'typescript', 'ruby', 'swift', 'kotlin',
  'go', 'rust', 'scala', 'r', 'perl',
  'objective-c', 'c', 'bash', 'lua', 'haskell', 'pandas', 'numpy', 'pytorch', 'tensorflow', 'keras'
];


# %%


# %%
def contains_any_tag(tags_str, tag_list):
    if isinstance(tags_str, str):
        tags = tags_str.lower().split('|')
        return any(tag in tag_list for tag in tags)
    return False

# %%
def is_related(row, ai_terms, check_body=True, check_title=True, check_tags=True):
    is_ai = False
    
    if check_tags and isinstance(row.get('Tags'), str):
        tags = row['Tags'].lower()
        is_ai = contains_any_tag(tags, ai_terms)
        if is_ai:
            return True
    
    # Check title (faster than body)
    if check_title and isinstance(row.get('Title'), str):
        title = row['Title'].lower()
        # Use word boundaries to avoid partial matches
        is_ai = any(re.search(r'\b{}\b'.format(term), title) for term in ai_terms)
        if is_ai:
            return True
    
    # Check body (most comprehensive but slowest)
    if check_body and isinstance(row.get('Body'), str):
        # Strip HTML tags first
        body = re.sub('<[^<]+?>', ' ', row['Body']).lower()
        # Use word boundaries to avoid partial matches
        is_ai = any(re.search(r'\b{}\b'.format(term), body) for term in ai_terms)
        if is_ai:
            return True
            
    return is_ai

ai_terms_expanded = []
for term in ai_tags:
    if '-' in term:
        ai_terms_expanded.append(term)
        ai_terms_expanded.append(term.replace('-', ''))
    else:
        ai_terms_expanded.append(term)

import re
df['is_ai_content'] = df.apply(lambda row: is_related(row, ai_terms_expanded), axis=1)
df.to_csv("full_dataset_labelled_ai.csv", index=False)

# %%
df['has_ai_tag'] = df['Tags'].apply(lambda x: contains_any_tag(x, ai_tags))
tag_count = df['has_ai_tag'].sum()
content_count = df['is_ai_content'].sum()
print(f"AI posts identified by tags only: {tag_count}")
print(f"AI posts identified by content analysis: {content_count}")
print(f"Additional AI posts found: {content_count - tag_count}")

# %%
for lang in language_tags:
    df[f'has_{lang}_tag'] = df['Tags'].apply(lambda x: contains_any_tag(x, [lang]))

# %%
df.head()

# %%
chatgpt_release = pd.to_datetime('2022-11-01')

df['post_chatgpt'] = df['CreationDate'] >= chatgpt_release

# %%
df[df['post_chatgpt'] == True]

# %%
df["quarter_start"] = pd.to_datetime(df["CreationDate"]).dt.to_period("Q").dt.start_time
ai_per_q = df[df["is_ai_content"]].groupby("quarter_start").size().rename("ai_posts")
total_per_q = df.groupby("quarter_start").size().rename("total_posts")
by_q = pd.concat([ai_per_q, total_per_q], axis=1).fillna(0)

by_q["ai_proportion"] = by_q["ai_posts"] / by_q["total_posts"]
for lang in language_tags:
    lang_col = f'has_{lang}_tag'
    lang_per_q = df[df[lang_col]].groupby("quarter_start").size().rename(f"{lang}_posts")
    by_q = pd.concat([by_q, lang_per_q], axis=1).fillna(0)
    by_q[f"{lang}_proportion"] = by_q[f"{lang}_posts"] / by_q["total_posts"]
plt.figure(figsize=(14, 8))
top_langs = sorted(
    [(lang, by_q[f"{lang}_proportion"].iloc[-1]) for lang in language_tags],
    key=lambda x: x[1], 
    reverse=True
)[:5]
for lang, prop in top_langs:
    plt.plot(by_q.index, by_q[f"{lang}_proportion"], label=f"{lang} posts", linewidth=2)
plt.axvline(x=chatgpt_release, color='r', linestyle='--', label='ChatGPT Release')
plt.title('Proportion of Posts by Category by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Proportion of Posts')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('quarterly_trends.png')
plt.show()

# %%
test_df = df[df['CreationDate'] >= pd.to_datetime('2021-05-01')]
contingency = pd.crosstab(test_df['post_chatgpt'], test_df['is_ai_content'])
print("Contingency Table:")
print(contingency)


pre_chatgpt = test_df[test_df['post_chatgpt'] == False]
post_chatgpt = test_df[test_df['post_chatgpt'] == True]
pre_ai_count = pre_chatgpt['is_ai_content'].sum()
pre_total = len(pre_chatgpt)
pre_ai_prop = pre_ai_count / pre_total
post_ai_count = post_chatgpt['is_ai_content'].sum()
post_total = len(post_chatgpt)
post_ai_prop = post_ai_count / post_total

print("\nPre-ChatGPT (2021-05-01 - 2022-11-01) vs Post-ChatGPT (2022-11-01 - 2024-05-01) Analysis:")
print(f"Pre-ChatGPT posts: {pre_total}")
print(f"Post-ChatGPT posts: {post_total}")
print(f"Pre-ChatGPT AI proportion: {pre_ai_prop:.4f} ({pre_ai_count} AI posts)")
print(f"Post-ChatGPT AI proportion: {post_ai_prop:.4f} ({post_ai_count} AI posts)")
print(f"Absolute difference: {abs(post_ai_prop - pre_ai_prop):.4f}")
print(f"Relative change: {((post_ai_prop / pre_ai_prop) - 1) * 100:.2f}%")

#Chi-square test
chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
print("\nChi-square Test:")
print(f"Chi-square value: {chi2:.4f}")
print(f"p-value: {p_chi2:.8f}")
print(f"Significant difference at α=0.05: {p_chi2 < 0.05}")