# Retrieval comparison

Corpus `data/documentation.json`; embedding model `all-MiniLM-L6-v2@1110a24`. Document-level hits against `labels.expected_doc_ids`; empty rates at each retriever's default floor. Produced by `python scripts/compare_retrieval.py`.

### development

| Chunking | Retriever | Passages | Floor | hit@1 | hit@3 | Empty (answerable) | Empty (unanswerable) | hit@3 fluent | hit@3 non-fluent |
|---|---|---|---|---|---|---|---|---|---|
| article | lexical | 29 | 0.26 | 0.7395 | 0.7955 | 0.1176 | 0.1608 | 0.8074 | 0.7586 |
| article | embedding | 29 | 0.35 | 0.8908 | 0.9356 | 0.014 | 0.1259 | 0.9519 | 0.8851 |
| section | lexical | 116 | 0.26 | 0.6751 | 0.7031 | 0.2297 | 0.3217 | 0.7222 | 0.6437 |
| section | embedding | 116 | 0.35 | 0.8992 | 0.9496 | 0.0056 | 0.0979 | 0.9667 | 0.8966 |
| window | lexical | 58 | 0.26 | 0.7339 | 0.7731 | 0.1401 | 0.1818 | 0.7852 | 0.7356 |
| window | embedding | 58 | 0.35 | 0.8908 | 0.9356 | 0.0112 | 0.1189 | 0.9519 | 0.8851 |

### validation

| Chunking | Retriever | Passages | Floor | hit@1 | hit@3 | Empty (answerable) | Empty (unanswerable) | hit@3 fluent | hit@3 non-fluent |
|---|---|---|---|---|---|---|---|---|---|
| article | lexical | 29 | 0.26 | 0.7358 | 0.8113 | 0.0943 | 0.2963 | 0.7727 | 1.0 |
| article | embedding | 29 | 0.35 | 0.9434 | 0.9623 | 0.0189 | 0.2593 | 0.9545 | 1.0 |
| section | lexical | 116 | 0.26 | 0.6604 | 0.6981 | 0.2453 | 0.5185 | 0.6818 | 0.7778 |
| section | embedding | 116 | 0.35 | 0.9434 | 0.9623 | 0.0189 | 0.2593 | 0.9545 | 1.0 |
| window | lexical | 58 | 0.26 | 0.7358 | 0.7925 | 0.0943 | 0.2963 | 0.75 | 1.0 |
| window | embedding | 58 | 0.35 | 0.9623 | 0.9623 | 0.0 | 0.2593 | 0.9545 | 1.0 |

Validation has only 9 answerable non-fluent tickets, so its non-fluent column is not a reliable estimate; use the development figures.

### Floor sweep (development)

| Chunking | Retriever | Floor | hit@1 | hit@3 | Empty (answerable) | Empty (unanswerable) |
|---|---|---|---|---|---|---|
| article | lexical | 0.26 | 0.7395 | 0.7955 | 0.1176 | 0.1608 |
| article | lexical | 0.30 | 0.6975 | 0.7227 | 0.1821 | 0.2238 |
| article | lexical | 0.35 | 0.6359 | 0.6583 | 0.2745 | 0.3566 |
| article | lexical | 0.40 | 0.4986 | 0.5098 | 0.4342 | 0.5315 |
| article | lexical | 0.45 | 0.3725 | 0.3754 | 0.5938 | 0.6993 |
| article | lexical | 0.50 | 0.2997 | 0.2997 | 0.6723 | 0.7902 |
| article | embedding | 0.20 | 0.8992 | 0.9496 | 0.0 | 0.0699 |
| article | embedding | 0.25 | 0.8992 | 0.9468 | 0.0028 | 0.1049 |
| article | embedding | 0.30 | 0.8964 | 0.9412 | 0.0084 | 0.1259 |
| article | embedding | 0.35 | 0.8908 | 0.9356 | 0.014 | 0.1259 |
| article | embedding | 0.40 | 0.888 | 0.9244 | 0.0224 | 0.1469 |
| article | embedding | 0.45 | 0.8403 | 0.8543 | 0.0728 | 0.2168 |
| article | embedding | 0.50 | 0.7227 | 0.7311 | 0.2073 | 0.3497 |
| section | lexical | 0.26 | 0.6751 | 0.7031 | 0.2297 | 0.3217 |
| section | lexical | 0.30 | 0.5294 | 0.5406 | 0.4118 | 0.5105 |
| section | lexical | 0.35 | 0.3697 | 0.3697 | 0.5966 | 0.6993 |
| section | lexical | 0.40 | 0.2493 | 0.2493 | 0.7283 | 0.8182 |
| section | lexical | 0.45 | 0.1261 | 0.1261 | 0.8543 | 0.951 |
| section | lexical | 0.50 | 0.07 | 0.07 | 0.916 | 0.972 |
| section | embedding | 0.20 | 0.8992 | 0.958 | 0.0 | 0.0 |
| section | embedding | 0.25 | 0.8992 | 0.958 | 0.0 | 0.035 |
| section | embedding | 0.30 | 0.8992 | 0.9552 | 0.0 | 0.049 |
| section | embedding | 0.35 | 0.8992 | 0.9496 | 0.0056 | 0.0979 |
| section | embedding | 0.40 | 0.8936 | 0.9412 | 0.0112 | 0.1329 |
| section | embedding | 0.45 | 0.8739 | 0.9132 | 0.0364 | 0.1469 |
| section | embedding | 0.50 | 0.8627 | 0.8796 | 0.0476 | 0.1469 |
| window | lexical | 0.26 | 0.7339 | 0.7731 | 0.1401 | 0.1818 |
| window | lexical | 0.30 | 0.6723 | 0.6835 | 0.2297 | 0.2937 |
| window | lexical | 0.35 | 0.5994 | 0.6078 | 0.3333 | 0.4615 |
| window | lexical | 0.40 | 0.4258 | 0.4286 | 0.535 | 0.6434 |
| window | lexical | 0.45 | 0.3081 | 0.3109 | 0.6583 | 0.8112 |
| window | lexical | 0.50 | 0.2353 | 0.2353 | 0.7395 | 0.8531 |
| window | embedding | 0.20 | 0.8964 | 0.9524 | 0.0 | 0.042 |
| window | embedding | 0.25 | 0.8964 | 0.9524 | 0.0 | 0.0769 |
| window | embedding | 0.30 | 0.8964 | 0.944 | 0.0028 | 0.0839 |
| window | embedding | 0.35 | 0.8908 | 0.9356 | 0.0112 | 0.1189 |
| window | embedding | 0.40 | 0.8824 | 0.9244 | 0.0252 | 0.1399 |
| window | embedding | 0.45 | 0.8683 | 0.9048 | 0.042 | 0.1818 |
| window | embedding | 0.50 | 0.7843 | 0.7927 | 0.1345 | 0.2587 |
