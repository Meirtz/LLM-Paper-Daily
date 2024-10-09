# LLM-Paper-Daily

## Awesome LLM Research

Welcome to the **Awesome LLM Research** repository! This project curates a list of high-quality resources related to LLM Research, including research papers, tools, libraries, and more.

## Daily Updates

### 2024-10-09

### 1. [Title:
          TIS-DPO: Token-level Importance Sampling for Direct Preference Optimization With Estimated Weights](https://arxiv.org/pdf/2410.04350)
**Summary**: The paper introduces TIS-DPO, a novel approach to Direct Preference Optimization (DPO) that addresses the issue of token importance in preference alignment for Large Language Models (LLMs). By using token-level importance sampling and estimating token weights through contrastive LLMs, TIS-DPO significantly improves performance on tasks related to harmlessness, helpfulness, and summarization, outperforming existing methods.

### 2. [Title:
          ReTok: Replacing Tokenizer to Enhance Representation Efficiency in Large Language Model](https://arxiv.org/pdf/2410.04335)
**Summary**: The paper introduces ReTok, a method to enhance the efficiency of large language models (LLMs) by replacing tokenizers. By reinitializing the input and output layers with the original model's parameters and training them while keeping other parameters fixed, ReTok maintains model performance while significantly improving decoding speed for long texts across various LLMs.

### 3. [Title:
          Inference Scaling for Long-Context Retrieval Augmented Generation](https://arxiv.org/pdf/2410.04343)
**Summary**: The paper investigates how scaling inference computation in retrieval augmented generation (RAG) can enhance the performance of long-context large language models (LLMs) by focusing on in-context learning and iterative prompting. The study finds that optimal allocation of inference computation leads to nearly linear improvements in RAG performance, and develops a model to predict the best inference parameters for given computation budgets, achieving up to 58.9% gains on benchmark datasets compared to standard RAG.

### 4. [Title:
          Blocks Architecture (BloArk): Efficient, Cost-Effective, and Incremental Dataset Architecture for Wikipedia Revision History](https://arxiv.org/pdf/2410.04410)
**Summary**: The paper introduces Blocks Architecture (BloArk), a novel data processing framework designed to efficiently handle Wikipedia Revision History (WikiRevHist) datasets. BloArk reduces computational resource demands and processing time by converting WikiRevHist data from XML to JSON Lines format and enabling incremental modifications to existing datasets. The architecture is scalable and open-source, facilitating downstream NLP applications.

### 5. [Title:
          Lens: Rethinking Multilingual Enhancement for Large Language Models](https://arxiv.org/pdf/2410.04407)
**Summary**: The paper introduces Lens, a novel approach to enhance the multilingual capabilities of large language models (LLMs) by manipulating their internal language representation spaces. By drawing target languages closer to a central language in the language-agnostic subspace and pushing them apart in the language-specific subspace, Lens improves multilingual performance without compromising the original language capabilities, outperforming existing post-training methods with less computational resources.

### 6. [Title:
          Ordinal Preference Optimization: Aligning Human Preferences via NDCG](https://arxiv.org/pdf/2410.04346)
**Summary**: The paper introduces Ordinal Preference Optimization (OPO), a novel listwise approach that leverages the Normalized Discounted Cumulative Gain (NDCG) to better utilize the ranking information in multiple responses for aligning Large Language Models (LLMs) with human preferences. By approximating NDCG with a differentiable surrogate loss, OPO outperforms existing pairwise and listwise methods in aligning multi-response datasets, demonstrating improved performance on benchmarks like AlpacaEval. The study also highlights the benefits of increasing the pool of

### 7. [Title:
          CopyLens: Dynamically Flagging Copyrighted Sub-Dataset Contributions to LLM Outputs](https://arxiv.org/pdf/2410.04454)
**Summary**: The paper introduces CopyLens, a framework designed to dynamically flag copyrighted sub-dataset contributions in Large Language Model (LLM) outputs. It employs a two-stage approach, combining token representations and a lightweight LSTM-based network, to analyze dataset contributions and enhance copyright detection. The framework demonstrates significant improvements in efficiency and accuracy, outperforming existing methods in experiments.

### 8. [Title:
          SWEb: A Large Web Dataset for the Scandinavian Languages](https://arxiv.org/pdf/2410.04456)
**Summary**: The paper introduces SWEb, the largest pretraining dataset for Scandinavian languages, containing over one trillion tokens. It describes a novel model-based text extractor for data collection and processing, which simplifies the process compared to traditional rule-based methods. Additionally, the authors introduce a new Swedish cloze-style benchmark for evaluating language models, showing competitive performance of models trained on SWEb compared to those trained on FineWeb.

### 9. [Title:
          MindScope: Exploring cognitive biases in large language models through Multi-Agent Systems](https://arxiv.org/pdf/2410.04452)
**Summary**: The paper introduces 'MindScope,' a comprehensive dataset designed to detect cognitive biases in large language models (LLMs). It combines static questions across 72 bias categories with a dynamic multi-agent framework for generating dialogues, enhancing detection capabilities. The proposed multi-agent detection method, integrating RAG, competitive debate, and reinforcement learning, significantly improves accuracy by 35.10% compared to GPT-4.

### 10. [Title:
          Hyper-multi-step: The Truth Behind Difficult Long-context Tasks](https://arxiv.org/pdf/2410.04422)
**Summary**: The paper investigates the challenges faced by long-context language models (LCLMs) in completing difficult long-context tasks, identifying "multi-matching retrieval" and "logic-based retrieval" as the primary sources of difficulty. These issues are found to be hyper-multi-step in nature, requiring numerous steps to solve, which explains why even advanced LCLMs struggle with these tasks and suggests a new perspective for developing solutions.

### 11. [Title:
          Collapsed Language Models Promote Fairness](https://arxiv.org/pdf/2410.04472)
**Summary**: The paper investigates the relationship between Neural Collapse and fairness in language models, finding that debiased models exhibit collapsed alignment between token representations and word embeddings. This insight leads to a new fine-tuning method that enhances fairness across various debiasing techniques while maintaining performance on standard language understanding tasks.

### 12. [Title:
          Fine-Grained Prediction of Reading Comprehension from Eye Movements](https://arxiv.org/pdf/2410.04484)
**Summary**: The paper investigates whether reading comprehension can be predicted from eye movements during reading, focusing on the fine-grained task of predicting comprehension at the level of individual questions over a passage. The authors introduce three new multimodal language models and compare them with existing models, evaluating their performance across different reading conditions and participant groups. The findings indicate that while challenging, eye movements provide valuable signals for fine-grained comprehension prediction.

### 13. [Title:
          Revisiting In-context Learning Inference Circuit in Large Language Models](https://arxiv.org/pdf/2410.04468)
**Summary**: The paper introduces a comprehensive circuit to model the inference dynamics of In-context Learning (ICL) in large language models, breaking down the process into three major operations: summarization, semantics merging, and feature retrieval/copy. The proposed circuit effectively captures observed ICL phenomena and demonstrates its critical role through ablation analysis, while also identifying parallel bypass mechanisms that contribute to ICL performance.

### 14. [Title:
          DAdEE: Unsupervised Domain Adaptation in Early Exit PLMs](https://arxiv.org/pdf/2410.04424)
**Summary**: The paper introduces DAdEE, an unsupervised domain adaptation framework for early exit Pre-trained Language Models (PLMs) that addresses the issue of domain sensitivity in exit classifiers. By employing multi-level adaptation through knowledge distillation and GAN-based adversarial adaptation, DAdEE reduces the domain gap and enhances inference speed while improving domain adaptation performance across various tasks.

### 15. [Title:
          Wrong-of-Thought: An Integrated Reasoning Framework with Multi-Perspective Verification and Wrong Information](https://arxiv.org/pdf/2410.04463)
**Summary**: The paper introduces Wrong-of-Thought (WoT), a novel reasoning framework designed to improve the performance of Large Language Models (LLMs) by addressing two key issues: reliance on single verification methods and ignorance of wrong information. WoT incorporates multi-perspective verification to refine reasoning processes and utilizes wrong information to prevent recurring errors. Experimental results across multiple datasets and LLMs show that WoT outperforms existing methods, particularly in challenging computational tasks.

### 16. [Title:
          RevMUX: Data Multiplexing with Reversible Adapters for Efficient LLM Batch Inference](https://arxiv.org/pdf/2410.04519)
**Summary**: The paper introduces RevMUX, a novel data multiplexing framework for efficient batch inference in large language models (LLMs). By incorporating reversible adapters in the multiplexer, RevMUX allows for the efficient merging and separation of multiple inputs, enabling higher throughput without significant performance degradation. Experiments across various datasets and LLM backbones show that RevMUX enhances inference efficiency while maintaining classification accuracy.

### 17. [Title:
          DAMRO: Dive into the Attention Mechanism of LVLM to Reduce Object Hallucination](https://arxiv.org/pdf/2410.04514)
**Summary**: The paper introduces DAMRO, a training-free strategy to reduce object hallucination in Large Vision-Language Models (LVLMs) by analyzing and correcting the attention distribution of the LLM decoder. DAMRO uses the classification token (CLS) of ViT to filter out high-attention outlier tokens in the background, thereby improving the model's focus on relevant objects and reducing hallucination. Evaluations on various benchmarks show significant improvements in alleviating hallucination in LVLMs.

### 18. [Title:
          How Does the Disclosure of AI Assistance Affect the Perceptions of Writing?](https://arxiv.org/pdf/2410.04545)
**Summary**: The study investigates how disclosing the use of AI assistance in writing affects perceptions of writing quality. Results indicate that revealing AI involvement, particularly in content generation, lowers average quality ratings and increases variability in individual evaluations. Factors like writing confidence and AI familiarity moderate these effects, and disclosure may reduce the proportion of AI-assisted writings ranked highly.

### 19. [Title:
          ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](https://arxiv.org/pdf/2410.04509)
**Summary**: The paper introduces ErrorRadar, a novel benchmark for evaluating the complex mathematical reasoning capabilities of Multimodal Large Language Models (MLLMs) through error detection tasks. ErrorRadar assesses models' abilities to identify and categorize errors in mathematical problem-solving, using a dataset of 2,500 high-quality K-12 problems with detailed annotations. Despite the best performance of models like GPT-4o, significant gaps remain compared to human evaluators, highlighting ongoing challenges in

### 20. [Title:
          Leveraging Large Language Models for Suicide Detection on Social Media with Limited Labels](https://arxiv.org/pdf/2410.04501)
**Summary**: The paper introduces a novel approach for detecting suicidal content on social media using Large Language Models (LLMs), leveraging pseudo-labels generated by prompting LLMs and fine-tuning techniques. An ensemble model combining Qwen2-72B-Instruct, Llama3-8B, Llama3.1-8B, and Gemma2-9B significantly improves detection accuracy, achieving F1 scores of 0.770 and 0.731 on public and

### 21. [Title:
          Towards Secure Tuning: Mitigating Security Risks Arising from Benign Instruction Fine-Tuning](https://arxiv.org/pdf/2410.04524)
**Summary**: The paper investigates the security risks associated with Instruction Fine-Tuning (IFT) of Large Language Models (LLMs), even when the tuning instructions are benign. The authors propose a novel Modular Layer-wise Learning Rate (ML-LR) strategy to mitigate these risks by analyzing module robustness and applying differentiated learning rates to robust modules. Experimental results demonstrate that this approach effectively reduces the harmfulness of LLMs post-IFT without compromising their usability or expertise.

### 22. [Title:
          LRQ-Fact: LLM-Generated Relevant Questions for Multimodal Fact-Checking](https://arxiv.org/pdf/2410.04616)
**Summary**: The paper introduces LRQ-Fact, an automated framework for multimodal fact-checking that uses Vision-Language Models and Large Language Models to generate relevant questions and answers for assessing the veracity of content. A rule-based decision-maker then evaluates the generated information to determine accuracy. Experiments demonstrate improved detection accuracy for multimodal misinformation and highlight the framework's generalizability across different model architectures.

### 23. [Title:
          Punctuation Prediction for Polish Texts using Transformers](https://arxiv.org/pdf/2410.04621)
**Summary**: The paper presents a solution for the Poleval 2022 Task 1 on punctuation prediction for Polish texts, achieving a Weighted F1 score of 71.44. The approach involves fine-tuning a single HerBERT model on both competition data and an external dataset to enhance punctuation accuracy in speech recognition outputs.

### 24. [Title:
          Upsample or Upweight? Balanced Training on Heavily Imbalanced Datasets](https://arxiv.org/pdf/2410.04579)
**Summary**: The paper investigates the equivalence of upsampling (Temperature Sampling) and upweighting (Scalarization) in training language models on imbalanced datasets, particularly in multilingual settings. It finds that while these methods are theoretically equivalent under full gradient descent, they diverge in practice with stochastic gradient descent, where upsampling converges faster but risks overfitting. The paper introduces Cooldown, a method that adjusts sampling temperature during training to balance convergence speed and overfitting, demonstrating competitive performance and computational efficiency.

### 25. [Title:
          FAMMA: A Benchmark for Financial Domain Multilingual Multimodal Question Answering](https://arxiv.org/pdf/2410.04526)
**Summary**: The paper introduces FAMMA, an open-source benchmark for financial multilingual multimodal question answering, designed to evaluate the performance of multimodal large language models (MLLMs) in complex financial reasoning tasks. The benchmark includes 1,758 question-answer pairs from university textbooks and exams, covering 8 major finance subfields and presented in mixed text and image formats. Despite testing advanced models like GPT-4o and Claude-35-Sonnet, FAMMA reveals significant

### 26. [Title:
          Reasoning-Enhanced Healthcare Predictions with Knowledge Graph Community Retrieval](https://arxiv.org/pdf/2410.04585)
**Summary**: The paper introduces KARE, a framework that integrates knowledge graph (KG) community-level retrieval with large language model (LLM) reasoning to improve healthcare predictions. KARE constructs a comprehensive multi-source KG and uses hierarchical community detection for precise information retrieval, enhancing prediction accuracy by up to 15.0% on MIMIC datasets. The framework also leverages LLM reasoning to make clinical predictions more interpretable and trustworthy.

### 27. [Title:
          Passage Retrieval of Polish Texts Using OKAPI BM25 and an Ensemble of Cross Encoders](https://arxiv.org/pdf/2410.04620)
**Summary**: The paper introduces a winning solution for the Poleval 2023 Task 3: Passage Retrieval challenge, combining OKAPI BM25 for document retrieval with an ensemble of multilingual Cross Encoders for reranking. While fine-tuning the reranker models improved performance in the trivia domain, it led to worse results in the legal and customer support domains, highlighting the challenges of domain adaptation in neural models.

### 28. [Title:
          ProtocoLLM: Automatic Evaluation Framework of LLMs on Domain-Specific Scientific Protocol Formulation Tasks](https://arxiv.org/pdf/2410.04601)
**Summary**: The paper introduces ProtocoLLM, an automatic evaluation framework for assessing the capabilities of Large Language Models (LLMs) in generating scientific protocols for domain-specific tasks. By using GPT-4 to generate pseudocode as a baseline and Llama-3 as an evaluator, the framework evaluates various LLMs, finding that GPT and Cohere excel in protocol formulation. Additionally, the authors present BIOPROT 2.0, a dataset to support LLM training and evaluation in this

### 29. [Title:
          Evaluation of Code LLMs on Geospatial Code Generation](https://arxiv.org/pdf/2410.04617)
**Summary**: The paper introduces a benchmark for evaluating Large Language Models (LLMs) on geospatial code generation tasks, addressing the unique challenges posed by this domain. The authors created a dataset of manually curated coding problems that test spatial reasoning, data processing, and tool usage, along with test scenarios for automated correctness checks. They also tested existing LLMs and made the dataset and evaluation code publicly available, aiming to facilitate the development of more accurate geospatial coding assistants.

### 30. [Title:
          Control Large Language Models via Divide and Conquer](https://arxiv.org/pdf/2410.04628)
**Summary**: The paper examines the limitations of large language models (LLMs) in satisfying lexical constraints through prompt-based control, identifying issues such as position bias, low responsiveness to decoding parameters, and difficulty with complex constraints. To address these challenges, the authors propose a Divide and Conquer Generation strategy, which significantly improves the success rate of lexical constrained generation tasks, offering a promising approach for more sophisticated text generation applications.

### 31. [Title:
          The LLM Effect: Are Humans Truly Using LLMs, or Are They Being Influenced By Them Instead?](https://arxiv.org/pdf/2410.04699)
**Summary**: The paper examines the effectiveness of Large Language Models (LLMs) in specialized analytical tasks by integrating them with human experts in a two-stage study. While LLMs show significant overlap with human-generated topic lists and improve task completion speed, they also introduce anchoring bias, potentially compromising the depth and nuance of the analysis. This raises concerns about the trade-off between efficiency and the risk of biased outcomes.

### 32. [Title:
          Adversarial Multi-Agent Evaluation of Large Language Models through Iterative Debates](https://arxiv.org/pdf/2410.04663)
**Summary**: The paper introduces a novel framework for evaluating large language models (LLMs) by treating them as advocates in a multi-agent system, where they defend their outputs through iterative debates judged by other LLMs. This approach aims to provide a more dynamic and comprehensive evaluation compared to traditional methods, and the authors present a probabilistic model to measure error reduction in such systems. The paper also outlines experiments to validate the effectiveness of this multi-advocate architecture and suggests future research directions.

### 33. [Title:
          Contrastive Learning to Improve Retrieval for Real-world Fact Checking](https://arxiv.org/pdf/2410.04657)
**Summary**: The paper introduces Contrastive Fact-Checking Reranker (CFR), a novel retriever designed to improve evidence retrieval for complex fact-checking tasks by leveraging contrastive learning and fine-tuning on the AVeriTeC dataset. CFR enhances retrieval accuracy by considering indirect relevance and multiple training signals, leading to a 6% improvement in veracity classification accuracy on the AVeriTeC dataset and demonstrating transferable gains to other datasets.

### 34. [Title:
          $\textbf{Only-IF}$:Revealing the Decisive Effect of Instruction Diversity on Generalization](https://arxiv.org/pdf/2410.04717)
**Summary**: The paper investigates the importance of instruction diversity in training large language models (LLMs) to generalize effectively to unseen tasks. It finds that generalization only emerges when training data spans multiple semantic domains, and that cross-domain diversification, even with limited data, significantly enhances a model's adaptability. The study emphasizes the critical role of strategic data diversification in improving both specialist and generalist models' performance.

### 35. [Title:
          Efficient transformer with reinforced position embedding for language models](https://arxiv.org/pdf/2410.04731)
**Summary**: The paper introduces an efficient transformer architecture that enhances performance by reinforcing positional embedding, achieving superior results with fewer encoder-decoder layers. By concatenating positional encoding with trainable token embeddings and normalizing the token embedding matrix, the method significantly reduces training and validation losses, and training time, outperforming a baseline model in Portuguese-English translation tasks across multiple datasets.

### 36. [Title:
          Rule-based Data Selection for Large Language Models](https://arxiv.org/pdf/2410.04715)
**Summary**: The paper introduces a novel rule-based framework for selecting high-quality training data for large language models (LLMs), using the orthogonality of score vectors as a metric for rule evaluation. The framework employs an automated pipeline that generates diverse rules, rates data based on these rules, and selects the most orthogonal score vectors using the determinantal point process (DPP). Experimental results show that this method consistently outperforms other approaches in terms of rating precision and model performance across various tasks and domains.

### 37. [Title:
          MathHay: An Automated Benchmark for Long-Context Mathematical Reasoning in LLMs](https://arxiv.org/pdf/2410.04698)
**Summary**: The paper introduces MathHay, an automated benchmark for evaluating the long-context mathematical reasoning abilities of large language models (LLMs). Unlike previous benchmarks that focus on information retrieval, MathHay requires models to perform complex mathematical reasoning over extended texts. Experiments on eight top-performing LLMs reveal that even the best model, Gemini-1.5-Pro-002, achieves only 51.26% accuracy at 128K tokens, indicating substantial room for improvement in this

### 38. [Title:
          Formality is Favored: Unraveling the Learning Preferences of Large Language Models on Data with Conflicting Knowledge](https://arxiv.org/pdf/2410.04784)
**Summary**: The study investigates how large language models (LLMs) handle conflicting information in training data, finding that they prefer formal texts and those with fewer spelling errors, similar to human preferences. This preference leads to faster learning and better retention of knowledge, especially in larger models, and can be influenced by manipulating data consistency.

### 39. [Title:
          DAPE V2: Process Attention Score as Feature Map for Length Extrapolation](https://arxiv.org/pdf/2410.04798)
**Summary**: The paper introduces a novel approach to improving Transformer models by treating attention scores as feature maps and applying convolution operations to enhance their expressiveness. This method addresses the limitations of the traditional key-query dot product in handling length extrapolation, translating the problem into a feature map processing issue. Experimental results show significant performance improvements, suggesting potential for further advancements in Transformer architectures.

### 40. [Title:
          Representing the Under-Represented: Cultural and Core Capability Benchmarks for Developing Thai Large Language Models](https://arxiv.org/pdf/2410.04795)
**Summary**: The paper introduces two benchmarks, Thai-H6 and Thai Cultural and Linguistic Intelligence Benchmark (ThaiCLI), to address the lack of evaluation frameworks for Thai large language models (LLMs). These benchmarks aim to enhance both the core capabilities and cultural understanding of Thai LLMs, providing a comprehensive evaluation tool for researchers and developers. The datasets and evaluation code will be made publicly available to support further research in this area.

### 41. [Title:
          Forgetting Curve: A Reliable Method for Evaluating Memorization Capability for Long-context Models](https://arxiv.org/pdf/2410.04727)
**Summary**: The paper identifies limitations in current methods for evaluating the memorization capabilities of long-context language models and introduces a new method called the "forgetting curve." This method is robust, independent of prompts, and applicable to various model sizes and architectures, providing empirical evidence on the effectiveness of transformer extensions and questioning the effective length of RNN/SSM models.

### 42. [Title:
          TableRAG: Million-Token Table Understanding with Language Models](https://arxiv.org/pdf/2410.04739)
**Summary**: The paper introduces TableRAG, a Retrieval-Augmented Generation framework designed to improve language models' ability to understand large tables by efficiently retrieving and encoding crucial information, thereby reducing prompt lengths and enhancing scalability. The authors developed new million-token benchmarks from Arcade and BIRD-SQL datasets, showing that TableRAG achieves state-of-the-art performance in large-scale table understanding by significantly improving retrieval quality.

### 43. [Title:
          GARLIC: LLM-Guided Dynamic Progress Control with Hierarchical Weighted Graph for Long Document QA](https://arxiv.org/pdf/2410.04790)
**Summary**: The paper introduces GARLIC, a novel retrieval method for long document QA that constructs a Hierarchical Weighted Directed Acyclic Graph and leverages LLM attention weights for dynamic retrieval. GARLIC outperforms state-of-the-art baselines, including Llama 3.1, on multiple QA datasets while maintaining computational efficiency.

### 44. [Title:
          Document-level Causal Relation Extraction with Knowledge-guided Binary Question Answering](https://arxiv.org/pdf/2410.04752)
**Summary**: The paper introduces a Knowledge-guided binary Question Answering (KnowQA) method for Event-Event Causal Relation Extraction (ECRE), addressing challenges like lack of document-level modeling and causal hallucinations. The proposed method, involving Event Structure Construction and Binary Question Answering, achieves state-of-the-art performance on the MECI dataset and shows high generalizability and low inconsistency, especially with complete event structures post-fine-tuning.

### 45. [Title:
          LPZero: Language Model Zero-cost Proxy Search from Zero](https://arxiv.org/pdf/2410.04808)
**Summary**: The paper introduces LPZero, a novel framework for automatically designing Zero-cost (ZC) proxies in Neural Architecture Search (NAS) that significantly reduces computational demands. Unlike existing ZC proxies, LPZero uses genetic programming to find optimal symbolic compositions, outperforming human-designed proxies in ranking consistency and downstream task performance across models like FlexiBERT, GPT-2, and LLaMA-7B.

### 46. [Title:
          MINER: Mining the Underlying Pattern of Modality-Specific Neurons in Multimodal Large Language Models](https://arxiv.org/pdf/2410.04819)
**Summary**: The paper introduces MINER, a framework for identifying modality-specific neurons (MSNs) in multimodal large language models (MLLMs), addressing the lack of explainability in these models. The framework consists of four stages and demonstrates that deactivating a small percentage of MSNs significantly impacts model performance, indicating the importance of these neurons in processing multimodal data.

### 47. [Title:
          As Simple as Fine-tuning: LLM Alignment via Bidirectional Negative Feedback Loss](https://arxiv.org/pdf/2410.04834)
**Summary**: The paper introduces a novel loss function called Bidirectional Negative Feedback (BNF) for aligning large language models (LLMs), which addresses the instability and hyperparameter sensitivity issues of Direct Preference Optimization (DPO). BNF simplifies the alignment process by eliminating the need for pairwise contrastive losses and extra hyperparameters, achieving comparable performance on QA benchmarks and better balance between value alignment and reasoning ability on reasoning benchmarks.

### 48. [Title:
          Intent Classification for Bank Chatbots through LLM Fine-Tuning](https://arxiv.org/pdf/2410.04925)
**Summary**: The study investigates the use of large language models (LLMs) for intent classification in banking chatbots, comparing SlovakBERT with multilingual models like Llama 8b instruct and Gemma 7b instruct. The results show that SlovakBERT achieves superior performance in terms of accuracy and false positive rates, making it the preferred model for this application.

### 49. [Title:
          SkillMatch: Evaluating Self-supervised Learning of Skill Relatedness](https://arxiv.org/pdf/2410.05006)
**Summary**: The paper introduces SkillMatch, a benchmark for evaluating skill relatedness in human resources, constructed from expert knowledge mined from millions of job ads. The authors also propose a self-supervised learning approach using Sentence-BERT adapted for skill co-occurrence, significantly outperforming traditional models. The release of SkillMatch aims to advance research in skill-based recommendation systems.

### 50. [Title:
          Rationale-Aware Answer Verification by Pairwise Self-Evaluation](https://arxiv.org/pdf/2410.04838)
**Summary**: The paper introduces REPS, a method for improving answer verification by focusing on the validity of rationales in addition to the correctness of final answers. REPS uses pairwise self-evaluation to select valid rationales from LLM-generated candidates, leading to verifiers that outperform traditional methods on reasoning benchmarks. The study highlights the importance of rationale validity in training reliable verifiers for complex reasoning tasks.

### 51. [Title:
          Leveraging Grammar Induction for Language Understanding and Generation](https://arxiv.org/pdf/2410.04878)
**Summary**: The paper introduces an unsupervised grammar induction method for enhancing language understanding and generation tasks. By constructing a grammar parser that induces constituency structures and dependency relations, the authors integrate these features into the Transformer model as a syntactic mask. This approach outperforms the original Transformer and other models with external parsers across various tasks, demonstrating the effectiveness of explicitly modeling grammatical structure in neural networks.

### 52. [Title:
          Named Clinical Entity Recognition Benchmark](https://arxiv.org/pdf/2410.05046)
**Summary**: The paper introduces a Named Clinical Entity Recognition Benchmark, a standardized platform for evaluating language models in healthcare. It uses curated clinical datasets and assesses models on their ability to identify and classify entities like diseases and medications, with performance measured by the F1-score. The benchmark aims to promote transparency and innovation in clinical NLP by ensuring consistency and interoperability across healthcare systems.

### 53. [Title:
          ZEBRA: Zero-Shot Example-Based Retrieval Augmentation for Commonsense Question Answering](https://arxiv.org/pdf/2410.05077)
**Summary**: The paper introduces ZEBRA, a zero-shot question answering framework that enhances commonsense reasoning by combining retrieval, case-based reasoning, and introspection without requiring additional training of the language model. ZEBRA outperforms existing methods and strong language models across multiple benchmarks, achieving an average accuracy improvement of up to 4.5 points.

### 54. [Title:
          Activation Scaling for Steering and Interpreting Language Models](https://arxiv.org/pdf/2410.04962)
**Summary**: The paper explores the concept of steering language models by scaling activation vectors to correct incorrect predictions, such as flipping "Rome is in France" to "Rome is in Italy." The authors propose a three-term objective to ensure interventions are effective, faithful, and minimal. They demonstrate that activation scaling is comparable to steering vectors in performance but offers greater interpretability and sparsity, allowing for more precise identification of model components.

### 55. [Title:
          Initialization of Large Language Models via Reparameterization to Mitigate Loss Spikes](https://arxiv.org/pdf/2410.05052)
**Summary**: The paper addresses loss spikes in large language model pre-training by proposing a novel technique called weight scaling as reparameterization (WeSaR). This method introduces gate parameters to adjust the norm of model parameters uniformly, leading to more stable and accelerated training. Experimental results demonstrate that WeSaR outperforms other initialization methods across various model sizes.

### 56. [Title:
          Explanation sensitivity to the randomness of large language models: the case of journalistic text classification](https://arxiv.org/pdf/2410.05085)
**Summary**: The paper investigates the impact of random elements in the training of large language models (LLMs) on the explainability of their predictions, focusing on French journalistic text classification. It finds that different random seeds yield models with similar accuracy but varying explanations, suggesting the need to characterize the statistical distribution of explanations. The study also explores a simpler model that offers stable explanations but lower accuracy, and demonstrates that incorporating features from LLM explanations can improve this simpler model.

### 57. [Title:
          A test suite of prompt injection attacks for LLM-based machine translation](https://arxiv.org/pdf/2410.05047)
**Summary**: The paper introduces a comprehensive test suite for evaluating prompt injection attacks (PIAs) on LLM-based machine translation systems, extending previous work by Sun and Miceli-Barone. The suite includes attacks across all language pairs in the WMT 2024 General Machine Translation task and incorporates additional attack formats to assess the robustness of these systems against malicious input interference.

### 58. [Title:
          Investigating large language models for their competence in extracting grammatically sound sentences from transcribed noisy utterances](https://arxiv.org/pdf/2410.05099)
**Summary**: The study investigates whether large language models (LLMs) can effectively extract grammatically sound sentences from noisy transcribed dialogues, mimicking human cognitive abilities to separate meaningful content from speech-specific noise. The experiments, conducted in Polish, reveal that while LLMs can extract some well-structured utterances, many are not correctly formed, suggesting that LLMs either do not fully acquire or cannot effectively apply syntactic-semantic rules, indicating a superficial comprehension compared to human capabilities.

### 59. [Title:
          SparsePO: Controlling Preference Alignment of LLMs via Sparse Token Masks](https://arxiv.org/pdf/2410.05102)
**Summary**: The paper introduces SparsePO, a novel approach to preference optimization (PO) for language models that focuses on weighting tokens differently based on their relevance to human preferences. By learning sparse weight masks, SparsePO allows the model to prioritize certain tokens during training, leading to improved performance in tasks such as sentiment control, dialogue, summarization, and text-to-code generation. The method demonstrates better alignment with human preferences and enhances reasoning capabilities compared to existing PO methods.

### 60. [Title:
          ScienceAgentBench: Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery](https://arxiv.org/pdf/2410.05080)
**Summary**: The paper introduces ScienceAgentBench, a benchmark designed to rigorously assess the capabilities of language agents in automating data-driven scientific discovery. It includes 102 tasks from 44 peer-reviewed publications across four disciplines, validated by experts, and evaluates the performance of five LLMs using various frameworks. The results show that even the best-performing agents can only solve a fraction of the tasks independently, highlighting the current limitations of language agents in fully automating scientific workflows.

### 61. [Title:
          CTC-GMM: CTC guided modality matching for fast and accurate streaming speech translation](https://arxiv.org/pdf/2410.05146)
**Summary**: The paper introduces CTC-GMM, a method that enhances streaming speech translation (ST) by using Connectionist Temporal Classification (CTC) to align speech and text sequences, allowing the incorporation of machine translation (MT) data to improve accuracy. Evaluations on FLEURS and CoVoST2 datasets show significant improvements in translation accuracy and decoding speed.

### 62. [Title:
          Deciphering the Interplay of Parametric and Non-parametric Memory in Retrieval-augmented Language Models](https://arxiv.org/pdf/2410.05162)
**Summary**: The study investigates how Retrieval-Augmented Generation (RAG) models like \textsc{Atlas} balance parametric (internal) and non-parametric (retrieved) memory during information processing. Through causal mediation analysis, it reveals that the model prioritizes retrieved context over internal knowledge when both are available. The analysis also identifies two key mechanisms within the model: determining context relevance and computing output representations for copying relevant information.

### 63. [Title:
          Enhancing Equity in Large Language Models for Medical Applications](https://arxiv.org/pdf/2410.05180)
**Summary**: The paper discusses the potential biases in large language models (LLMs) used for medical applications, particularly affecting racial, gender, and underrepresented groups. To address these inequities, the authors introduce EquityGuard, a framework that detects and mitigates biases in LLM-based medical tools, thereby promoting more equitable outcomes in healthcare.

### 64. [Title:
          ReasoningRank: Teaching Student Models to Rank through Reasoning-Based Knowledge Distillation](https://arxiv.org/pdf/2410.05168)
**Summary**: The paper introduces ReasoningRank, a reranking approach that uses large language models to generate explicit and comparison reasoning for document relevance, then distills this knowledge into smaller student models. These student models, while not as fast as the LLMs, are more resource-efficient and achieve competitive reranking performance, enhancing interpretability and accuracy in information retrieval tasks.

### 65. [Title:
          RevisEval: Improving LLM-as-a-Judge via Response-Adapted References](https://arxiv.org/pdf/2410.05193)
**Summary**: The paper introduces RevisEval, a new evaluation method for text generation that uses response-adapted references generated by large language models (LLMs) to improve the reliability of LLM-as-a-Judge assessments. By leveraging LLM's text revision capabilities, RevisEval adapts the response into a reference that is more relevant for evaluation, outperforming traditional reference-free and reference-based methods across various tasks. This approach not only enhances classical text metrics like BLEU and BERTScore

### 66. [Title:
          Beyond Correlation: Interpretable Evaluation of Machine Translation Metrics](https://arxiv.org/pdf/2410.05183)
**Summary**: The paper introduces an interpretable evaluation framework for machine translation (MT) metrics, moving beyond traditional correlation-based assessments to provide clearer insights into metric performance, particularly for data filtering and translation re-ranking use cases. The authors use Precision, Recall, and F-score to evaluate MT metrics, highlighting concerns about the reliability of manually curated data and low agreement with MQM annotations.

### 67. [Title:
          SFTMix: Elevating Language Model Instruction Tuning with Mixup Recipe](https://arxiv.org/pdf/2410.05248)
**Summary**: The paper introduces SFTMix, a novel approach to improve instruction-tuning in large language models (LLMs) by leveraging Mixup-based regularization to address uneven confidence levels in the semantic representation space. Unlike traditional methods that rely on high-quality, curated datasets, SFTMix enhances performance across various tasks without the need for extensive data filtering, demonstrating scalability and adaptability across different LLM families and datasets.

### 68. [Title:
          Cookbook: A framework for improving LLM generative abilities via programmatic data generating templates](https://arxiv.org/pdf/2410.05224)
**Summary**: The paper introduces Cookbook, a framework for programmatically generating training data to improve the generative capabilities of large language models (LLMs). By using simple patterns over random tokens, Cookbook creates scalable and cost-effective datasets that avoid legal and privacy issues. The framework demonstrates significant performance improvements on various tasks, with Mistral-7B fine-tuned using Cookbook-generated data achieving the highest accuracy on multiple benchmarks.

### 69. [Title:
          Differential Transformer](https://arxiv.org/pdf/2410.05258)
**Summary**: The paper introduces Diff Transformer, a novel architecture that enhances the attention mechanism by amplifying relevant context while reducing noise through a differential attention approach. This method outperforms traditional Transformers in various language modeling tasks, particularly in long-context modeling, key information retrieval, and hallucination mitigation. Diff Transformer also improves robustness in in-context learning and reduces activation outliers, positioning it as a promising advancement in large language models.

### 70. [Title:
          CasiMedicos-Arg: A Medical Question Answering Dataset Annotated with Explanatory Argumentative Structures](https://arxiv.org/pdf/2410.05235)
**Summary**: The paper introduces CasiMedicos-Arg, the first multilingual dataset for Medical Question Answering that includes natural language explanations from doctors, annotated with argumentative structures such as premises, claims, and relations like support and attack. The dataset, comprising 558 clinical cases in four languages, aims to aid in training medical residents' explanation skills and evaluating AI models in understanding and generating argumentative medical explanations.

### 71. [Title:
          Causal Micro-Narratives](https://arxiv.org/pdf/2410.05252)
**Summary**: The paper introduces a method for classifying causal micro-narratives from text using only a subject-specific ontology of causes and effects. Applied to inflation narratives, the approach leverages a human-annotated dataset and evaluates several large language models, with the fine-tuned Llama 3.1 8B achieving high F1 scores. The research highlights linguistic ambiguity as a significant challenge and suggests the framework has broad applications in social science.

### 72. [Title:
          Data Advisor: Dynamic Data Curation for Safety Alignment of Large Language Models](https://arxiv.org/pdf/2410.05269)
**Summary**: The paper introduces Data Advisor, a dynamic data curation method for improving the safety alignment of large language models (LLMs). By monitoring and analyzing the generated data, Data Advisor identifies deficiencies and guides subsequent data generation to enhance quality and coverage. Experiments show that Data Advisor effectively improves model safety across multiple LLMs without compromising utility.

### 73. [Title:
          TurtleBench: Evaluating Top Language Models via Real-World Yes/No Puzzles](https://arxiv.org/pdf/2410.05262)
**Summary**: The paper introduces TurtleBench, a novel evaluation benchmark for Large Language Models (LLMs) that uses real-world user guesses from the Turtle Soup Puzzle platform to assess logical reasoning capabilities. Unlike static datasets, TurtleBench provides a dynamic and user-centric approach, enhancing the reliability of evaluations. The study finds that OpenAI's o1 series models did not perform exceptionally well, suggesting potential limitations in their reasoning strategies and the need for further research on Chain-of-Thought techniques.

### 74. [Title:
          GLEE: A Unified Framework and Benchmark for Language-based Economic Environments](https://arxiv.org/pdf/2410.05254)
**Summary**: The paper introduces GLEE, a unified framework and benchmark for studying the behavior of Large Language Models (LLMs) in language-based economic environments. It defines standardized games to evaluate LLMs' rationality, human-like behavior, and outcomes in terms of efficiency and fairness, addressing the challenges of comparing diverse studies. The framework includes datasets of LLM vs. LLM and human vs. LLM interactions, enabling analysis of agent performance and the impact of economic environment characteristics.

### 75. [Title:
          Grounding Partially-Defined Events in Multimodal Data](https://arxiv.org/pdf/2410.05267)
**Summary**: The paper introduces a multimodal approach to grounding partially-defined events in video and text data, addressing the challenges of understanding complex events from unstructured video snippets. It proposes a three-stage span retrieval task and a benchmark, MultiVENT-G, with densely annotated videos and text documents, to evaluate LLM-driven methods for multimodal event analysis. The results highlight the difficulties in abstract event understanding and show potential for improving event-centric video-language systems.

### 76. [Title:
          SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](https://arxiv.org/pdf/2410.03750)
**Summary**: The paper introduces SQFT, a method for low-precision sparse parameter-efficient fine-tuning of large pre-trained models (LPMs) in resource-constrained environments. SQFT enables the merging of sparse weights with low-rank adapters while maintaining sparsity and accuracy, addressing challenges related to different numerical precisions. The effectiveness of SQFT is demonstrated across various adaptation scenarios and sparsity levels.

### 77. [Title:
          Getting in the Door: Streamlining Intake in Civil Legal Services with Large Language Models](https://arxiv.org/pdf/2410.03762)
**Summary**: The paper explores the use of large language models (LLMs) to streamline the legal intake process for civil legal services, aiming to reduce the time and resources required to determine eligibility. By integrating logical rules with LLMs, the authors develop a digital platform that provides eligibility recommendations, achieving an F1 score of .82 with the best model, thereby potentially aiding in closing the access to justice gap.

### 78. [Title:
          Efficient Streaming LLM for Speech Recognition](https://arxiv.org/pdf/2410.03752)
**Summary**: The paper introduces SpeechLLM-XL, a decoder-only model designed for efficient streaming speech recognition, addressing the limitations of existing methods that struggle with long audio inputs and high computational costs. By processing audio in configurable chunks with limited attention, SpeechLLM-XL achieves linear scaling and maintains recognition accuracy, even for long utterances significantly longer than those seen during training.

### 79. [Title:
          FutureFill: Fast Generation from Convolutional Sequence Models](https://arxiv.org/pdf/2410.03766)
**Summary**: The paper introduces FutureFill, a method for accelerating auto-regressive generation in sequence prediction models using convolutional operators. It reduces generation time from linear to square root relative to context length and requires a smaller cache size compared to standard models. Experimental results confirm its efficiency and correctness.

### 80. [Title:
          Metadata Matters for Time Series: Informative Forecasting with Transformers](https://arxiv.org/pdf/2410.03806)
**Summary**: The paper introduces MetaTST, a novel approach that integrates metadata into Transformer models for time series forecasting, enhancing interpretability and accuracy. By converting unstructured metadata into structured text and encoding it with large language models, MetaTST enriches the model's embedding with contextual information, leading to superior performance in both short- and long-term forecasting benchmarks across diverse scenarios.

### 81. [Title:
          Can Mamba Always Enjoy the "Free Lunch"?](https://arxiv.org/pdf/2410.03810)
**Summary**: The paper investigates the theoretical limitations of Mamba, a model known for its constant-size overhead during inference, in comparison to Transformers. It finds that while Mamba performs well in many sequence modeling tasks, it may face bottlenecks in tasks requiring the COPY operation and in solving dynamic programming (DP) problems, especially when the sequence length scales. The study concludes that Mamba's efficiency gains are context-dependent and not universally applicable, thus questioning the notion of a "free lunch"

### 82. [Title:
          Variational Language Concepts for Interpreting Foundation Language Models](https://arxiv.org/pdf/2410.03964)
**Summary**: The paper introduces a new approach called VAriational Language Concept (VALC) to enhance the interpretability of Foundation Language Models (FLMs) by moving beyond word-level interpretations to concept-level interpretations. The authors propose a variational Bayesian framework that optimizes language concepts to better explain FLM predictions, demonstrating its effectiveness through empirical results on various datasets.

### 83. [Title:
          Enhancing Future Link Prediction in Quantum Computing Semantic Networks through LLM-Initiated Node Features](https://arxiv.org/pdf/2410.04251)
**Summary**: The paper explores enhancing link prediction in quantum computing semantic networks by using Large Language Models (LLMs) to initialize node features. This approach reduces the need for manual feature creation and improves the performance of link prediction models compared to traditional node embedding techniques.

### 84. [Title:
          Fundamental Limitations on Subquadratic Alternatives to Transformers](https://arxiv.org/pdf/2410.04271)
**Summary**: The paper demonstrates that any subquadratic alternative to the Transformer architecture, such as heuristic algorithms or models like Mamba, cannot perform certain important tasks, particularly document similarity tasks, that Transformers can handle. This implies that for tasks involving document similarity, the quadratic running time of Transformers cannot be avoided.

### 85. [Title:
          Language Model-Driven Data Pruning Enables Efficient Active Learning](https://arxiv.org/pdf/2410.04275)
**Summary**: The paper introduces ActivePrune, a novel data pruning strategy for active learning that uses language models to reduce the computational cost of selecting informative instances for annotation. By employing a two-stage pruning process and a perplexity reweighting method, ActivePrune enhances diversity and efficiency, outperforming existing methods in various tasks while significantly reducing the time required for active learning.

### 86. [Title:
          Hyperbolic Fine-tuning for Large Language Models](https://arxiv.org/pdf/2410.04010)
**Summary**: The paper explores the suitability of Euclidean space for embedding tokens in large language models (LLMs) and finds that token embeddings exhibit a high degree of hyperbolicity, suggesting a tree-like structure. To leverage this, the authors propose HypLoRA, a method for fine-tuning LLMs in hyperbolic space, which significantly improves performance on complex reasoning tasks, as evidenced by a 13.0% improvement on the AQuA dataset.

### 87. [Title:
          Harnessing Task Overload for Scalable Jailbreak Attacks on Large Language Models](https://arxiv.org/pdf/2410.04190)
**Summary**: The paper introduces a scalable jailbreak attack on Large Language Models (LLMs) that exploits resource constraints to bypass safety mechanisms. By engaging the LLM in a computationally intensive preliminary task, the attack saturates the model's processing capacity, preventing the activation of safety protocols when executing the target instruction. This method demonstrates high success rates across various LLMs and emphasizes the need for more robust safety measures that consider resource limitations.

### 88. [Title:
          Improving Arabic Multi-Label Emotion Classification using Stacked Embeddings and Hybrid Loss Function](https://arxiv.org/pdf/2410.03979)
**Summary**: The paper introduces a novel approach to improve Arabic multi-label emotion classification by combining stacked embeddings from fine-tuned language models (ArabicBERT, MarBERT, and AraBERT), a meta-learner, and a hybrid loss function. The hybrid loss function, which includes class weighting, label correlation matrix, and contrastive learning, effectively addresses class imbalances and enhances the model's performance, particularly in predicting minority emotions. The proposed method outperforms baseline approaches and demonstrates a more balanced classification across different emotions

### 89. [Title:
          OD-Stega: LLM-Based Near-Imperceptible Steganography via Optimized Distributions](https://arxiv.org/pdf/2410.04328)
**Summary**: The paper introduces OD-Stega, a method for near-imperceptible steganography using Large Language Models (LLMs) to generate stego-texts with minimal token usage. It optimizes the entropy of token replacement probabilities to ensure natural language fluency while embedding secret messages. The approach addresses tokenization mismatches and combines optimized distributions with vocabulary truncation and sequence-level heuristics for enhanced efficiency.

### 90. [Title:
          TUBench: Benchmarking Large Vision-Language Models on Trustworthiness with Unanswerable Questions](https://arxiv.org/pdf/2410.04107)
**Summary**: The paper introduces TUBench, a benchmark designed to evaluate the reliability of Large Vision-Language Models (LVLMs) using unanswerable questions, addressing the issue of hallucination in models that generate incorrect or unfaithful content. TUBench includes a diverse set of unanswerable questions across four domains—code, natural images, geometry, and statistical tables—to test LVLMs' trustworthiness in various reasoning tasks.

### 91. [Title:
          Latent Feature Mining for Predictive Model Enhancement with Large Language Models](https://arxiv.org/pdf/2410.04347)
**Summary**: The paper introduces FLAME, a framework that uses large language models to infer latent features from text data, thereby enhancing predictive models in domains with limited and ethically challenging data. The framework is validated in the criminal justice and healthcare domains, showing that the inferred latent features improve the performance of downstream classifiers.

### 92. [Title:
          Algorithmic Capabilities of Random Transformers](https://arxiv.org/pdf/2410.04368)
**Summary**: The paper investigates the algorithmic capabilities of randomly initialized transformers, focusing on tasks that can be learned by optimizing only the embedding layers. It finds that these random transformers can perform a variety of meaningful tasks, including arithmetic and associative recall, suggesting that some algorithmic capabilities are inherent in transformers even before training.

### 93. [Title:
          Suspiciousness of Adversarial Texts to Human](https://arxiv.org/pdf/2410.04377)
**Summary**: The paper investigates the concept of human suspiciousness in adversarial texts, which differs from imperceptibility in images as text must maintain semantic coherence while remaining undetected by human readers. The study introduces a novel dataset of human evaluations on the suspiciousness of adversarial sentences and develops a regression-based model to quantify and reduce this suspiciousness, providing a baseline for future research in adversarial text generation.

### 94. [Title:
          Learning Code Preference via Synthetic Evolution](https://arxiv.org/pdf/2410.03837)
**Summary**: The paper introduces CodeFavor, a framework for training pairwise code preference models using synthetic evolution data, and CodePrefBench, a benchmark for evaluating code preferences across correctness, efficiency, and security. The evaluation demonstrates that CodeFavor significantly improves model accuracy and cost-effectiveness compared to larger models, while highlighting the limitations and costs of human-based code preference assessments.

### 95. [Title:
          DOTS: Learning to Reason Dynamically in LLMs via Optimal Reasoning Trajectories Search](https://arxiv.org/pdf/2410.03864)
**Summary**: The paper introduces DOTS, a method that enables large language models (LLMs) to dynamically reason by searching for optimal reasoning trajectories tailored to each question and the LLM's capabilities. The approach involves defining atomic reasoning actions, searching for the best action sequences for training questions, and using these sequences to train the LLM to plan reasoning for new questions. Experiments demonstrate that DOTS outperforms static reasoning methods and improves LLMs' ability to adapt reasoning depth based on problem complexity.

### 96. [Title:
          Learning How Hard to Think: Input-Adaptive Allocation of LM Computation](https://arxiv.org/pdf/2410.04707)
**Summary**: The paper introduces a method for adaptively allocating computational resources during language model decoding, based on predicting the difficulty of generating accurate outputs for different inputs. By dynamically adjusting the amount of computation used, the approach reduces computational costs by up to 50% without compromising output quality or improves quality by up to 10% within a fixed budget, across various tasks like code generation, numerical reasoning, and dialog.

### 97. [Title:
          Realizing Video Summarization from the Path of Language-based Semantic Understanding](https://arxiv.org/pdf/2410.04511)
**Summary**: The paper introduces a novel video summarization framework that leverages the strengths of multiple Video-based Large Language Models (VideoLLMs) without requiring fine-tuning, inspired by the Mixture of Experts (MoE) paradigm. This approach integrates visual and audio content to generate comprehensive and coherent textual summaries, enhancing semantic understanding and performance in downstream tasks like summary video generation.

### 98. [Title:
          TLDR: Token-Level Detective Reward Model for Large Vision Language Models](https://arxiv.org/pdf/2410.04734)
**Summary**: The paper introduces a Token-Level Detective Reward Model (TLDR) to provide fine-grained annotations for each text token in multimodal language models, addressing the limitations of existing binary feedback systems. TLDR uses a perturbation-based method to generate synthetic hard negatives and their token-level labels, enhancing model performance and speeding up human annotation by three times.

### 99. [Title:
          Can LLMs plan paths with extra hints from solvers?](https://arxiv.org/pdf/2410.05045)
**Summary**: The paper investigates enhancing Large Language Models' (LLMs) planning capabilities in robotic tasks by integrating solver-generated feedback. Four feedback strategies, including visual feedback, are tested on three LLMs across 110 planning problems. Results show improved performance on moderately difficult tasks but limited success on harder problems, highlighting the impact of different hinting strategies and LLM planning tendencies.

### 100. [Title:
          ImProver: Agent-Based Automated Proof Optimization](https://arxiv.org/pdf/2410.04753)
**Summary**: The paper introduces ImProver, an agent-based system using large language models (LLMs) to optimize formal proofs in Lean by rewriting them to meet user-defined criteria such as length, readability, or modularity. ImProver incorporates improvements like the Chain-of-States technique and error-correction mechanisms, demonstrating its ability to significantly enhance the quality of proofs across various mathematical domains.

### 101. [Title:
          Deeper Insights Without Updates: The Power of In-Context Learning Over Fine-Tuning](https://arxiv.org/pdf/2410.04691)
**Summary**: The paper challenges the conventional belief that fine-tuning outperforms in-context learning (ICL) with sufficient training data, demonstrating that ICL excels in capturing implicit patterns in tasks. Through experiments on specialized datasets, the study shows that ICL models, even with fewer parameters, achieve higher accuracy and better pattern recognition compared to fine-tuned models. The authors propose a circuit shift theory to explain this phenomenon, suggesting that ICL's ability to quickly grasp deep patterns is a significant advantage.

### 102. [Title:
          Intriguing Properties of Large Language and Vision Models](https://arxiv.org/pdf/2410.04751)
**Summary**: The paper investigates the intriguing properties of large language and vision models (LLVMs), highlighting their surprising performance on advanced reasoning tasks despite weaker performance on fundamental perception tasks. Through extensive experiments across 10 benchmarks, the study reveals that LLVMs process images globally, can solve math problems without detailed numerical perception, and that cross-modal alignment is overfitted to complex reasoning tasks, potentially compromising original perceptual capabilities. The findings suggest future directions for improving LLVMs and creating more challenging evaluation benchmarks.

### 103. [Title:
          DEPT: Decoupled Embeddings for Pre-training Language Models](https://arxiv.org/pdf/2410.05021)
**Summary**: The paper introduces DEPT, a novel pre-training framework that decouples embedding layers from the transformer model, allowing for more robust and efficient training across heterogeneous data sources. DEPT reduces parameter count and communication costs, enhances model generalization, and enables custom vocabularies per data source, demonstrated through a 1.3 billion-parameter model pre-training across diverse languages.

### 104. [Title:
          Regressing the Relative Future: Efficient Policy Optimization for Multi-turn RLHF](https://arxiv.org/pdf/2410.04612)
**Summary**: The paper introduces REFUEL, an efficient policy optimization approach for multi-turn reinforcement learning from human feedback (RLHF) in large language models (LLMs), addressing the covariate shift issue by using a single model to estimate Q-values and training on self-generated data. REFUEL outperforms state-of-the-art methods like DPO and REBEL, and even a smaller model fine-tuned with REFUEL surpasses a larger model in long multi-turn dialogues.

### 105. [Title:
          TidalDecode: Fast and Accurate LLM Decoding with Position Persistent Sparse Attention](https://arxiv.org/pdf/2410.05076)
**Summary**: The paper introduces TidalDecode, a novel approach to improve the decoding efficiency of large language models (LLMs) by using position persistent sparse attention. By selectively applying full attention to a few layers to identify relevant tokens and sparse attention to the rest, TidalDecode reduces decoding latency by up to 2.1x without compromising the quality of generated text, outperforming existing sparse attention methods.

### 106. [Title:
          Efficient Inference for Large Language Model-based Generative Recommendation](https://arxiv.org/pdf/2410.05165)
**Summary**: The paper introduces AtSpeed, an alignment framework designed to accelerate Large Language Model (LLM)-based generative recommendation by improving top-K sequence alignment between the draft model and the target LLM, and by relaxing the verification strategy to reduce unnecessary LLM calls. Empirical results show significant speedups, with near 2x improvement under strict top-K verification and up to 2.5x under relaxed sampling verification.

### 107. [Title:
          Understanding Warmup-Stable-Decay Learning Rates: A River Valley Loss Landscape Perspective](https://arxiv.org/pdf/2410.05192)
**Summary**: The paper introduces the Warmup-Stable-Decay (WSD) learning rate schedule, which allows for indefinite training without a fixed compute budget by maintaining a constant learning rate during the stable phase and rapidly decaying it during the decay phase. The authors propose a "river valley" loss landscape to explain the observed behavior, where the stable phase facilitates rapid progress along the river, and the decay phase moves the model closer to the river's edge, improving optimization. They introduce WSD-S,

### 108. [Title:
          Density estimation with LLMs: a geometric investigation of in-context learning trajectories](https://arxiv.org/pdf/2410.05218)
**Summary**: The paper investigates how large language models (LLMs) like LLaMA-2 estimate probability density functions (PDFs) through in-context learning, using Intensive Principal Component Analysis (InPCA) to visualize their learning trajectories. It finds that LLMs follow distinct trajectories from traditional methods, resembling a KDE with adaptive kernel width and shape, and offers insights into the unique probabilistic reasoning mechanisms of LLMs.

### 109. [Title:
          Precise Model Benchmarking with Only a Few Observations](https://arxiv.org/pdf/2410.05222)
**Summary**: The paper introduces an empirical Bayes (EB) estimator to improve the precision of large language models' (LLM) accuracy estimates for specific topics within a dataset, especially when sample sizes are small. By balancing direct and regression estimates, the EB approach consistently reduces mean squared error and provides more reliable confidence intervals compared to traditional methods. This approach is also validated across different data types, demonstrating its versatility and effectiveness.

### 110. [Title:
          Preserving Multi-Modal Capabilities of Pre-trained VLMs for Improving Vision-Linguistic Compositionality](https://arxiv.org/pdf/2410.05210)
**Summary**: The paper introduces Fine-grained Selective Calibrated CLIP (FSC-CLIP) to enhance compositional understanding in pre-trained vision and language models without degrading multi-modal capabilities. By integrating local hard negative loss and selective calibrated regularization, FSC-CLIP maintains representational integrity and performs well on both compositionality and multi-modal tasks, outperforming state-of-the-art models in extensive evaluations.

### 111. [Title:
          Paraphrase Identification with Deep Learning: A Review of Datasets and Methods](https://arxiv.org/pdf/2212.06933)
**Summary**: The paper reviews current methods and datasets for paraphrase identification in NLP, highlighting the challenges posed by the inconsistent representation of paraphrase types in training data. It introduces a refined typology (ReParaphrased) to address these disparities and suggests future research directions to improve AI-based plagiarism detection.

### 112. [Title:
          Diversity Over Size: On the Effect of Sample and Topic Sizes for Topic-Dependent Argument Mining Datasets](https://arxiv.org/pdf/2205.11472)
**Summary**: The paper investigates the impact of dataset composition on Argument Mining performance in few- and zero-shot settings, finding that fine-tuning is crucial but that carefully selected training samples can significantly reduce dataset size without compromising performance. The study demonstrates consistent gains across multiple tasks and datasets, and introduces a new dataset for future research.

### 113. [Title:
          Self-Contradictory Reasoning Evaluation and Detection](https://arxiv.org/pdf/2311.09603)
**Summary**: The paper investigates self-contradictory reasoning in large language models (LLMs), finding that these models often produce inconsistent reasoning, particularly in tasks requiring contextual understanding or commonsense. While GPT-4 can detect some self-contradictions, its performance is significantly lower than human ability, highlighting the need for more robust evaluation methods in reasoning tasks beyond just final answer accuracy.

### 114. [Title:
          Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents](https://arxiv.org/pdf/2410.05243)
**Summary**: The paper proposes a novel approach for GUI agents to navigate digital environments using visual grounding models, akin to human perception, rather than relying on text-based representations like HTML. By leveraging a large dataset of GUI elements and referring expressions, the authors develop UGround, a universal visual grounding model that significantly outperforms existing models. This approach enables GUI agents to operate more effectively across various platforms, demonstrating the potential for human-like digital navigation.

### 115. [Title:
          PrefixQuant: Static Quantization Beats Dynamic through Prefixed Outliers in LLMs](https://arxiv.org/pdf/2410.05265)
**Summary**: The paper introduces PrefixQuant, a novel technique that isolates outlier tokens offline to enable efficient per-tensor static quantization in Large Language Models (LLMs), outperforming traditional per-token dynamic quantization methods. PrefixQuant achieves significant improvements in perplexity and accuracy, while also enhancing inference speed, making it a more efficient solution for deploying LLMs.

### 116. [Title:
          Social Bias Probing: Fairness Benchmarking for Language Models](https://arxiv.org/pdf/2311.09090)
**Summary**: The paper introduces a new framework for probing social biases in language models, focusing on disparate treatment across diverse demographic groups. It introduces SoFa, a large-scale benchmark that expands beyond binary comparisons to reveal more nuanced biases. The study finds that biases related to religious identities are particularly pronounced and that the models reflect real-world adversities faced by various groups.

### 117. [Title:
          Prompts have evil twins](https://arxiv.org/pdf/2311.07064)
**Summary**: The paper introduces "evil twins," unintelligible prompts that elicit similar behavior in language models as their natural-language counterparts, despite being uninterpretable to humans. These prompts are shown to transfer across different models and are generated by solving a maximum-likelihood problem, which has broader applications in understanding and manipulating model behavior.

### 118. [Title:
          Augmenting Black-box LLMs with Medical Textbooks for Biomedical Question Answering (Published in Findings of EMNLP 2024)](https://arxiv.org/pdf/2309.02233)
**Summary**: The paper introduces LLM-AMT, a system that enhances large language models (LLMs) like ChatGPT with medical textbooks to improve their performance in biomedical question answering. By integrating medical textbooks through specialized modules, LLM-AMT significantly boosts accuracy in medical QA tasks, outperforming even specialized models like Med-PaLM 2. The study highlights the effectiveness of medical textbooks as a knowledge source, proving more beneficial than Wikipedia in the medical domain.

### 119. [Title:
          When "A Helpful Assistant" Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models](https://arxiv.org/pdf/2311.10054)
**Summary**: The study investigates the impact of personas in system prompts on the performance of Large Language Models (LLMs), finding that adding personas does not consistently improve performance across various tasks. However, the gender, type, and domain of the persona can influence prediction accuracy, and aggregating results from the best persona for each question can enhance accuracy, though automatically identifying the best persona remains challenging.

### 120. [Title:
          Model Editing Harms General Abilities of Large Language Models: Regularization to the Rescue](https://arxiv.org/pdf/2401.04700)
**Summary**: The paper investigates the side effects of model editing on large language models (LLMs), finding that while editing improves factuality, it often degrades general abilities like reasoning and question answering. To address this, the authors propose RECT, a regularization method that constrains the complexity of weight updates, effectively mitigating side effects while preserving editing performance.

### 121. [Title:
          Large Language Models for Propaganda Span Annotation](https://arxiv.org/pdf/2311.09812)
**Summary**: The paper explores the use of Large Language Models (LLMs) like GPT-4 for detecting and annotating propagandistic spans in text, addressing the challenge of limited training data for lower-resourced languages. The study finds that providing more context to GPT-4 in prompts enhances its performance over human annotators and that GPT-4's labels can train specialized models achieving state-of-the-art results on an Arabic test set. This work demonstrates the potential of LLMs in creating

### 122. [Title:
          PILLOW: Enhancing Efficient Instruction Fine-tuning via Prompt Matching](https://arxiv.org/pdf/2312.05621)
**Summary**: The paper introduces PILLOW, a method to enhance the performance of Low-Rank Adaptation (LoRA) in fine-tuning Large Language Models (LLMs) by using a discrimination-based prompting approach. PILLOW leverages in-context learning and a matching network to select prompts from a user-defined pool, significantly reducing computational costs while maintaining comparable performance to traditional instruction fine-tuning methods.

### 123. [Title:
          Length Extrapolation of Transformers: A Survey from the Perspective of Positional Encoding](https://arxiv.org/pdf/2312.17044)
**Summary**: The paper surveys methods for enhancing the length extrapolation capabilities of Transformers, focusing on positional encoding (PE) as the primary factor influencing this ability. It categorizes existing approaches into extrapolatable PEs, position interpolation, and randomized position methods, and highlights challenges and future research directions in this area. The survey aims to provide a comprehensive understanding of current techniques and inspire further advancements in handling long input sequences.

### 124. [Title:
          SH2: Self-Highlighted Hesitation Helps You Decode More Truthfully](https://arxiv.org/pdf/2401.05930)
**Summary**: The paper introduces Self-Highlighted Hesitation (SH2), an inference-time method designed to enhance the factual accuracy of large language models (LLMs) by focusing on tokens with lower prediction probabilities. SH2 highlights these tokens and incorporates them into the decoding process, encouraging the model to reconsider and hesitate on potentially factual information. The method, which requires no additional data or models, shows significant improvements in reducing hallucinations across multiple LLMs and tasks.

### 125. [Title:
          Multi-User Chat Assistant (MUCA): a Framework Using LLMs to Facilitate Group Conversations](https://arxiv.org/pdf/2401.04883)
**Summary**: The paper introduces the Multi-User Chat Assistant (MUCA), a framework using large language models (LLMs) to facilitate group conversations by addressing the complexities of multi-user interactions through three design dimensions: "What" to say, "When" to respond, and "Who" to answer. MUCA's modules—Sub-topic Generator, Dialog Analyzer, and Conversational Strategies Arbitrator—work together to determine appropriate responses, timings, and addressees, enhancing user engagement in

### 126. [Title:
          Large Language Models are Geographically Biased](https://arxiv.org/pdf/2402.02680)
**Summary**: The paper investigates the geographic biases present in large language models (LLMs), revealing that these models exhibit systemic errors in geospatial predictions, particularly favoring regions with higher socioeconomic conditions. The study introduces a bias score to quantify these biases and highlights the need for addressing such inaccuracies to achieve fairness in AI models.

### 127. [Title:
          Numerical Claim Detection in Finance: A New Financial Dataset, Weak-Supervision Model, and Market Analysis](https://arxiv.org/pdf/2402.11728)
**Summary**: The paper introduces a new financial dataset for detecting numerical claims in analyst reports and earnings calls, aiming to analyze their impact on market returns. It proposes a weak-supervision model that integrates expert knowledge, outperforming existing methods, and demonstrates its utility by creating an optimism measure linked to earnings surprises and returns.

### 128. [Title:
          Corrective Retrieval Augmented Generation](https://arxiv.org/pdf/2401.15884)
**Summary**: The paper introduces Corrective Retrieval Augmented Generation (CRAG), a method designed to enhance the robustness of retrieval-augmented generation (RAG) by incorporating a retrieval evaluator to assess the quality of retrieved documents. CRAG employs large-scale web searches to augment retrieval results and uses a decompose-then-recompose algorithm to focus on key information, improving the performance of RAG-based approaches across various tasks.

### 129. [Title:
          R-Judge: Benchmarking Safety Risk Awareness for LLM Agents](https://arxiv.org/pdf/2401.10019)
**Summary**: The paper introduces R-Judge, a benchmark designed to assess the safety risk awareness of large language model (LLM) agents in interactive environments. R-Judge includes 569 multi-turn interaction records across 27 risk scenarios, covering various application categories and risk types. Evaluation of 11 LLMs reveals significant room for improvement in risk awareness, with GPT-4o achieving the highest score of 74.42%. The study highlights the complexity of risk awareness in open

### 130. [Title:
          Pedagogical Alignment of Large Language Models](https://arxiv.org/pdf/2402.05000)
**Summary**: The paper explores the use of Learning from Human Preferences (LHP) algorithms to align Large Language Models (LLMs) with effective teaching strategies, termed "pedagogical alignment." By generating synthetic datasets to overcome the scarcity of high-quality preference data, the study demonstrates that LHP methods outperform standard supervised fine-tuning, improving alignment accuracy. The authors also introduce new perplexity-based metrics to quantitatively assess pedagogical alignment, highlighting the potential of LHP methods to enhance LLMs' effectiveness in educational

### 131. [Title:
          Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models](https://arxiv.org/pdf/2402.10353)
**Summary**: The paper introduces a null-input prompting method to calibrate intrinsic bias in pre-trained language models, aiming to improve zero/few-shot learning performance while maintaining computational efficiency. By using GPT-4-generated null-meaning inputs and a distribution disparity loss, the method adjusts bias parameters to create a more equitable starting point for language models, leading to significant improvements in zero/few-shot learning across various datasets.

### 132. [Title:
          Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://arxiv.org/pdf/2402.13950)
**Summary**: The paper investigates the faithfulness of reasoning steps in large language models (LLMs) and finds that LLMs do not consistently use their intermediate reasoning steps to generate final answers. To address this, the authors introduce FRODO, a framework that trains small-sized LMs to produce correct reasoning steps and faithfully reason over them, leading to improved robustness and generalization on out-of-distribution test sets.

### 133. [Title:
          Head-wise Shareable Attention for Large Language Models](https://arxiv.org/pdf/2402.11819)
**Summary**: The paper introduces head-wise shareable attention as a method to reduce the memory footprint of large language models (LLMs) by sharing parameters across attention heads. Two approaches, **DirectShare** and **PostShare**, are proposed to implement this fine-grained weight sharing without significant performance degradation. The experiments show that the head-wise shared models maintain satisfactory capabilities, indicating the feasibility of applying fine-grained weight sharing to LLMs.

### 134. [Title:
          Unraveling Babel: Exploring Multilingual Activation Patterns of LLMs and Their Applications](https://arxiv.org/pdf/2402.16367)
**Summary**: The paper investigates the internal neuron activation patterns of large language models (LLMs) when processing different languages by converting dense models into fine-grained mixture-of-experts (MoE) architectures. Through visual analysis and experiments, the study identifies patterns in expert activations across languages, revealing insights into multilingual processing mechanisms. The findings are applied to improve sparse activation and model pruning techniques, demonstrating superior performance compared to random pruning and even unpruned models in some cases.

### 135. [Title:
          Can LLM Generate Culturally Relevant Commonsense QA Data? Case Study in Indonesian and Sundanese](https://arxiv.org/pdf/2402.17302)
**Summary**: The study explores the capability of Large Language Models (LLMs) to generate culturally relevant commonsense question-answering (QA) datasets for Indonesian and Sundanese languages. By creating datasets through both LLM-based and human-annotated methods, the researchers found that while GPT-4 Turbo can generate questions with adequate general knowledge, it falls short in capturing deeper cultural nuances compared to human annotations. The study also highlights significant fluency errors in the Sundanese dataset, indicating challenges in adapting LLMs for

### 136. [Title:
          FAC$^2$E: Better Understanding Large Language Model Capabilities by Dissociating Language and Cognition](https://arxiv.org/pdf/2403.00126)
**Summary**: The paper introduces FAC$^2$E, a framework for evaluating large language models (LLMs) by distinguishing between language and cognitive capabilities. It breaks down the evaluation process into three sub-steps: knowledge recall, knowledge utilization, and problem-solving, providing a detailed diagnosis of LLMs' performance. The study identifies a common weakness in knowledge utilization and suggests a knowledge-enhanced method to improve LLM performance.

### 137. [Title:
          Editing Conceptual Knowledge for Large Language Models](https://arxiv.org/pdf/2403.06259)
**Summary**: The paper introduces a novel approach to editing conceptual knowledge in Large Language Models (LLMs) by creating the ConceptEdit benchmark dataset and new evaluation metrics. It finds that while existing editing methods can modify concept-level definitions, they often distort related instance-level knowledge, highlighting the need for more refined techniques to balance these changes.

### 138. [Title:
          Tokenization Is More Than Compression](https://arxiv.org/pdf/2402.18376)
**Summary**: The paper challenges the common belief that fewer tokens lead to better performance in natural language processing by introducing PathPiece, a tokenizer designed to minimize token count. Through extensive experiments, the authors find that fewer tokens do not necessarily improve downstream task performance, questioning the current understanding of effective tokenization. The study highlights the importance of pre-tokenization and the benefits of using BPE for vocabulary initialization, offering new insights into tokenizer design.

### 139. [Title:
          sDPO: Don't Use Your Data All at Once](https://arxiv.org/pdf/2403.19270)
**Summary**: The paper introduces stepwise DPO (sDPO), an extension of direct preference optimization (DPO) that aligns large language models (LLMs) with human preferences by dividing preference datasets and using them in stages. This method enhances the precision of reference models and results in a final model that outperforms other LLMs, including those with more parameters.

### 140. [Title:
          Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/pdf/2402.17840)
**Summary**: The paper investigates the vulnerability of Retrieval-Augmented Generation (RAG) systems, particularly those using instruction-tuned Language Models (LMs), to datastore leakage through prompt injection. The study demonstrates that adversaries can exploit these systems to extract verbatim text data, with the vulnerability increasing with model size. The authors also propose mitigation strategies, such as position bias elimination, and show successful attacks on production RAG models like GPTs, extracting significant amounts of text data with minimal queries.<｜end▁of▁sentence｜>

### 141. [Title:
          HateCOT: An Explanation-Enhanced Dataset for Generalizable Offensive Speech Detection via Large Language Models](https://arxiv.org/pdf/2403.11456)
**Summary**: The paper introduces HateCOT, a large English dataset with over 52,000 samples for offensive speech detection, enhanced with GPT-3.5Turbo-generated explanations. Pretraining on HateCOT improves the generalization of Large Language Models on various offensive content detection benchmarks, even in zero-shot and few-shot scenarios, and enhances the quality of model explanations.

### 142. [Title:
          Evalverse: Unified and Accessible Library for Large Language Model Evaluation](https://arxiv.org/pdf/2404.00943)
**Summary**: The paper introduces Evalverse, a unified and accessible library designed to simplify the evaluation of Large Language Models (LLMs) by consolidating various evaluation tools into a single framework. Evalverse allows users with minimal AI expertise to request and receive detailed LLM evaluation reports through integrated communication platforms like Slack, making it a valuable resource for both researchers and practitioners.

### 143. [Title:
          The Generation Gap: Exploring Age Bias in the Value Systems of Large Language Models](https://arxiv.org/pdf/2404.08760)
**Summary**: The paper investigates age bias in Large Language Models (LLMs) by comparing their value systems to those of different age groups using data from the World Value Survey. It finds that LLMs tend to align more with younger demographics, particularly in the US, and that this bias varies across different value categories. The study also examines the effect of including age identity in prompts, revealing challenges in reducing value discrepancies between LLMs and various age cohorts.

### 144. [Title:
          MetaAligner: Towards Generalizable Multi-Objective Alignment of Language Models](https://arxiv.org/pdf/2403.17141)
**Summary**: The paper introduces MetaAligner, a novel approach for generalizable multi-objective alignment of language models that is policy-agnostic and adaptable to new objectives. It achieves this through a three-stage process involving dynamic objective reformulation, conditional weak-to-strong correction, and a generalizable inference method, which together significantly reduce training costs and improve alignment performance across various models and objectives.

### 145. [Title:
          Robust Pronoun Fidelity with English LLMs: Are they Reasoning, Repeating, or Just Biased?](https://arxiv.org/pdf/2404.03134)
**Summary**: The paper introduces the task of pronoun fidelity, aiming to measure the robustness of pronoun usage in language models. Using the RUFF dataset, the study evaluates 37 model variants and finds that while models generally reuse pronouns correctly, they struggle with specific pronouns like "she," "they," and neopronouns, and are easily distracted by unrelated sentences. The results highlight the need for improved pronoun handling and careful evaluation to avoid overestimating model performance.

### 146. [Title:
          Characterizing LLM Abstention Behavior in Science QA with Context Perturbations](https://arxiv.org/pdf/2404.12452)
**Summary**: The paper investigates the ability of large language models (LLMs) to abstain from answering science questions when provided with insufficient or incorrect context. Through experiments on four QA datasets and six LLMs, the study reveals significant variability in abstention behavior across models and question types, with some models struggling to abstain on boolean questions. The analysis also shows that altering context, such as replacing gold context with irrelevant information, can paradoxically improve both abstention and overall task performance, suggesting the need for changes in

### 147. [Title:
          SpaceByte: Towards Deleting Tokenization from Large Language Modeling](https://arxiv.org/pdf/2404.14408)
**Summary**: The paper introduces SpaceByte, a novel byte-level decoder architecture designed to eliminate the need for tokenization in large language models while maintaining performance. By incorporating larger transformer blocks after specific byte markers like spaces, SpaceByte significantly improves byte-level modeling and achieves performance comparable to tokenized models, demonstrating its effectiveness within a fixed computational budget.

### 148. [Title:
          NegotiationToM: A Benchmark for Stress-testing Machine Theory of Mind on Negotiation Surrounding](https://arxiv.org/pdf/2404.13627)
**Summary**: The paper introduces NegotiationToM, a benchmark designed to evaluate the Theory of Mind (ToM) capabilities of large language models (LLMs) in real-world negotiation scenarios involving complex mental states. The benchmark, based on the Belief-Desire-Intention (BDI) theory, reveals that current state-of-the-art LLMs struggle significantly compared to humans, even with advanced reasoning techniques like chain-of-thought.

### 149. [Title:
          Representation noising effectively prevents harmful fine-tuning on LLMs](https://arxiv.org/pdf/2405.14577)
**Summary**: The paper introduces Representation Noising (RepNoise), a defense mechanism against harmful fine-tuning attacks on large language models (LLMs). RepNoise removes information about harmful representations, making it difficult for attackers to recover them during fine-tuning, even when they have access to the model weights. The method is shown to generalize across different types of harm and does not impair the model's performance on harmless tasks.

### 150. [Title:
          Red Teaming Language Models for Processing Contradictory Dialogues](https://arxiv.org/pdf/2405.10128)
**Summary**: The paper introduces a novel task for processing contradictory dialogues in language models, inspired by context faithfulness and dialogue comprehension research. It develops a dataset with contradictory dialogues labeled for explanations and proposes a Red Teaming framework that detects, explains, and modifies contradictions. The framework improves detection and explanation accuracy, emphasizing the significance of logical consistency in conversational AI.

### 151. [Title:
          Language in Vivo vs. in Silico: Size Matters but Larger Language Models Still Do Not Comprehend Language on a Par with Humans](https://arxiv.org/pdf/2404.14883)
**Summary**: The study examines the performance of three large language models (LLMs) on a grammaticality judgment task, comparing their accuracy and stability to human performance. While the largest model, ChatGPT-4, outperforms humans in recognizing grammatical sentences, it shows less stability and sensitivity to ungrammaticality compared to humans. The findings suggest that model scaling alone may not fully bridge the gap between human and LLM comprehension of language.

### 152. [Title:
          Generate-on-Graph: Treat LLM as both Agent and KG in Incomplete Knowledge Graph Question Answering](https://arxiv.org/pdf/2404.14741)
**Summary**: The paper introduces Generate-on-Graph (GoG), a training-free method that leverages Large Language Models (LLMs) to address Incomplete Knowledge Graph Question Answering (IKGQA) by generating new factual triples when the provided Knowledge Graph (KG) is insufficient. GoG operates through a Thinking-Searching-Generating framework, treating LLMs as both agents and KGs, and outperforms previous methods in experimental evaluations on two datasets.

### 153. [Title:
          Student Data Paradox and Curious Case of Single Student-Tutor Model: Regressive Side Effects of Training LLMs for Personalized Learning](https://arxiv.org/pdf/2404.15156)
**Summary**: The paper identifies the "Student Data Paradox," where training Large Language Models (LLMs) on extensive student-tutor dialogue datasets to personalize education leads to a decline in the models' factual knowledge and reasoning abilities. The study demonstrates this paradox through quantitative analysis and introduces "hallucination tokens" as a partial solution, highlighting the ongoing challenge of balancing accurate student behavior modeling with maintaining the LLM's educational integrity.

### 154. [Title:
          Promoting Constructive Deliberation: Reframing for Receptiveness](https://arxiv.org/pdf/2405.15067)
**Summary**: The paper introduces a method to enhance online discussions by automatically reframing disagreeing responses to appear more receptive, based on six identified strategies. Experiments using a Reddit dataset show that these reframed replies are perceived as significantly more receptive than original replies and a generic baseline, demonstrating the potential of computational frameworks to align with human perceptions in content moderation.

### 155. [Title:
          Exploring the Compositional Deficiency of Large Language Models in Mathematical Reasoning](https://arxiv.org/pdf/2405.06680)
**Summary**: The paper investigates the compositionality of large language models (LLMs) in mathematical reasoning by introducing logical traps into the MATH and GSM8k datasets, revealing that LLMs struggle to spontaneously combine mathematical knowledge with logical reasoning. The study explores methods like natural language prompts, few-shot demonstrations, and fine-tuning to improve performance, highlighting that while external interventions can enhance results, systematic compositionality remains a challenge for LLMs.

### 156. [Title:
          DAPE: Data-Adaptive Positional Encoding for Length Extrapolation](https://arxiv.org/pdf/2405.14722)
**Summary**: The paper introduces Data-Adaptive Positional Encoding (DAPE), a novel method that dynamically adjusts positional encoding based on input context and learned priors, addressing the limitations of fixed positional encodings in transformers. Experiments on real-world datasets show that DAPE significantly improves model performance and length generalization, outperforming static methods, especially in handling longer sequences.

### 157. [Title:
          Superposed Decoding: Multiple Generations from a Single Autoregressive Inference Pass](https://arxiv.org/pdf/2405.18400)
**Summary**: The paper introduces Superposed Decoding, a novel algorithm that generates multiple text drafts from a single autoregressive inference pass, significantly reducing computational costs compared to traditional methods that require multiple passes. The approach combines and filters drafts using n-gram interpolation, resulting in coherent and factual outputs that are faster and preferred by users in compute-normalized settings.

### 158. [Title:
          WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models](https://arxiv.org/pdf/2405.14768)
**Summary**: The paper introduces WISE, a novel approach for lifelong model editing in large language models (LLMs) that addresses the challenge of balancing reliability, generalization, and locality when updating knowledge. WISE employs a dual parametric memory scheme, with a main memory for pretrained knowledge and a side memory for edited knowledge, along with a router to manage queries. A knowledge-sharding mechanism ensures that edits are stored in distinct parameter subspaces, preventing conflicts and enabling effective merging into a shared memory. Experiments demonstrate WISE's superior

### 159. [Title:
          RLSF: Reinforcement Learning via Symbolic Feedback](https://arxiv.org/pdf/2405.16661)
**Summary**: The paper introduces Reinforcement Learning via Symbolic Feedback (RLSF), a novel fine-tuning method for Large Language Models (LLMs) that leverages reasoning tools to provide detailed, token-level feedback, addressing the limitations of traditional scalar reward models. RLSF enables LLMs to achieve superior performance in domain-specific tasks, outperforming larger models like GPT-4 in various applications, including program synthesis and chemistry tasks.

### 160. [Title:
          Evaluating and Safeguarding the Adversarial Robustness of Retrieval-Based In-Context Learning](https://arxiv.org/pdf/2405.15984)
**Summary**: The paper investigates the robustness of Retrieval-Augmented In-Context Learning (ICL) methods against adversarial attacks, finding that while they improve robustness against test sample attacks, they are more vulnerable to demonstration attacks. The study introduces a training-free defense method, DARD, which enhances robustness by enriching the example pool with attacked samples, achieving a 15% reduction in Attack Success Rate (ASR) compared to baselines.

### 161. [Title:
          Two Tales of Persona in LLMs: A Survey of Role-Playing and Personalization](https://arxiv.org/pdf/2406.01171)
**Summary**: The paper presents a comprehensive survey on the use of personas in large language models (LLMs), categorizing current research into two main areas: LLM Role-Playing, where personas are assigned to LLMs, and LLM Personalization, where LLMs adapt to user personas. The survey also introduces methods for evaluating LLM personality and aims to provide a systematic taxonomy for future research in this area.

### 162. [Title:
          Auto-Arena: Automating LLM Evaluations with Agent Peer Battles and Committee Discussions](https://arxiv.org/pdf/2405.20267)
**Summary**: The paper introduces Auto-Arena, a framework that automates the evaluation of large language models (LLMs) through agent-based peer battles and committee discussions. By generating questions and having LLM candidates compete in multi-round battles, followed by a collaborative decision-making process, Auto-Arena achieves a 92.14% correlation with human preferences, outperforming existing benchmarks without requiring manual effort. This method offers a reliable and efficient alternative to human-based evaluation platforms.

### 163. [Title:
          Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning CodeLLMs](https://arxiv.org/pdf/2405.20179)
**Summary**: The paper introduces ROBO-INSTRUCT, a method that combines a robot simulator (ROBOSIM) with an LLM-aided alignment process (INSTALIGN) to generate diverse and correct programs for fine-tuning Code LLMs in robot applications. This approach significantly improves the performance of the fine-tuned model, achieving better results than proprietary LLMs like GPT-3.5-Turbo and Gemini-Pro.

### 164. [Title:
          MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark (Published at NeurIPS 2024 Track Datasets and Benchmarks)](https://arxiv.org/pdf/2406.01574)
**Summary**: The paper introduces MMLU-Pro, an advanced version of the MMLU benchmark, designed to challenge language models with more complex reasoning tasks and a larger choice set. MMLU-Pro significantly reduces model accuracy and demonstrates greater stability under different prompts, making it a more robust and challenging benchmark for evaluating AI language comprehension and reasoning capabilities.

### 165. [Title:
          ComplexTempQA: A Large-Scale Dataset for Complex Temporal Question Answering](https://arxiv.org/pdf/2406.04866)
**Summary**: The paper introduces ComplexTempQA, a massive dataset of over 100 million question-answer pairs designed to challenge AI models in temporal question answering. It surpasses existing benchmarks in scale and complexity, featuring questions that require advanced reasoning skills like temporal comparison and multi-hop reasoning. The dataset, enriched with detailed metadata, aims to enhance the temporal reasoning capabilities of large language models and advance research in question answering and information retrieval.

### 166. [Title:
          Understanding Jailbreak Success: A Study of Latent Space Dynamics in Large Language Models](https://arxiv.org/pdf/2406.09289)
**Summary**: The paper investigates how different jailbreak techniques circumvent safeguards in large language models by analyzing model activations. It identifies a common jailbreak vector that reduces the model's perception of prompt harmfulness, suggesting a shared internal mechanism across various jailbreak types. This insight could inform the development of more robust countermeasures against jailbreaking.

### 167. [Title:
          FacLens: Transferable Probe for Foreseeing Non-Factuality in Large Language Models](https://arxiv.org/pdf/2406.05328)
**Summary**: The paper introduces FacLens, a lightweight model designed to predict non-factual responses from large language models (LLMs) before they are generated. FacLens leverages hidden representations of questions to identify potential non-factuality, demonstrating transferability across different LLMs, which reduces development costs. The model outperforms existing methods in both effectiveness and efficiency.

### 168. [Title:
          Test-Time Fairness and Robustness in Large Language Models](https://arxiv.org/pdf/2406.07685)
**Summary**: The paper introduces a novel approach to address biases in large language models (LLMs) at test time, particularly when only well-resourced entities can train these models. It proposes a stratified invariance framework that allows for explicit debiasing requirements through causality, overcoming limitations of standard causal debiasing methods. The authors demonstrate that their prompting strategy effectively reduces bias in LLMs across various benchmarks without additional data or retraining.

### 169. [Title:
          Advancing Semantic Textual Similarity Modeling: A Regression Framework with Translated ReLU and Smooth K2 Loss](https://arxiv.org/pdf/2406.05326)
**Summary**: The paper introduces a regression framework for Semantic Textual Similarity (STS) that addresses the limitations of contrastive learning by proposing two new loss functions, Translated ReLU and Smooth K2 Loss. This approach allows for fine-grained similarity modeling and outperforms existing methods on seven STS benchmarks, suggesting potential improvements for contrastive learning models.

### 170. [Title:
          Annotation alignment: Comparing LLM and human annotations of conversational safety](https://arxiv.org/pdf/2406.06369)
**Summary**: The paper investigates the alignment between human and Large Language Model (LLM) perceptions of conversational safety using the DICES dataset. GPT-4 demonstrates a higher correlation with average human safety ratings compared to individual annotators, but the study highlights the need for larger datasets to assess potential disparities across demographic groups. The analysis also reveals significant variation within groups, indicating that race and gender alone do not fully explain differences in alignment.

### 171. [Title:
          WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild](https://arxiv.org/pdf/2406.04770)
**Summary**: The paper introduces WildBench, an automated evaluation framework for large language models (LLMs) using challenging, real-world user queries from over one million conversation logs. It employs two metrics, WB-Reward and WB-Score, to systematically evaluate model outputs, with WB-Reward showing a strong correlation (0.98 Pearson correlation) with human-voted Elo ratings on hard tasks, outperforming other benchmarks.

### 172. [Title:
          Understanding "Democratization" in NLP and ML Research](https://arxiv.org/pdf/2406.11598)
**Summary**: The paper examines how the term "democratization" is used in NLP and ML research, finding that it often refers to the accessibility of technologies rather than deeper democratic principles. The authors advocate for a more theoretically grounded use of the term to promote truly democratic technologies beyond mere access.

### 173. [Title:
          Mixture-of-Skills: Learning to Optimize Data Usage for Fine-Tuning Large Language Models](https://arxiv.org/pdf/2406.08811)
**Summary**: The paper introduces Mixture-of-Skills (MoS), a reinforcement learning framework designed to optimize data usage during the fine-tuning of large language models (LLMs), addressing the challenges posed by heterogeneous and imbalanced datasets. MoS dynamically adjusts the focus on different datasets to ensure comprehensive skill development, and its effectiveness is demonstrated through extensive experiments on diverse LLM backbones and benchmarks. Additionally, the paper proposes MoSpec, an adaptation for task-specific fine-tuning, highlighting the importance of

### 174. [Title:
          Pcc-tuning: Breaking the Contrastive Learning Ceiling in Semantic Textual Similarity](https://arxiv.org/pdf/2406.09790)
**Summary**: The paper investigates the limitations of current contrastive learning methods in Semantic Textual Similarity (STS) tasks, identifying an upper limit of 87.5 for Spearman's correlation scores. To overcome this ceiling, the authors propose Pcc-tuning, which uses Pearson's correlation coefficient as a loss function, significantly improving performance with minimal fine-grained annotations.

### 175. [Title:
          Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/pdf/2406.08464)
**Summary**: The paper introduces Magpie, a method for synthesizing high-quality instruction data by prompting aligned large language models (LLMs) like Llama-3-Instruct with partial templates. Magpie generates 4 million instruction-response pairs, from which 300K high-quality instances are selected. Fine-tuning Llama-3-8B-Base with Magpie data shows comparable performance to the official Llama-3-8B-Instruct, even surpassing previous

### 176. [Title:
          Learn Beyond The Answer: Training Language Models with Reflection for Mathematical Reasoning](https://arxiv.org/pdf/2406.12050)
**Summary**: The paper introduces "reflective augmentation," a novel technique that enhances language models' mathematical reasoning by embedding problem reflection into training instances, encouraging deeper understanding and reflective thinking. This method outperforms existing data augmentation techniques, especially in complex scenarios requiring reflective reasoning, as validated by extensive experiments.

### 177. [Title:
          Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/pdf/2406.11695)
**Summary**: The paper introduces MIPRO, an algorithm for optimizing prompts in multi-stage Language Model Programs (LMPs) by refining free-form instructions and few-shot demonstrations. MIPRO employs program- and data-aware techniques, a stochastic evaluation function, and a meta-optimization procedure to enhance performance. It demonstrates superior results on diverse LMP tasks, outperforming baseline optimizers by up to 13% accuracy using the Llama-3-8B model.

### 178. [Title:
          Self-MoE: Towards Compositional Large Language Models with Self-Specialized Experts](https://arxiv.org/pdf/2406.12034)
**Summary**: The paper introduces Self-MoE, a method that converts monolithic large language models (LLMs) into modular systems of self-specialized experts (MiXSE) using self-generated synthetic data. This approach enhances the LLM's performance across various tasks without requiring extensive human-labeled data, showing significant improvements (6.5% on average) over the base LLM in benchmarks. The study also emphasizes the benefits of modularity and self-improvement in creating efficient and adaptable systems

### 179. [Title:
          What Matters in Memorizing and Recalling Facts? Multifaceted Benchmarks for Knowledge Probing in Language Models](https://arxiv.org/pdf/2406.12277)
**Summary**: The paper introduces BELIEF(ICL), a knowledge probing benchmark designed to evaluate the factual knowledge recall abilities of both encoder- and decoder-based pre-trained language models (PLMs). It uses a multi-prompt dataset, MyriadLAMA, to assess accuracy, consistency, and reliability in recalling facts and investigates factors influencing knowledge recall, such as model size and pretraining strategies.

### 180. [Title:
          COMMUNITY-CROSS-INSTRUCT: Unsupervised Instruction Generation for Aligning Large Language Models to Online Communities](https://arxiv.org/pdf/2406.12074)
**Summary**: The paper introduces Community-Cross-Instruct, an unsupervised framework for aligning large language models (LLMs) to online communities by automatically generating instruction-output pairs from community discussions. This method allows for the fine-tuning of foundational LLMs to accurately represent and evaluate the beliefs of specific communities, demonstrated through applications on Reddit. Unlike previous methods, it does not require human-authored instructions, making it scalable and applicable across various domains.

### 181. [Title:
          When Parts Are Greater Than Sums: Individual LLM Components Can Outperform Full Models](https://arxiv.org/pdf/2406.13131)
**Summary**: The paper investigates in-context learning by analyzing the individual contributions of attention heads and MLPs within large language models. It identifies various types of components—good-performing, bad-performing, and label-biased—and demonstrates that reweighting these components can significantly enhance model accuracy. The proposed method, component reweighting, improves performance by an average of 6.0% accuracy points across multiple tasks with just 24 labeled examples.

### 182. [Title:
          Insights into LLM Long-Context Failures: When Transformers Know but Don't Tell](https://arxiv.org/pdf/2406.14673)
**Summary**: The study investigates the limitations of Large Language Models (LLMs) in handling long contexts, revealing that while these models encode the position of relevant information, they frequently fail to utilize it effectively in generating accurate responses. This "know but don't tell" phenomenon highlights a disconnect between information retrieval and utilization, with the analysis also exploring the relationship between extraction time and final accuracy in transformer models.

### 183. [Title:
          First Heuristic Then Rational: Dynamic Use of Heuristics in Language Model Reasoning](https://arxiv.org/pdf/2406.16078)
**Summary**: The paper investigates how language models (LMs) use heuristics during multi-step reasoning tasks, finding that LMs rely more on heuristics early in the process and shift to more rational strategies as they approach the final answer. This dynamic use of heuristics and rational reasoning suggests that LMs can effectively combine both approaches to improve performance in complex reasoning tasks.

### 184. [Title:
          Mental Disorder Classification via Temporal Representation of Text](https://arxiv.org/pdf/2406.15470)
**Summary**: The paper introduces a novel framework for mental disorder classification by compressing chronological social media posts into a time-variant numerical representation, addressing the limitations of current language models in handling sequential text data. The approach significantly improves classification performance across depression, self-harm, and anorexia, with a 5% absolute increase in F1 score, and demonstrates the importance of temporal properties in textual data analysis. Additionally, the framework is applied to a cross-domain study, revealing potential commonalities and inter

### 185. [Title:
          From Insights to Actions: The Impact of Interpretability and Analysis Research on NLP](https://arxiv.org/pdf/2406.12618)
**Summary**: The paper investigates the impact of interpretability and analysis (IA) research on the broader field of NLP, finding that IA work is well-cited and central in the NLP citation graph. Survey responses and manual annotations indicate that NLP researchers widely acknowledge the importance of IA findings for advancing the field, with many novel methods influenced by IA research. The study concludes by identifying gaps in current IA work and calls for more impactful future research.

### 186. [Title:
          SoK: Membership Inference Attacks on LLMs are Rushing Nowhere (and How to Fix It)](https://arxiv.org/pdf/2406.17975)
**Summary**: The paper examines the growing concern of Membership Inference Attacks (MIAs) on Large Language Models (LLMs) and highlights the lack of randomization in current evaluation methods, leading to significant distribution shifts that undermine the validity of these attacks. The authors propose new evaluation strategies, including randomized test splits and unique sequence injections, to address these issues and suggest comprehensive benchmarks for future research.

### 187. [Title:
          FastMem: Fast Memorization of Prompt Improves Context Awareness of Large Language Models](https://arxiv.org/pdf/2406.16069)
**Summary**: The paper introduces FastMem, a method to improve the context awareness of large language models (LLMs) by quickly memorizing the prompt before inference. By optimizing only the last Feed-Forward Network module, FastMem enhances the model's ability to comprehend and follow context, leading to significant improvements in tasks like reading comprehension and text summarization. The method shows notable accuracy gains and reduced output structure failures in experiments with models like Llama and Qwen.

### 188. [Title:
          Native Design Bias: Studying the Impact of English Nativeness on Language Model Performance](https://arxiv.org/pdf/2406.17385)
**Summary**: The study examines whether large language models (LLMs) exhibit performance discrepancies in response quality based on the nativeness of English speakers, finding that non-native English speakers often receive lower-quality or factually incorrect responses. The research highlights a strong anchoring effect where the model's recognition of a user's nativeness further degrades response quality for non-native speakers, supported by a dataset of over 12,000 annotations from diverse annotators.

### 189. [Title:
          Multi-LogiEval: Towards Evaluating Multi-Step Logical Reasoning Ability of Large Language Models](https://arxiv.org/pdf/2406.17169)
**Summary**: The paper introduces Multi-LogiEval, a comprehensive evaluation dataset designed to assess the multi-step logical reasoning capabilities of Large Language Models (LLMs) across various inference rules and depths, including non-monotonic reasoning. The dataset covers three logic types and includes over 30 inference rules and their combinations. Experimental results indicate a significant performance decline in LLMs as reasoning depth increases, highlighting the need for further research to enhance their logical reasoning abilities.

### 190. [Title:
          Is It Really Long Context if All You Need Is Retrieval? Towards Genuinely Difficult Long Context NLP](https://arxiv.org/pdf/2407.00402)
**Summary**: The paper argues that grouping various long-context NLP tasks solely by input length is unproductive and proposes a more nuanced taxonomy based on two axes of difficulty: Diffusion (finding necessary information) and Scope (amount of necessary information). It highlights the under-exploration of tasks with highly diffused and extensive necessary information and calls for more informed research and benchmark design in long-context NLP.

### 191. [Title:
          Detection and Measurement of Syntactic Templates in Generated Text](https://arxiv.org/pdf/2407.00211)
**Summary**: The paper introduces syntactic templates as a method to analyze repetition in text generated by large language models (LLMs), beyond word-level features. It finds that models frequently produce templated text, with 76% of these templates originating from pre-training data, and that these templates persist through fine-tuning. The study demonstrates that syntactic templates can differentiate between models, tasks, and domains, and serve as a valuable tool for evaluating model behavior and style memorization.

### 192. [Title:
          Make Some Noise: Unlocking Language Model Parallel Inference Capability through Noisy Training](https://arxiv.org/pdf/2406.17404)
**Summary**: The paper introduces the Make Some Noise (MSN) training framework, which enhances the parallel decoding capability of large language models by introducing noise during training, without requiring additional model structures or memory-intensive processes. The authors also propose a tree-based retrieval-augmented Jacobi (TR-Jacobi) decoding strategy to further boost inference speed. Experiments demonstrate that MSN can achieve up to 2.7x faster inference without compromising model performance, comparable to state-of-the-art models with additional

### 193. [Title:
          DogeRM: Equipping Reward Models with Domain Knowledge through Model Merging](https://arxiv.org/pdf/2407.01470)
**Summary**: The paper introduces DogeRM, a framework that enhances reward models in reinforcement learning from human feedback by integrating domain-specific knowledge through model merging. This approach reduces the need for extensive paired preference data collection, particularly in specialized domains, and demonstrates improved performance across various benchmarks, highlighting the potential for more efficient model alignment.

### 194. [Title:
          A Survey on Natural Language Counterfactual Generation](https://arxiv.org/pdf/2407.03993)
**Summary**: The paper surveys natural language counterfactual generation, a technique that modifies texts to change their classification outcomes, providing insights into model predictions and enhancing robustness. It categorizes methods into four groups and discusses evaluation metrics, highlighting ongoing challenges and future research directions.

### 195. [Title:
          MalAlgoQA: Pedagogical Evaluation of Counterfactual Reasoning in Large Language Models and Implications for AI in Education](https://arxiv.org/pdf/2407.00938)
**Summary**: The paper introduces MalAlgoQA, a dataset designed to evaluate counterfactual reasoning in Large Language Models (LLMs) through mathematics and reading comprehension questions. It focuses on assessing LLMs' ability to identify flawed reasoning paths (malgorithms) associated with incorrect answer choices, revealing significant challenges in counterfactual reasoning. The study highlights the need for improved LLMs in educational applications, particularly for AI-powered tutoring systems.

### 196. [Title:
          To Forget or Not? Towards Practical Knowledge Unlearning for Large Language Models](https://arxiv.org/pdf/2407.01920)
**Summary**: The paper introduces KnowUnDo, a benchmark for evaluating the effectiveness of knowledge unlearning in large language models (LLMs), focusing on the risk of erasing essential knowledge along with sensitive data. The authors propose MemFlex, a method that uses gradient information to selectively unlearn sensitive parameters without excessive loss of general knowledge, outperforming existing unlearning techniques in precision and retention.

### 197. [Title:
          OffsetBias: Leveraging Debiased Data for Tuning Evaluators](https://arxiv.org/pdf/2407.06551)
**Summary**: The paper identifies six types of biases in judge models used to evaluate generated responses from Large Language Models (LLMs) and introduces EvalBiasBench, a collection of test cases to assess these biases. The authors propose methods to construct a debiased dataset, OffsetBias, and demonstrate that fine-tuning on this dataset improves the robustness and performance of judge models in various evaluation scenarios.

### 198. [Title:
          Cactus: Towards Psychological Counseling Conversations using Cognitive Behavioral Theory](https://arxiv.org/pdf/2407.03103)
**Summary**: The paper introduces Cactus, a multi-turn dialogue dataset designed to simulate real-life psychological counseling sessions using Cognitive Behavioral Therapy (CBT). By creating diverse client personas and systematically applying CBT techniques, the dataset aims to improve the accessibility of counseling through large language models. Experimental results show that a model trained on Cactus, named Camel, outperforms others in counseling skills, indicating its potential as a counseling agent.

### 199. [Title:
          MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/pdf/2407.02483)
**Summary**: The paper introduces MMedAgent, the first multi-modal agent specifically designed for the medical field, which leverages a curated dataset of medical tools to select the most appropriate tool for various tasks. MMedAgent outperforms state-of-the-art open-source methods and even GPT-4 in medical tasks, demonstrating efficiency in tool selection and integration.

### 200. [Title:
          Unlocking the Potential of Model Merging for Low-Resource Languages](https://arxiv.org/pdf/2407.03994)
**Summary**: The paper introduces model merging as an alternative to the conventional continual pre-training and supervised fine-tuning approach for adapting large language models (LLMs) to low-resource languages. By combining models with distinct capabilities, the authors demonstrate that model merging can effectively equip LLMs with task-solving abilities without requiring additional training data in the target languages. Their experiments with Llama-2-7B show that this method outperforms the traditional CT-then-SFT approach, especially in scenarios with extremely limited

### 201. [Title:
          Self-training Language Models for Arithmetic Reasoning](https://arxiv.org/pdf/2407.08400)
**Summary**: The paper investigates the effectiveness of self-training language models for arithmetic reasoning without additional annotated data, using automated feedback to improve performance. It finds significant improvements in both offline and online self-training scenarios, with preference optimization methods outperforming traditional supervised training in online settings due to better stability and robustness.

### 202. [Title:
          Historical Ink: 19th Century Latin American Spanish Newspaper Corpus with LLM OCR Correction](https://arxiv.org/pdf/2407.12838)
**Summary**: The paper introduces a novel dataset of 19th-century Latin American newspaper texts, filling a critical gap in historical and linguistic research. Additionally, it presents a flexible framework using a Large Language Model for OCR error correction and linguistic surface form detection, which is applied to the new dataset, making it adaptable for various contexts and datasets.

### 203. [Title:
          NativQA: Multilingual Culturally-Aligned Natural Query for LLMs](https://arxiv.org/pdf/2407.09823)
**Summary**: The paper introduces NativQA, a scalable, language-independent framework designed to create culturally and regionally aligned QA datasets in native languages for evaluating and fine-tuning large language models (LLMs). The authors demonstrate the framework's efficacy by developing MultiNativQA, a multilingual dataset with ~64k QA pairs in seven languages, sourced from native speakers across nine regions. The dataset is used to benchmark LLMs and highlight the framework's utility in generating fine-tuning data for low

### 204. [Title:
          Knowledge-based Consistency Testing of Large Language Models](https://arxiv.org/pdf/2407.12830)
**Summary**: The paper introduces KonTest, an automated framework for evaluating the consistency and knowledge gaps in Large Language Models (LLMs) using a knowledge graph. KonTest identifies inconsistencies and knowledge gaps in LLMs, with a 19.2% error rate and a 16.5% knowledge gap across tested models. The framework also reduces knowledge gaps by 32.48% through a weighted model ensemble, and highlights GPT3.5's limitations in knowledge construction.

### 205. [Title:
          Are Large Language Models Capable of Generating Human-Level Narratives?](https://arxiv.org/pdf/2407.13248)
**Summary**: The paper examines the storytelling abilities of Large Language Models (LLMs) by analyzing narrative development and plot progression through story arcs, turning points, and affective dimensions. It finds that human-written stories are more suspenseful and diverse compared to LLM-generated stories, which tend to be homogeneously positive and lack tension. The study concludes that LLMs generally fall short in narrative reasoning and suggests that integrating discourse features can significantly improve LLM storytelling.

### 206. [Title:
          When Can Transformers Count to n?](https://arxiv.org/pdf/2407.15160)
**Summary**: The paper investigates whether transformer-based language models can perform simple counting tasks, specifically counting the occurrences of a token in a string. It demonstrates that transformers with state dimensions linear in context length can solve these tasks, but scalability beyond this limit is theoretically and empirically shown to be impossible. The study highlights the limitations of transformers in handling basic counting tasks and underscores the need for a deeper understanding of their capabilities.

### 207. [Title:
          Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together](https://arxiv.org/pdf/2407.10930)
**Summary**: The paper introduces a novel approach called BetterTogether that combines fine-tuning and prompt optimization to enhance the performance of modular NLP pipelines, particularly in scenarios where intermediate labels or gradient flow are absent. By alternating between optimizing language model weights and prompt templates, the method achieves significant improvements in downstream task metrics, outperforming individual optimizations by up to 60% and 6% on average across various models and tasks.

### 208. [Title:
          Knowledge Mechanisms in Large Language Models: A Survey and Perspective](https://arxiv.org/pdf/2407.15017)
**Summary**: The paper surveys knowledge mechanisms in Large Language Models (LLMs), categorizing them into knowledge utilization and evolution. It explores how LLMs memorize, comprehend, apply, and create knowledge, as well as how knowledge evolves within individual and group models. The study also addresses the fragility of parametric knowledge and hypothesizes about potential "dark knowledge" challenges, aiming to guide future research on understanding and improving LLMs.

### 209. [Title:
          A Comparison of Language Modeling and Translation as Multilingual Pretraining Objectives](https://arxiv.org/pdf/2407.15489)
**Summary**: The paper compares multilingual pretraining objectives, focusing on language modeling and translation, in a controlled environment to establish best practices for NLP research. It finds that the choice of pretraining objective is influenced by the model architecture and that multilingual translation is highly effective under suitable conditions. The study provides code, data, and model weights for reproducibility.

### 210. [Title:
          Think-on-Graph 2.0: Deep and Faithful Large Language Model Reasoning with Knowledge-guided Retrieval Augmented Generation](https://arxiv.org/pdf/2407.10805)
**Summary**: The paper introduces Think-on-Graph 2.0 (ToG-2), a hybrid retrieval-augmented generation framework that enhances large language models by integrating knowledge graphs and document retrieval for deep and accurate reasoning. ToG-2 iteratively retrieves information from both structured and unstructured sources, improving context and graph retrieval, and demonstrates superior performance on knowledge-intensive datasets, even enabling smaller models to match the reasoning capabilities of larger models like GPT-3.5.

### 211. [Title:
          Counter Turing Test ($CT^2$): Investigating AI-Generated Text Detection for Hindi -- Ranking LLMs based on Hindi AI Detectability Index ($ADI_{hi}$)](https://arxiv.org/pdf/2407.15694)
**Summary**: The paper investigates the detection of AI-generated text in Hindi, evaluating the proficiency of 26 Large Language Models (LLMs) in generating Hindi text and assessing the effectiveness of five AGTD techniques. It introduces the AI-generated news article in Hindi dataset and proposes the Hindi AI Detectability Index ($ADI_{hi}$) to measure the detectability of AI-generated text, providing insights into the evolving landscape of AI-generated eloquence in Hindi.

### 212. [Title:
          Revisiting Who's Harry Potter: Towards Targeted Unlearning from a Causal Intervention Perspective](https://arxiv.org/pdf/2407.16997)
**Summary**: The paper revisits the Who's Harry Potter (WHP) method for Large Language Model (LLM) unlearning, proposing a new task of targeted unlearning where only specific information about a given target is removed from the model. The authors introduce a causal intervention framework to model the unlearning process, demonstrating that their approach, without explicit optimization for specific criteria, performs competitively across various datasets.

### 213. [Title:
          ReAttention: Training-Free Infinite Context with Finite Attention Scope](https://arxiv.org/pdf/2407.15176)
**Summary**: The paper introduces **ReAttention**, a training-free method that allows Large Language Models (LLMs) to handle infinite context lengths with a finite attention scope, overcoming the limitations imposed by the self-attention mechanism. By performing position-agnostic top-$k$ attention before the standard position-aware self-attention, ReAttention enables LLMs to support extremely long contexts, such as up to 1M tokens, and even extends the context length of models like LLaMA3.2-

### 214. [Title:
          A Novel Metric for Measuring the Robustness of Large Language Models in Non-adversarial Scenarios](https://arxiv.org/pdf/2408.01963)
**Summary**: The paper introduces a novel metric to measure the robustness of large language models in non-adversarial scenarios, focusing on their insensitivity to meaning-preserving variations in input. By evaluating multiple models on benchmark datasets with naturally occurring perturbations and semantically equivalent paraphrases, the authors demonstrate the effectiveness of their proposed metric in assessing model robustness.

### 215. [Title:
          CMR Scaling Law: Predicting Critical Mixture Ratios for Continual Pre-training of Language Models](https://arxiv.org/pdf/2407.17467)
**Summary**: The paper introduces the Critical Mixture Ratio (CMR) scaling law, which predicts the optimal balance between general and domain-specific data for continual pre-training of large language models (LLMs) to prevent catastrophic forgetting and enhance domain-specific performance. The study reveals a power-law relationship between loss, mixture ratio, and training tokens, providing a practical guideline for efficient and effective LLM training in specialized domains.

### 216. [Title:
          DYNAMICQA: Tracing Internal Knowledge Conflicts in Language Models](https://arxiv.org/pdf/2407.17023)
**Summary**: The paper introduces DynamicQA, a novel dataset designed to study intra-memory conflicts in Language Models (LMs), where conflicting knowledge within the model's parameters affects its ability to integrate relevant context. The dataset includes facts with temporal and disputable dynamics, allowing for the analysis of how LMs handle knowledge conflicts. The study finds that LMs exhibit more intra-memory conflict with dynamic facts and struggle to update these facts with new context, highlighting challenges for retrieval-augmented generation.

### 217. [Title:
          Quantifying the Role of Textual Predictability in Automatic Speech Recognition](https://arxiv.org/pdf/2407.16537)
**Summary**: The paper introduces a novel method to quantify the impact of textual predictability on automatic speech recognition (ASR) errors, represented by a single parameter \( k \). This approach is used to compare the performance of Wav2Vec 2.0 and hybrid ASR models, revealing that Wav2Vec 2.0 effectively utilizes textual context despite lacking an explicit language model. The study also highlights that poor performance on African-American English is primarily due to acoustic-phonetic modeling issues rather

### 218. [Title:
          S2-Attention: Hardware-Aware Context Sharding Among Attention Heads](https://arxiv.org/pdf/2407.17678)
**Summary**: The paper introduces S2-Attention, a Triton library for optimizing sparse attention mechanisms, which allows for hardware-aware context sharding across attention heads. This approach enables significant wall-clock speedups (up to 25.3X) over dense attention baselines like FlashAttention-2, while maintaining strong downstream performance and retrieval accuracy, especially at large context lengths. The library offers customizable APIs for integration into existing frameworks like Megatron and vLLM.

### 219. [Title:
          Adaptive Contrastive Decoding in Retrieval-Augmented Generation for Handling Noisy Contexts](https://arxiv.org/pdf/2408.01084)
**Summary**: The paper introduces Adaptive Contrastive Decoding (ACD), an extension of contrastive decoding methods designed to handle noisy contexts in retrieval-augmented generation for large language models (LLMs). ACD enhances the robustness of LLMs in open-domain question answering by effectively leveraging contextual information while minimizing the impact of irrelevant or noisy data, outperforming baseline methods in terms of accuracy and reliability.

### 220. [Title:
          Investigating Critical Period Effects in Language Acquisition through Neural Language Models](https://arxiv.org/pdf/2407.19325)
**Summary**: The study investigates whether critical period effects in language acquisition are innate or result from experience-driven neural stabilization by using neural language models. The models, which lack innate maturational stages, do not exhibit critical period effects when exposed to a second language later in "learning." This suggests that critical period effects may be due to innate mechanisms rather than solely statistical learning. The study also demonstrates that introducing a regularizer during training can simulate maturational changes, indicating that additional factors beyond L1 learning are needed

### 221. [Title:
          Optimal and efficient text counterfactuals using Graph Neural Networks](https://arxiv.org/pdf/2408.01969)
**Summary**: The paper introduces a framework for generating counterfactual explanations in NLP models using Graph Neural Networks, which creates semantically edited inputs that alter model predictions. The framework is tested on binary sentiment and topic classification tasks, demonstrating that it produces contrastive, fluent, and minimal edits while being significantly faster than existing methods.

### 222. [Title:
          Counterfactuals As a Means for Evaluating Faithfulness of Attribution Methods in Autoregressive Language Models](https://arxiv.org/pdf/2408.11252)
**Summary**: The paper introduces a novel method for evaluating the faithfulness of attribution methods in autoregressive language models by using counterfactual generation. This approach addresses the challenge of creating out-of-distribution inputs in traditional faithfulness evaluations, ensuring that the generated counterfactuals are both fluent and in-distribution, thereby enhancing the reliability of the evaluation process.

### 223. [Title:
          Mathfish: Evaluating Language Model Math Reasoning via Grounding in Educational Curricula](https://arxiv.org/pdf/2408.04226)
**Summary**: The paper introduces MathFish, a framework for evaluating language models' mathematical reasoning by grounding it in K-12 educational standards. It presents two datasets: one detailing math skills and concepts, and another with math problems labeled according to these standards. The study finds that while LMs can predict standards related to problems, they often fail to fully align with the ground truth, highlighting the need for careful evaluation when using LMs in educational content generation.

### 224. [Title:
          HySem: A context length optimized LLM pipeline for unstructured tabular extraction](https://arxiv.org/pdf/2408.09434)
**Summary**: The paper introduces HySem, a pipeline designed to extract and semantically represent unstructured tabular data from HTML tables, particularly useful for regulatory compliance in the pharmaceutical industry. HySem optimizes context length to enhance accuracy and addresses limitations of large language models, offering competitive performance against models like OpenAI GPT-4 while being cost-effective and suitable for small and medium enterprises.

### 225. [Title:
          MAG-SQL: Multi-Agent Generative Approach with Soft Schema Linking and Iterative Sub-SQL Refinement for Text-to-SQL](https://arxiv.org/pdf/2408.07930)
**Summary**: The paper introduces MAG-SQL, a multi-agent generative approach for the Text-to-SQL task, addressing challenges in complex schema and difficult questions by incorporating soft schema linking and iterative Sub-SQL refinement. The framework uses entity-based column selection and a novel decomposition method, along with an iterative generation module with oversight, to improve performance. Evaluations on BIRD and Spider benchmarks show significant accuracy improvements over baseline models.

### 226. [Title:
          Rater Cohesion and Quality from a Vicarious Perspective](https://arxiv.org/pdf/2408.08411)
**Summary**: The paper investigates the use of vicarious annotation to reduce disagreement in human feedback for AI systems, particularly in politically charged contexts. It examines how rater cohesion and quality metrics, influenced by political affiliations and demographics, impact the agreement among raters both personally and vicariously. The study employs CrowdTruth's rater quality metrics to assess the consistency and reliability of annotations across different groups of raters.

### 227. [Title:
          Correcting FLORES Evaluation Dataset for Four African Languages](https://arxiv.org/pdf/2409.00626)
**Summary**: The paper addresses inconsistencies and inaccuracies in the FLORES evaluation dataset for Hausa, Northern Sotho, Xitsonga, and isiZulu by implementing corrections identified through a meticulous review by native speakers. The authors argue that these improvements enhance the dataset's linguistic accuracy and reliability, thereby supporting more effective NLP evaluations, and recommend greater native speaker involvement in future translation projects.

### 228. [Title:
          Advancing Adversarial Suffix Transfer Learning on Aligned Large Language Models](https://arxiv.org/pdf/2408.14866)
**Summary**: The paper introduces DeGCG, a two-stage transfer learning framework for improving the efficiency of adversarial suffix generation in large language models (LLMs). By decoupling the search process into pre-searching and post-searching stages, DeGCG enhances suffix transferability across different models and datasets. The interleaved variant, i-DeGCG, further accelerates the search process by leveraging self-transferability, achieving significant improvements in adversarial success rates (ASRs) on Llama2-chat-

### 229. [Title:
          Evidence-backed Fact Checking using RAG and Few-Shot In-Context Learning with LLMs](https://arxiv.org/pdf/2408.12060)
**Summary**: The paper introduces an automated fact-checking system that leverages the Averitec dataset to verify online claims, employing a Retrieve and Generate (RAG) pipeline and few-shot In-Context Learning with large language models (LLMs). The system significantly improves upon the baseline with an 'Averitec' score of 0.33, demonstrating a 22% absolute improvement in performance.

### 230. [Title:
          UPCS: Unbiased Persona Construction for Dialogue Generation](https://arxiv.org/pdf/2409.05257)
**Summary**: The paper introduces the UPCS framework, designed to create unbiased persona profiles for dialogue generation by categorizing character descriptions into eight dimensions with bias mitigation strategies. Experimental results show that UPCS outperforms existing methods in accuracy, diversity, bias elimination, and user satisfaction, significantly improving the reliability of narrative systems.

### 231. [Title:
          Native vs Non-Native Language Prompting: A Comparative Analysis](https://arxiv.org/pdf/2409.07054)
**Summary**: The study compares native and non-native language prompting strategies across 11 NLP tasks using Arabic datasets, involving 197 experiments with three large language models. The findings indicate that non-native prompts generally yield better performance, followed by mixed and native prompts, highlighting the importance of prompt language in eliciting effective responses from LLMs.

### 232. [Title:
          Deconfounded Causality-aware Parameter-Efficient Fine-Tuning for Problem-Solving Improvement of LLMs](https://arxiv.org/pdf/2409.02686)
**Summary**: The paper investigates the reasoning limitations of Large Language Models (LLMs) and proposes a novel parameter-efficient fine-tuning method called Deconfounded Causal Adaptation (DCA) to enhance their problem-solving capabilities. By formulating the reasoning process into a causal framework and visualizing the text generation at different levels, the authors demonstrate that DCA significantly improves LLM performance across benchmarks with minimal tunable parameters, achieving better or comparable results to other fine-tuning methods.

### 233. [Title:
          Propaganda to Hate: A Multimodal Analysis of Arabic Memes with Multi-Agent LLMs](https://arxiv.org/pdf/2409.07246)
**Summary**: The paper investigates the intersection of propaganda and hate in Arabic memes using a multi-agent large language model (LLM) approach. It extends the existing propagandistic meme dataset with hate labels and finds a significant association between propaganda and hate in memes. The study provides a baseline for future research and will make its resources publicly available.

### 234. [Title:
          The Faetar Benchmark: Speech Recognition in a Very Under-Resourced Language](https://arxiv.org/pdf/2409.08103)
**Summary**: The paper introduces the Faetar Automatic Speech Recognition Benchmark, a challenging dataset for low-resource speech recognition, focusing on the under-resourced Franco-Provençal variety spoken in Italy. The corpus, consisting of 5 hours of transcribed and 20 hours of unlabelled noisy field recordings, tests the limits of current multilingual models, achieving a baseline phone error rate of 30.4% after continued pre-training.

### 235. [Title:
          WinoPron: Revisiting English Winogender Schemas for Consistency, Coverage, and Grammatical Case](https://arxiv.org/pdf/2409.05653)
**Summary**: The paper addresses issues in the Winogender Schemas dataset, which is used to evaluate gender bias in coreference resolution, by identifying problems such as inconsistent treatment of pronominal forms and template violations. The authors introduce WinoPron, a revised dataset, and use it to evaluate coreference resolution systems, finding that accusative pronouns pose greater challenges. They also propose a new method to assess pronominal bias beyond binary distinctions, revealing variations in bias across different pronoun forms.

### 236. [Title:
          IndicVoices-R: Unlocking a Massive Multilingual Multi-speaker Speech Corpus for Scaling Indian TTS](https://arxiv.org/pdf/2409.05356)
**Summary**: The paper introduces IndicVoices-R (IV-R), a massive multilingual Indian TTS dataset derived from ASR data, featuring 1,704 hours of high-quality speech from 10,496 speakers across 22 languages. The authors demonstrate improved zero-shot speaker generalization by fine-tuning an English pre-trained model on a combined dataset of IndicTTS and IV-R, addressing the scarcity of high-quality TTS data for Indian languages.

### 237. [Title:
          FoodPuzzle: Developing Large Language Model Agents as Flavor Scientists](https://arxiv.org/pdf/2409.12832)
**Summary**: The paper introduces FoodPuzzle, a benchmark for developing large language model agents to generate hypotheses in flavor science, addressing the need for rapid innovation in the food industry. By integrating in-context learning and retrieval augmented techniques, the proposed Scientific Agent significantly outperforms traditional methods in predicting flavor profiles, suggesting a transformative approach to flavor development.

### 238. [Title:
          Pretraining Data Detection for Large Language Models: A Divergence-based Calibration Method](https://arxiv.org/pdf/2409.14781)
**Summary**: The paper introduces a divergence-based calibration method for pretraining data detection in large language models (LLMs), addressing limitations of the Min-K% Prob method by using cross-entropy to measure divergence between token probability and frequency distributions. The proposed method significantly outperforms existing techniques, as demonstrated by experiments on both English and a new Chinese-language benchmark, PatentMIA.

### 239. [Title:
          Enhancing adversarial robustness in Natural Language Inference using explanations](https://arxiv.org/pdf/2409.07423)
**Summary**: The paper investigates enhancing the robustness of Natural Language Inference (NLI) models against adversarial attacks by using natural language explanations as a defense strategy. By fine-tuning a classifier on explanations rather than premise-hypothesis pairs, the authors achieve improved robustness compared to traditional methods. They also explore the correlation between language generation metrics and human perception to validate the semantic validity of generated explanations, ensuring a more robust NLI model.

### 240. [Title:
          Obliviate: Neutralizing Task-agnostic Backdoors within the Parameter-efficient Fine-tuning Paradigm](https://arxiv.org/pdf/2409.14119)
**Summary**: The paper introduces Obliviate, a defense mechanism for neutralizing task-agnostic backdoors in parameter-efficient fine-tuning (PEFT) of large language models. The proposed techniques amplify benign neurons and penalize trigger tokens, effectively reducing the success rate of state-of-the-art backdoor attacks by 83.6%. Obliviate also demonstrates robust defense against task-specific backdoors and adaptive attacks.

### 241. [Title:
          Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations](https://arxiv.org/pdf/2409.14399)
**Summary**: The paper introduces PC-CRS, a method designed to enhance the credibility of explanations in conversational recommender systems (CRS) by integrating credibility-aware persuasive strategies and post-hoc self-reflection. The approach aims to balance persuasion with trustworthiness, demonstrated through experiments showing improved credibility and potential to enhance recommendation accuracy.

### 242. [Title:
          IDGen: Item Discrimination Induced Prompt Generation for LLM Evaluation](https://arxiv.org/pdf/2409.18892)
**Summary**: The paper introduces IDGen, a framework for generating discriminative prompts to evaluate Large Language Models (LLMs) based on Item Discrimination (ID) theory from educational assessment. IDGen aims to create challenging and specific prompts that reveal performance differences between models, with a self-correct mechanism and predictive models for prompt discrimination and difficulty. The generated data is shown to be more challenging and discriminative than previous methods, with plans to release a dataset of over 3,000 prompts for LLM

### 243. [Title:
          PEAR: Position-Embedding-Agnostic Attention Re-weighting Enhances Retrieval-Augmented Generation with Zero Inference Overhead](https://arxiv.org/pdf/2409.19745)
**Summary**: The paper introduces PEAR, a method that enhances the context awareness of large language models (LLMs) in retrieval-augmented generation (RAG) tasks without any inference overhead. PEAR identifies and re-weights attention heads that suppress context awareness, optimizing their impact through learnable coefficients, and is position-embedding agnostic, making it more versatile and efficient compared to existing methods.

### 244. [Title:
          Visual Question Decomposition on Multimodal Large Language Models](https://arxiv.org/pdf/2409.19339)
**Summary**: The paper investigates the question decomposition capabilities of Multimodal Large Language Models (MLLMs) and introduces a systematic evaluation framework to assess the quality of decomposed sub-questions. It identifies limitations in current MLLMs and proposes a finetuning dataset, DecoVQA+, and an efficient finetuning pipeline to enhance selective decomposition, leading to improved sub-question quality and higher accuracy on Visual Question Answering (VQA) benchmarks.

### 245. [Title:
          From Code to Correctness: Closing the Last Mile of Code Generation with Hierarchical Debugging](https://arxiv.org/pdf/2410.01215)
**Summary**: The paper introduces Multi-Granularity Debugger (MGDebugger), a hierarchical code debugger that addresses subtle errors in generated code by isolating and resolving bugs at multiple levels of granularity. MGDebugger outperforms existing systems, achieving significant improvements in accuracy and repair success rates, and effectively handles bugs across various categories and difficulty levels.

### 246. [Title:
          KV-Compress: Paged KV-Cache Compression with Variable Compression Rates per Attention Head](https://arxiv.org/pdf/2410.00161)
**Summary**: The paper introduces KV-Compress, a novel method for compressing key-value (KV) cache in large language models (LLMs) to efficiently support long-context inference. By evicting contiguous KV blocks within a PagedAttention framework, KV-Compress achieves up to 8x compression rates with minimal performance impact and up to 64x compression while retaining over 90% of full-cache performance. This method significantly enhances throughput by enabling larger decoding batches.<｜end▁of▁sentence｜>

### 247. [Title:
          Aligning with Logic: Measuring, Evaluating and Improving Logical Consistency in Large Language Models](https://arxiv.org/pdf/2410.02205)
**Summary**: The paper introduces a framework to measure and improve the logical consistency of Large Language Models (LLMs), which is crucial for their reliability and trustworthiness. It proposes three fundamental proxies—transitivity, commutativity, and negation invariance—to quantify logical consistency and evaluates various LLMs using these measures. The study also presents a data refinement technique to enhance logical consistency while maintaining alignment with human preferences, demonstrating its impact on LLM-based decision-making systems.

### 248. [Title:
          What the Harm? Quantifying the Tangible Impact of Gender Bias in Machine Translation with a Human-centered Study](https://arxiv.org/pdf/2410.00545)
**Summary**: The paper investigates the tangible impact of gender bias in machine translation (MT) through a human-centered study involving 90 participants who post-edited MT outputs to ensure correct gender translation. The study reveals that feminine post-editing requires significantly more effort, time, and financial cost compared to masculine translations, highlighting a quality of service gap. The findings suggest that current bias measurements inadequately capture these disparities, advocating for more human-centered approaches to assess the societal impact of MT bias.

### 249. [Title:
          MetaMetrics: Calibrating Metrics For Generation Tasks Using Human Preferences](https://arxiv.org/pdf/2410.02381)
**Summary**: The paper introduces MetaMetrics, a calibrated meta-metric designed to evaluate generation tasks by optimizing the combination of existing metrics to better align with human preferences across different modalities. It demonstrates effectiveness in both language and vision tasks, showing significant benefits in multilingual and multi-domain scenarios, and is easily integrable into various applications, making it a powerful tool for improving the evaluation of generation tasks.

### 250. [Title:
          OpenMathInstruct-2: Accelerating AI for Math with Massive Open-Source Instruction Data](https://arxiv.org/pdf/2410.01560)
**Summary**: The paper introduces OpenMathInstruct-2, a massive open-source dataset for mathematical reasoning, consisting of 14M question-solution pairs, significantly larger than previous datasets. Through ablation experiments, the authors identify key factors affecting SFT performance, such as solution format and question diversity, and demonstrate that finetuning with OpenMathInstruct-2 improves model performance on the MATH benchmark by 15.9%. The dataset, code, and models are released under a permissive

### 251. [Title:
          Better Instruction-Following Through Minimum Bayes Risk](https://arxiv.org/pdf/2410.02902)
**Summary**: The paper explores using Minimum Bayes Risk (MBR) decoding with reference-based LLM judges to improve the performance of instruction-following LLMs, finding that it outperforms other decoding methods. Additionally, the authors investigate iterative self-training on MBR-decoded outputs, which leads to significant performance gains, often matching or exceeding the performance of base models with MBR decoding.

### 252. [Title:
          LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations](https://arxiv.org/pdf/2410.02707)
**Summary**: The paper investigates the internal representations of large language models (LLMs) and finds that these models encode more information about the truthfulness of their outputs than previously recognized. It highlights that truthfulness information is concentrated in specific tokens, enhancing error detection, but also shows that such detectors do not generalize across datasets, indicating that truthfulness encoding is multifaceted. The study further reveals that LLMs can encode correct answers internally but still generate incorrect outputs, suggesting a discrepancy between internal knowledge and external behavior.<｜end▁of▁sentence｜>

### 253. [Title:
          Contextual Document Embeddings](https://arxiv.org/pdf/2410.02525)
**Summary**: The paper introduces contextual document embeddings, arguing that traditional dense embeddings are insufficiently contextual for retrieval tasks. It proposes two methods: a contrastive learning objective that incorporates neighboring documents and a contextual architecture that encodes neighbor information. These methods outperform traditional biencoders, achieving state-of-the-art results on the MTEB benchmark without complex techniques like hard negative mining or large batch sizes.

### 254. [Title:
          Measuring and Improving Persuasiveness of Large Language Models](https://arxiv.org/pdf/2410.02653)
**Summary**: The paper introduces PersuasionBench and PersuasionArena, the first large-scale benchmarks for measuring the persuasiveness of large language models (LLMs). It finds that while larger models tend to be more persuasive, smaller models can be significantly improved through targeted training, challenging the assumption that scale alone determines persuasiveness. The study highlights the need for more comprehensive metrics to regulate AI's societal impact, beyond computational power.

### 255. [Title:
          Efficient Model-Agnostic Multi-Group Equivariant Networks](https://arxiv.org/pdf/2310.09675)
**Summary**: The paper introduces efficient model-agnostic equivariant network designs for scenarios with multiple input groups or large product groups acting on a single input. It proposes novel fusion layers, called IS layers, which satisfy invariance-symmetry constraints and are shown to be universal approximators. The designs are tested on various tasks, demonstrating robustness and competitive performance with reduced computational cost compared to existing methods.

### 256. [Title:
          Reconstruct Your Previous Conversations! Comprehensively Investigating Privacy Leakage Risks in Conversations with GPT Models](https://arxiv.org/pdf/2402.02987)
**Summary**: The paper investigates privacy risks in conversations with GPT models, introducing a Conversation Reconstruction Attack that can leak previous conversation contents through malicious prompts. Despite GPT-4's resilience, advanced attacks show significant privacy leakage across all models, and existing defense mechanisms are found to be ineffective. The study emphasizes the need for stronger safeguards to protect user privacy in interactions with GPT models.

### 257. [Title:
          READ: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling](https://arxiv.org/pdf/2312.06950)
**Summary**: The paper introduces READ, a novel recurrent adapter framework for parameter-efficient transfer learning in low-resource video-language modeling. READ incorporates temporal modeling through recurrent computation and uses a Partial Video-Language Alignment (PVLA) objective to preserve task-related information, outperforming existing fine-tuning strategies on multiple benchmarks.

### 258. [Title:
          EffiBench: Benchmarking the Efficiency of Automatically Generated Code](https://arxiv.org/pdf/2402.02037)
**Summary**: The paper introduces EffiBench, a benchmark designed to evaluate the efficiency of code generated by large language models (LLMs) in solving 1,000 efficiency-critical coding problems from LeetCode. The benchmark compares the performance of 42 LLMs against human-written canonical solutions, revealing that LLM-generated code generally exhibits worse efficiency, with GPT-4's code, for instance, averaging 3.12 times longer execution times compared to human solutions.<｜end▁of▁sentence｜>

### 259. [Title:
          Typing to Listen at the Cocktail Party: Text-Guided Target Speaker Extraction](https://arxiv.org/pdf/2310.07284)
**Summary**: The paper introduces LLM-TSE, a novel text-guided target speaker extraction method that leverages LLaMA 2 to process user-typed text for semantic cues, addressing privacy concerns and reducing reliance on voiceprints. Experimental results demonstrate competitive performance with text-based cues alone and achieve a new state-of-the-art when combined with pre-registered cues, marking the first integration of large language models with TSE and offering a versatile, privacy-conscious solution to the cocktail party problem

### 260. [Title:
          Rethinking the Role of Proxy Rewards in Language Model Alignment](https://arxiv.org/pdf/2402.03469)
**Summary**: The paper investigates the role of proxy rewards in aligning Large Language Models (LLMs) with human values through "reverse reward engineering," where interpretable features are used to create a white-box reward function. The study finds that replicating the gold reward signal requires responses to be relevant and sufficiently long for open-ended questions, and consistent for closed-ended questions. The resulting models show competitive performance in alignment benchmarks, suggesting the white-box reward could serve as a strong baseline for LLM alignment without

### 261. [Title:
          Decoding Intelligence: A Framework for Certifying Knowledge Comprehension in LLMs](https://arxiv.org/pdf/2402.15929)
**Summary**: The paper introduces a novel framework for certifying the knowledge comprehension capabilities of Large Language Models (LLMs) with formal probabilistic guarantees. It provides high-confidence bounds on the probability of correct answers to knowledge comprehension prompts, leveraging knowledge graphs like Wikidata5m, and demonstrates that model performance improves with increased size.

### 262. [Title:
          Comparing large language models and human programmers for generating programming code](https://arxiv.org/pdf/2403.00894)
**Summary**: The study evaluates the performance of seven large language models, with GPT-4 significantly outperforming others in generating programming code across various tasks and languages. GPT-4, using optimized prompt strategies, surpasses 85% of human participants in coding contests and shows strong code translation and error correction abilities, suggesting its potential as a reliable assistant in software development.

### 263. [Title:
          Successfully Guiding Humans with Imperfect Instructions by Highlighting Potential Errors and Suggesting Corrections](https://arxiv.org/pdf/2402.16973)
**Summary**: The paper introduces HEAR, a system that guides humans in simulated residential environments by highlighting potential errors in its instructions and suggesting corrections, despite generating imperfect instructions. Evaluation with 80 users shows that HEAR significantly improves success rates and reduces errors by 13% and 29%, respectively, compared to systems that only provide instructions, demonstrating the practical benefits of uncertainty communication in complex decision-making tasks.

### 264. [Title:
          Creative Beam Search: LLM-as-a-Judge For Improving Response Generation](https://arxiv.org/pdf/2405.00099)
**Summary**: The paper introduces Creative Beam Search, a method that combines Diverse Beam Search with a Large Language Model (LLM) acting as a judge to enhance response generation in creative tasks. The approach aims to mimic human-like intentionality and creativity in machine responses, outperforming standard sampling techniques. The study highlights the importance of a validation step to complement the generation process, ensuring higher quality outputs.

### 265. [Title:
          A Toolbox for Surfacing Health Equity Harms and Biases in Large Language Models](https://arxiv.org/pdf/2403.12025)
**Summary**: The paper introduces a toolbox and methodologies for identifying biases in large language models (LLMs) that could harm health equity, focusing on long-form responses to medical questions. It includes a multifactorial framework for human assessment and a dataset called EquityMedQA, enriched with adversarial queries, to evaluate biases in the Med-PaLM 2 LLM. The study highlights the importance of diverse assessment methods and rater backgrounds to effectively surface biases, emphasizing the need for comprehensive approaches to ensure equitable healthcare outcomes

### 266. [Title:
          "I Like Sunnie More Than I Expected!": Exploring User Expectation and Perception of an Anthropomorphic LLM-based Conversational Agent for Well-Being Support](https://arxiv.org/pdf/2405.13803)
**Summary**: The study investigates how users' expectations and perceptions of two LLM-based mental well-being support systems differ, with one system (Sunnie) featuring an anthropomorphic design. Results indicate that both systems exceeded users' expectations in utility, but Sunnie, with its anthropomorphic elements, significantly outperformed the non-anthropomorphic system in fostering relational warmth, suggesting the potential of such designs in enhancing mental health support.

### 267. [Title:
          Many-Shot In-Context Learning in Multimodal Foundation Models](https://arxiv.org/pdf/2405.09798)
**Summary**: The paper investigates the performance of multimodal foundation models, such as GPT-4o and Gemini 1.5 Pro, in many-shot in-context learning (ICL) across various domains and tasks. It finds that many-shot ICL, with up to nearly 2,000 demonstrating examples, significantly outperforms few-shot ICL, with Gemini 1.5 Pro showing log-linear improvement. The study also highlights differences between open and closed multimodal models and explores

### 268. [Title:
          Self-Play Preference Optimization for Language Model Alignment](https://arxiv.org/pdf/2405.00675)
**Summary**: The paper introduces Self-Play Preference Optimization (SPPO), a novel method for aligning language models by treating the alignment problem as a constant-sum two-player game. SPPO iteratively updates policies to approximate the Nash equilibrium, using a new objective function that is both theoretically grounded and effective in practice. Experiments demonstrate that SPPO, using only 60k prompts and a pre-trained preference model, achieves state-of-the-art performance in various benchmarks, outperforming existing methods like D

### 269. [Title:
          SpinQuant: LLM quantization with learned rotations](https://arxiv.org/pdf/2405.16406)
**Summary**: The paper introduces SpinQuant, a novel approach to LLM quantization that uses learned rotation matrices to enhance accuracy, particularly in the presence of outliers. By applying 4-bit quantization to weights, activations, and the KV cache, SpinQuant significantly reduces the accuracy gap in zero-shot reasoning tasks compared to other quantization methods, demonstrating superior performance over existing techniques like LLM-QAT, SmoothQuant, and QuaRot.

### 270. [Title:
          Implicit Multimodal Alignment: On the Generalization of Frozen LLMs to Multimodal Inputs](https://arxiv.org/pdf/2405.16700)
**Summary**: The paper investigates how frozen Large Language Models (LLMs) generalize to multimodal inputs, finding that perceptual and textual tokens are implicitly aligned within the model architecture, a phenomenon termed Implicit Multimodal Alignment (IMA). This alignment is linked to the model's architectural design and positively correlates with task performance, while negatively correlating with hallucinations. The study also proposes methods to reduce inference costs and compress models by leveraging the stability of perceptual tokens and shared weights across tasks.

### 271. [Title:
          Efficient Prompting for LLM-based Generative Internet of Things](https://arxiv.org/pdf/2406.10382)
**Summary**: The paper introduces a LLM-based Generative IoT (GIoT) system designed for local network deployment, addressing the limitations of open-source LLMs by employing prompt engineering and modular design to enhance performance. The system's effectiveness is demonstrated through a Table Question Answering task, showing competitive results compared to state-of-the-art LLMs, and highlighting its extensibility to new tasks without additional training.

### 272. [Title:
          Task Arithmetic can Mitigate Synthetic-to-Real Gap in Automatic Speech Recognition](https://arxiv.org/pdf/2406.02925)
**Summary**: The paper introduces a method using task vector arithmetic to mitigate the synthetic-to-real gap in automatic speech recognition (ASR) models, which improves performance when fine-tuning on synthetic data. The proposed SYN2REAL task vector achieves a 10.03% reduction in word error rate on the SLURP dataset and demonstrates enhanced adaptability to multiple text domains through averaging task vectors from real speeches.

### 273. [Title:
          mDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/pdf/2406.11839)
**Summary**: The paper introduces mDPO, a novel approach to conditional preference optimization for multimodal large language models (LLMs), addressing the issue of unconditional preference in multimodal scenarios. By incorporating image preferences and introducing a reward anchor, mDPO significantly enhances model performance, particularly in reducing hallucination, as demonstrated across various benchmarks.

### 274. [Title:
          "You Gotta be a Doctor, Lin": An Investigation of Name-Based Bias of Large Language Models in Employment Recommendations](https://arxiv.org/pdf/2406.12232)
**Summary**: The study investigates biases in employment recommendations made by Large Language Models (LLMs) like GPT-3.5-Turbo and Llama 3-70B-Instruct, finding a preference for candidates with White female-sounding names across various occupations. Salary recommendations also show significant variation based on name-based race and gender, with discrepancies up to 5% among equally qualified candidates, highlighting the need for further scrutiny of LLM-powered systems.

### 275. [Title:
          On Efficient Language and Vision Assistants for Visually-Situated Natural Language Understanding: What Matters in Reading and Reasoning](https://arxiv.org/pdf/2406.11823)
**Summary**: The paper investigates the challenges of creating efficient vision-language models for visually-situated natural language understanding, focusing on balancing model size with computational demands. By optimizing dataset formulation, vision modules, and supervision techniques, the study achieves improved inference throughput and performance across various model sizes, from 160M to 13B parameters. The research will be fully open-sourced to promote transparency and reproducibility.

### 276. [Title:
          WellDunn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions](https://arxiv.org/pdf/2406.12058)
**Summary**: The paper "WellDunn: On the Robustness and Explainability of Language Models and Large Language Models in Identifying Wellness Dimensions" evaluates the robustness and explainability of language models (LMs) and large language models (LLMs) in mental health applications, focusing on their ability to identify wellness dimensions. The study reveals that despite their advanced capabilities, models like GPT-3.5/4 and MedAlpaca underperform in terms of both performance and explanation fidelity, with significant discrepancies between

### 277. [Title:
          ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/pdf/2407.01449)
**Summary**: The paper introduces ColPali, a novel document retrieval model that efficiently leverages Vision Language Models to generate high-quality contextualized embeddings from document images, outperforming existing systems in retrieval tasks. The authors also present ViDoRe, a benchmark for evaluating visually rich document retrieval, highlighting the limitations of current methods and the advantages of ColPali's architecture.

### 278. [Title:
          BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions](https://arxiv.org/pdf/2406.15877)
**Summary**: The paper introduces BigCodeBench, a benchmark designed to evaluate the ability of Large Language Models (LLMs) to generate code for challenging and practical tasks by invoking multiple function calls from diverse libraries across various domains. The benchmark includes 1,140 fine-grained tasks with rigorous testing to assess LLMs' performance, revealing that current models struggle with complex instructions and precise function calls, achieving only up to 60% accuracy compared to human performance of 97%.

### 279. [Title:
          Unlocking Continual Learning Abilities in Language Models](https://arxiv.org/pdf/2406.17245)
**Summary**: The paper introduces MIGU, a rehearsal-free and task-label-free method for continual learning in language models, which updates model parameters based on the magnitude of outputs in linear layers. MIGU leverages inherent model behaviors to prevent catastrophic forgetting and achieves state-of-the-art performance across various language model architectures and continual learning benchmarks.

### 280. [Title:
          Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models](https://arxiv.org/pdf/2407.12327)
**Summary**: The paper introduces Spectra, a comprehensive suite of Large Language Models (LLMs) including Ternary Language Models (TriLMs), Quantized Language Models (QuantLMs), and traditional Floating-Point Language Models (FloatLMs). The study demonstrates that TriLMs, despite using fewer bits, outperform their QuantLM and FloatLM counterparts, especially at scales exceeding one billion parameters. This research highlights the potential for more efficient LLMs by exploring low-bitwidth models and provides over

### 281. [Title:
          Can Large Language Models Understand Symbolic Graphics Programs?](https://arxiv.org/pdf/2408.08313)
**Summary**: The paper investigates the ability of large language models (LLMs) to understand symbolic graphics programs, which generate visual data without requiring a vision encoder. By creating a benchmark for semantic visual understanding, the study evaluates LLMs' spatial-semantic reasoning skills and introduces Symbolic Instruction Tuning (SIT) to enhance their performance. The findings show that SIT not only improves understanding of symbolic programs but also boosts general reasoning capabilities across various benchmarks.

### 282. [Title:
          Automated Progressive Red Teaming](https://arxiv.org/pdf/2407.03876)
**Summary**: The paper introduces Automated Progressive Red Teaming (APRT), a framework designed to identify vulnerabilities in large language models (LLMs) by automating the process of generating adversarial prompts. APRT uses three core modules—Intention Expanding LLM, Intention Hiding LLM, and Evil Maker—to progressively explore and exploit LLM weaknesses through multi-round interactions. The framework also introduces a new metric, Attack Effectiveness Rate (AER), to evaluate the likelihood of eliciting unsafe but

### 283. [Title:
          CBF-LLM: Safe Control for LLM Alignment](https://arxiv.org/pdf/2408.15625)
**Summary**: The paper introduces CBF-LLM, a control-based framework that uses control barrier functions (CBFs) to align large language models (LLMs) and ensure safe text generation. By applying a safety filter to the token sequence output of a baseline LLM, the framework aims to reduce the need for interventions in user-specified alignment tasks. The system is implemented with Llama 3 and RoBERTa models, and experimental results show its effectiveness in controlling text generation.

### 284. [Title:
          Cybench: A Framework for Evaluating Cybersecurity Capabilities and Risks of Language Models](https://arxiv.org/pdf/2408.08926)
**Summary**: The paper introduces Cybench, a framework for evaluating the cybersecurity capabilities and risks of language models by testing them on 40 professional-level Capture the Flag (CTF) tasks. The framework includes subtasks to break down complex tasks and evaluates eight models, finding that some models, like Claude 3.5 Sonnet and GPT-4o, can solve tasks comparable to human performance in cybersecurity challenges.

### 285. [Title:
          An Adversarial Perspective on Machine Unlearning for AI Safety](https://arxiv.org/pdf/2409.18025)
**Summary**: The paper examines the effectiveness of machine unlearning methods in removing hazardous capabilities from large language models, particularly from an adversarial perspective. It demonstrates that existing jailbreak techniques can bypass unlearning protections when applied strategically and introduces adaptive methods that recover most unlearned capabilities, questioning the robustness of current unlearning approaches compared to traditional safety training.

### 286. [Title:
          The Crucial Role of Samplers in Online Direct Preference Optimization](https://arxiv.org/pdf/2409.19605)
**Summary**: The paper investigates the impact of different sampling strategies on the convergence rates of Direct Preference Optimization (DPO), finding that uniform sampling leads to linear convergence, while an online sampler achieves quadratic convergence. The proposed method, incorporating posterior distributions and logit mixing, significantly outperforms existing approaches in empirical evaluations, suggesting new directions for algorithm design in language model alignment.

### 287. [Title:
          Few-shot Prompting for Pairwise Ranking: An Effective Non-Parametric Retrieval Model](https://arxiv.org/pdf/2409.17745)
**Summary**: The paper introduces a pairwise few-shot ranker that enhances retrieval performance by leveraging a small number of training examples, showing improvements over zero-shot baselines in both in-domain and out-domain benchmarks. This method achieves near-supervised model performance without the need for complex training pipelines, demonstrating the effectiveness of few-shot prompting in ranking tasks.

### 288. [Title:
          Residual Stream Analysis with Multi-Layer SAEs](https://arxiv.org/pdf/2409.04185)
**Summary**: The paper introduces Multi-Layer Sparse Autoencoders (MLSAEs), which are trained on residual stream activations from all transformer layers, allowing for a unified analysis of information flow across layers. The study finds that individual latents are often active at a single layer per token or prompt, with significant variability across tokens, and that larger models exhibit greater similarity between adjacent layers. The findings provide insights into how representations evolve in transformers and are supported by code released for further analysis.

### 289. [Title:
          Representation Tuning](https://arxiv.org/pdf/2409.06927)
**Summary**: The paper introduces "representation tuning," a method for embedding behavioral vectors directly into large language models (LLMs) to control their output without requiring online adjustments. By fine-tuning the model with a dual loss function combining cosine similarity and token-based loss, the authors demonstrate enhanced control over honesty in model responses compared to traditional fine-tuning and online steering methods. This approach shows promise as a safety measure for LLMs.

### 290. [Title:
          MaPPER: Multimodal Prior-guided Parameter Efficient Tuning for Referring Expression Comprehension](https://arxiv.org/pdf/2409.13609)
**Summary**: The paper introduces MaPPER, a novel framework for Referring Expression Comprehension (REC) that leverages Multimodal Prior-guided Parameter Efficient Tuning to enhance performance while significantly reducing computational costs. MaPPER uses Dynamic Prior Adapters and Local Convolution Adapters to improve local visual perception and cross-modal alignment, outperforming full fine-tuning and other parameter-efficient methods with minimal tunable parameters.

### 291. [Title:
          Adversarial Suffixes May Be Features Too!](https://arxiv.org/pdf/2410.00451)
**Summary**: The paper investigates the hypothesis that adversarial suffixes in large language models (LLMs) like GPT-4 and LLaMA 3 are not just vulnerabilities but may represent features that can dominate model behavior. Through experiments, the authors demonstrate that benign features can be transformed into adversarial suffixes that compromise safety alignment and that these features can be introduced through fine-tuning with benign datasets alone. This suggests a critical risk posed by benign features in training data and emphasizes the need for further research to enhance LLM

### 292. [Title:
          Training Nonlinear Transformers for Chain-of-Thought Inference: A Theoretical Generalization Analysis](https://arxiv.org/pdf/2410.02167)
**Summary**: The paper presents a theoretical analysis of training nonlinear Transformers for Chain-of-Thought (CoT) inference, addressing the challenges of nonconvex optimization in nonlinear attention models. It quantifies the necessary training samples and iterations for achieving CoT generalization, proving its effectiveness on unseen tasks with distribution-shifted data, and characterizing conditions for accurate reasoning even with noisy examples. This contrasts with in-context learning, which may fail in similar scenarios.

### 293. [Title:
          Jailbreak Antidote: Runtime Safety-Utility Balance via Sparse Representation Adjustment in Large Language Models](https://arxiv.org/pdf/2410.02298)
**Summary**: The paper introduces Jailbreak Antidote, a method for dynamically adjusting the safety-utility balance in large language models (LLMs) by manipulating a sparse subset of the model's internal states during inference. This approach allows for real-time control over safety preferences without increasing computational overhead or inference latency, and it is shown to be effective across a range of LLMs and against various jailbreak attacks.

### 294. [Title:
          Frame-Voyager: Learning to Query Frames for Video Large Language Models](https://arxiv.org/pdf/2410.03226)
**Summary**: The paper introduces Frame-Voyager, a method that learns to select informative frame combinations from videos for Video Large Language Models (Video-LLMs), addressing the limitation of input token length. By ranking frame combinations based on prediction losses from a pre-trained Video-LLM, Frame-Voyager is trained to query the most relevant frames, significantly improving performance on Video Question Answering benchmarks across different Video-LLMs.

### 295. [Title:
          MedVisionLlama: Leveraging Pre-Trained Large Language Model Layers to Enhance Medical Image Segmentation](https://arxiv.org/pdf/2410.02458)
**Summary**: The paper introduces MedVisionLlama, a method that integrates pre-trained Large Language Model (LLM) transformer blocks into Vision Transformers (ViTs) to enhance medical image segmentation. By incorporating a frozen LLM transformer block in the encoder and using a Hybrid Attention Mechanism with Multi-Scale Fusion, the model achieves significant improvements in segmentation performance, with an average Dice score increase from 0.74 to 0.79 and enhancements in accuracy, precision, and the Jaccard



---

*Last updated on 2024-10-09*