

papers = [
    {
        "method_name" : "Turing ULR v6",
        "score" : 91.3,
        "arxiv_link" : "https://arxiv.org/abs/2210.14867",
        "title" : """Beyond English-Centric Bitexts for Better Multilingual Language Representation Learning""",
        "abstract" : """In this paper, we elaborate upon recipes
                        for building multilingual representation models that are not only competitive with existing state-of-the-art models but are also more
                        parameter efficient, thereby promoting better adoption in resource-constrained scenarios
                        and practical applications. We show that going beyond English-centric bitexts, coupled
                        with a novel sampling strategy aimed at reducing under-utilization of training data, substantially boosts performance across model
                        sizes for both Electra and MLM pre-training
                        objectives. We introduce XY-LENT: X-Y
                        bitext enhanced Language ENcodings using
                        Transformers which not only achieves state-ofthe-art performance over 5 cross-lingual tasks
                        within all model size bands, is also competitive across bands. Our XY-LENTXL variant outperforms XLM-RXXL and exhibits competitive performance with mT5XXL while being 5x and 6x smaller respectively. We then
                        show that our proposed method helps ameliorate the curse of multilinguality, with the
                        XY-LENTXL achieving 99.3% GLUE performance and 98.5% SQuAD 2.0 performance
                        compared to a SoTA English only model in the
                        same size band. We then analyze our models
                        performance on extremely low resource languages and posit that scaling alone may not
                        be sufficient for improving the performance in
                        this scenario.""",
        "introduction" : """Recent advancements in Natural Language Processing (NLP) have been a direct consequence of
                            leveraging foundational models (Bommasani et al.,
                            2021), pretrained on a large text corpora in a selfsupervised fashion. This has also been the case
                            for multilingual NLP where pre-trained models
                            like multilingual BERT (mBERT) (Devlin, 2018;
                            Devlin et al., 2019), XLM (Conneau and Lample)  XLM-Roberta (Conneau et al., 2020), XLMElectra (Chi et al., 2022) and mT5 (Xue et al., 2021)
                            have all shown non-trivial performance gains, especially in the setup of zero-shot transfer, and have
                            been the work-horse for a diverse number of multilingual tasks. Given their ubiquitous applicability in zero-shot downstream scenarios, improving
                            the quality and enabling their usage in resourceconstrained applications is also an important vein
                            of research which we explore in this paper.
                            A source of improvement for these models has
                            been leveraging bitext data for better representation
                            learning (Conneau and Lample, 2019; Chi et al.,
                            2022). Most prior work, however, has focused
                            on leveraging English-centric (EN-X) bitext data.
                            Contemporaneously, the related area of Massively Multilingual Machine Translation (a single model
                            for translating between different pairs of languages,
                            eg: Aharoni et al. (2019); Zhang et al. (2020); Fan
                            et al. (2021)) has shown tremendous progress, with
                            Fan et al. (2021) showing that a crucial aspect of
                            this improvement has been moving beyond EN-X
                            parallel corpora and leveraging web-based mined
                            X-Y bitexts spanning 1000s of translation directions
                            (Schwenk et al., 2021a; El-Kishky et al., 2020;
                            Schwenk et al., 2021b). This makes a compelling
                            case to explore if leveraging X-Y bitexts can also
                            improve multilingual representation learning.
                            In this work, we introduce XY-LENT (pronounced as "Excellent"): X-Y bitext enhanced
                            Language ENcodings using Transformers. We first
                            identify problems with using the commonly used
                            sampling strategy proposed in Fan et al. (2021),
                            showing that it induces sparse sampling distributions leading to under-utilization of data, and thus
                            propose a novel strategy to mitigate this issue
                            (§3.2). We then propose leveraging X-Y bitexts
                            in conjunction with the improved sampling strategy, as well as a VoCAP (Zheng et al., 2021) style
                            sentencepiece vocabulary re-construction for improving multilingual representation learning (§3.1).
                            We show that our proposed method improves performance across all model size bands (§6). Additionally, we show that the performance gains hold
                            for both Masked Language Models (MLM) and
                            ELECTRA style models, affording an almost 12x
                            speedup in training for the former (§6.2). We systematically analyse the impact of model scaling
                            with respect to the curse of multilinguality (Conneau et al., 2020) to observe that the gap between
                            current English only SoTA models and multilingual
                            models can be considerably reduced (§6.3). Our
                            analysis reveals that XY-LENT improves performance across language families (§6.4) and helps
                            reduce the cross-lingual transfer gap in multilingual
                            tasks (§6.5). We then demonstrate that the training
                            dynamics of such models can be used to better understand the underlying datasets and use it to find
                            interesting defects in them (§6.6). Finally, we show
                            some limitations of such multilingual representational models vis-à-vis extremely low resource languages, identifying potential shortcomings that are
                            not addressed with scaling of such models, as well
                            as issues around catastrophic forgetting in the way
                            current models are used for domain adaptation.
                            In doing so, we establish state of the art on 5 multilingual downstream tasks (XNLI, PAWS-X, TYDIQA, XQuAD and MLQA) within a model size
                            band, and achieve competitive performance across
                            size bands, thereby showing for the first time (to
                            the best of our knowledge) an interesting notion of
                            parameter efficiency: XY-LENTXL outperforms
                            XLM-RXXL (Goyal et al., 2021) and performs competitively with mT5XXL (Xue et al., 2021), whilst
                            being 5x and 6x smaller respectively (Figure 1).
                            Furthermore, our proposed model reduces the gap
                            for English specific tasks: XY-LENTXL achieves
                            99.3% GLUE performance and 98.5% SQuAD 2.0
                            performance compared to a SoTA English only
                            model in the same size band."""
    },

    {
        "method_name" : "ANNA",
        "score" : 89.8,
        "arxiv_link" : "https://arxiv.org/abs/2203.14507",
        "title" : """ANNA: Enhanced Language Representation for Question Answering""",
        "abstract" : """Pre-trained language models have brought significant improvements in performance in a
                        variety of natural language processing tasks.
                        Most existing models performing state-of-theart results have shown their approaches in the
                        separate perspectives of data processing, pretraining tasks, neural network modeling, or
                        fine-tuning. In this paper, we demonstrate
                        how the approaches affect performance individually, and that the language model performs
                        the best results on a specific question answering task when those approaches are jointly
                        considered in pre-training models. In particular, we propose an extended pre-training
                        task, and a new neighbor-aware mechanism
                        that attends neighboring tokens more to capture the richness of context for pre-training language modeling. Our best model achieves new
                        state-of-the-art results of 95.7 F1 and 90.6%
                        EM on SQuAD 1.1 and also outperforms existing pre-trained language models such as
                        RoBERTa, ALBERT, ELECTRA, and XLNet
                        on the SQuAD 2.0 benchmark.""",
        "introduction" : """Question answering (QA) is the task of answering given questions, which demands a high level
                            of language understanding and machine reading
                            comprehension abilities. As pre-trained language
                            models based on a transformer encoder (Vaswani
                            et al., 2017) have brought a huge improvement
                            in performance on a broad range of natural language processing (NLP) tasks including QA tasks,
                            methodologies for QA tasks are widely used to develop applications such as dialog systems (Bansal
                            et al., 2021) and chat-bots (Hemant et al., 2022;
                            Duggirala et al., 2021) in a variety of domains.
                            Pre-trained language models like BERT (Devlin
                            et al., 2018) are designed to represent individual
                            words for contextualization. However, recent extractive QA tasks such as Stanford Question Answering Dataset (SQuAD) benchmarks (Rajpurkar et al., 2016, 2018) involve reasoning relationships
                            between spans of texts that include a group of two
                            or more words in the evidence document (Lee et al.,
                            2016). In the example, as shown in Figure 1, “a
                            golden statue of the Virgin Mar”, the correct answer for the question “What sits on top of the Main
                            Building at Notre Dame?”, is a group of words
                            consisting of nouns and other words and is called
                            as a noun phrase, which performs as a noun in a
                            sentence. Since predicting a span of answer texts
                            including a start and end positions may be challenging for self-supervised training rather than predicting an individual word, we introduce a novel
                            pre-training approach that extends a standard masking scheme to wider spans of texts such as a nounphrase rather than an entity level and prove that
                            this approach is more effective for an extractive
                            QA task by outperforming existing models.
                            In this paper, we present a new pre-training
                            approach, ANNA (Approach of Noun-phrase
                            based language representation with Neighboraware Attention), which is designed to better understand syntactic and contextual information based
                            on comprehensive experimental evaluation of data
                            processing, pre-training tasks, attention mechanisms. First, we extend the conventional pretraining tasks. Our models are trained to predict
                            not only individual tokens but also an entire span
                            of noun phrases during the pre-training procedure.
                            This noun-phrase span masking scheme lets models
                            learn contextualized representations in the whole
                            span level, which benefits predicting answer texts
                            for the specific extractive QA tasks. Second, we
                            enhance the self-attention approach by incorporating a novel neighbor-aware mechanism in the
                            transformer architecture (Vaswani et al., 2017). We
                            find that more consideration of relationships between neighboring tokens by masking diagonality
                            in attention matrix is helpful for contextualized representations. Additionally, we use a larger volume
                            of corpora for pre-training language models and
                            find that using a lot of additional datasets does not
                            guarantee the best performance.
                            We evaluate our proposed models on the SQuAD
                            datasets which is a major extractive QA benchmarks for pre-trained language models. For
                            SQuAD 1.1 task, ANNA achieves new state-of-theart results of 90.6% Exact Match (EM) and 95.7%
                            F1-score (F1). When evaluated on the SQuAD 2.0
                            development dataset, the results show that our proposed approaches obtain competitive performance
                            outperforming self-supervised pre-training models
                            such as BERT, ALBERT, RoBERTa, and XLNet
                            models.
                            We summarize our main contributions as follows:
                            • We propose a new pre-trained language model,
                            ANNA that is designed to address extractive
                            QA tasks. ANNA is trained to predict the
                            masked group of words that is an entire noun
                            phrase, in order to better learn syntactic and
                            contextual information by taking advantage of
                            span-level representations.
                            • We introduce a novel transformer encoding
                            mechanism stacking new neighbor-aware selfattention on an original self-attention in the
                            transformer encoder block. The proposed
                            method takes into account neighbor tokens
                            more importantly than identical tokens during
                            the computation of attention scores.
                            • ANNA establishes new state-of-the-art results
                            on the SQuAD 1.1 leaderboard and outperforms existing pre-trained language models
                            for the SQuAD 2.0 dataset.""",
    },

    {
        "method_name" : "ConvBERT base", 
        "score" :  83.2,
        "arxiv_link" : "https://arxiv.org/abs/2008.02496",
        "title" : """ConvBERT: Improving BERT with Span-based Dynamic Convolution""",
        "abstract" : """Pre-trained language models like BERT and its variants have recently achieved
                        impressive performance in various natural language understanding tasks. However,
                        BERT heavily relies on the global self-attention block and thus suffers large
                        memory footprint and computation cost. Although all its attention heads query on
                        the whole input sequence for generating the attention map from a global perspective,
                        we observe some heads only need to learn local dependencies, which means the
                        existence of computation redundancy. We therefore propose a novel span-based
                        dynamic convolution to replace these self-attention heads to directly model local
                        dependencies. The novel convolution heads, together with the rest self-attention
                        heads, form a new mixed attention block that is more efficient at both global
                        and local context learning. We equip BERT with this mixed attention design and
                        build a ConvBERT model. Experiments have shown that ConvBERT significantly
                        outperforms BERT and its variants in various downstream tasks, with lower training
                        costs and fewer model parameters. Remarkably, ConvBERTBASE model achieves
                        86.4 GLUE score, 0.7 higher than ELECTRABASE, using less than 1/4 training
                        cost. """,
        "introduction" : """Language model pre-training has shown great power for improving many natural language processing
                            tasks [63, 52, 51, 33]. Most pre-training models, despite their variety, follow the BERT [14]
                            architecture heavily relying on multi-head self-attention [62] to learn comprehensive representations.
                            It has been found that 1) though the self-attention module in BERT is a highly non-local operator, a
                            large proportion of attention heads indeed learn local dependencies due to the inherent property of
                            natural language [31, 2]; 2) removing some attention heads during fine-tuning on downstream tasks
                            does not degrade the performance [41]. The two findings indicate that heavy computation redundancy
                            exists in the current model design. In this work, we aim to resolve this intrinsic redundancy issue
                            and further improve BERT w.r.t. its efficiency and downstream task performance. We consider such
                            a question: can we reduce the redundancy of attention heads by using a naturally local operation
                            to replace some of them? We notice that convolution has been very successful in extracting local
                            features [35, 32, 55, 23], and thus propose to use convolution layers as a more efficient complement
                            to self-attention for addressing local dependencies in natural language.
                            Specifically, we propose to integrate convolution into self-attention to form a mixed attention mechanism that combines the advantages of the two operations. Self-attention uses all input tokens to
                            generate attention weights for capturing global dependencies, while we expect to perform local “self-attention”, i.e., taking in a local span of the current token to generate “attention weights” of the
                            span to capture local dependencies. To achieve this, rather than deploying standard convolution with
                            fixed parameters shared for all input tokens, dynamic convolution [66] is a good choice that offers
                            higher flexibility in capturing local dependencies of different tokens. As shown in Fig. 1b, dynamic
                            convolution uses a kernel generator to produce different kernels for different input tokens [66].
                            However, such dynamic convolution cannot differentiate the same tokens within different context and
                            generate the same kernels (e.g., the three “can” in Fig. 1b).
                            We thus develop the span-based dynamic convolution, a novel convolution that produces more
                            adaptive convolution kernels by receiving an input span instead of only a single token, which enables
                            discrimination of generated kernels for the same tokens within different context. For example,
                            as shown in Fig. 1c, the proposed span-based dynamic convolution produces different kernels for
                            different “can” tokens. With span-based dynamic convolution, we build the mixed attention to
                            improve the conventional self-attention, which brings higher efficiency for pre-training as well as
                            better performance for capturing global and local information.
                            To further enhance performance and efficiency, we also add the following new architecture design to
                            BERT. First, a bottleneck structure is designed to reduce the number of attention heads by embedding
                            input tokens to a lower-dimensional space for self-attention. This also relieves the redundancy that lies
                            in attention heads and improves efficiency. Second, the feed-forward module in BERT consists of two
                            fully connected linear layers with an activation in between, but the dimensionality of the inner-layer
                            is set much higher (e.g., 4×) than that of input and output, which promises good performance but
                            brings large parameter number and computation. Thus we devise a grouped linear operator for the
                            feed-forward module, which reduces parameters without hurting representation power. Combining
                            these novelties all together makes our proposed model, termed ConvBERT, small and efficient.
                            Our contributions are summarized as follows. 1) We propose a new mixed attention to replace the
                            self-attention modules in BERT, which leverages the advantages of convolution to better capture local
                            dependency. To the best of our knowledge, we are the first to explore convolution for enhancing
                            BERT efficiency. 2) We introduce a novel span-based dynamic convolution operation to utilize
                            multiple input tokens to dynamically generate the convolution kernel. 3) Based on the proposed
                            span-based dynamic convolution and mixed attention, we build ConvBERT model. On the GLUE
                            benchmark, ConvBERTBASE achieves 86.4 GLUE score which is 5.5 higher than BERTBASE and
                            0.7 higher than ELECTRABASE while requiring less training cost and parameters. 4) ConvBERT
                            also incorporates some new model designs including the bottleneck attention and grouped linear
                            operator that are of independent interest for other NLP model development."""
    },

    {
        "method_name" : "AMBERT-BASE",
        "score" : 81.0,
        "arxiv_link" : "https://arxiv.org/abs/2008.11869",
        "title" : """AMBERT: A Pre-trained Language Model with Multi-Grained Tokenization""",
        "abstract" : """Pre-trained language models such as BERT
                        have exhibited remarkable performances in
                        many tasks in natural language understanding (NLU). The tokens in the models are usually fine-grained in the sense that for languages like English they are words or subwords and for languages like Chinese they are
                        characters. In English, for example, there
                        are multi-word expressions which form natural
                        lexical units and thus the use of coarse-grained
                        tokenization also appears to be reasonable.
                        In fact, both fine-grained and coarse-grained
                        tokenizations have advantages and disadvantages for learning of pre-trained language models. In this paper, we propose a novel pretrained language model, referred to as AMBERT (A Multi-grained BERT), on the basis
                        of both fine-grained and coarse-grained tokenizations. For English, AMBERT takes both
                        the sequence of words (fine-grained tokens)
                        and the sequence of phrases (coarse-grained
                        tokens) as input after tokenization, employs
                        one encoder for processing the sequence of
                        words and the other encoder for processing
                        the sequence of the phrases, utilizes shared
                        parameters between the two encoders, and finally creates a sequence of contextualized representations of the words and a sequence of
                        contextualized representations of the phrases.
                        Experiments have been conducted on benchmark datasets for Chinese and English, including CLUE, GLUE, SQuAD and RACE. The
                        results show that AMBERT can outperform
                        BERT in all cases, particularly the improvements are significant for Chinese. We also develop a method to improve the efficiency of
                        AMBERT in inference, which still performs
                        better than BERT with the same computational
                        cost as BERT.""",
        "introduction" : """Pre-trained models such as BERT, RoBERTa, and
                            ALBERT (Devlin et al., 2018; Liu et al., 2019; Lan
                            et al., 2019) have shown great power in natural
                            language understanding (NLU). The Transformerbased language models are first learned from a
                            large corpus in pre-training, and then learned from
                            labeled data of a downstream task in fine-tuning.
                            With Transformer (Vaswani et al., 2017), pretraining technique, and big data, the models can
                            effectively capture the lexical, syntactic, and semantic relations between the tokens in the input
                            text and achieve state-of-the-art performance in
                            many NLU tasks, such as sentiment analysis, text
                            entailment, and machine reading comprehension.
                            In BERT, for example, pre-training is mainly
                            conducted based on masked language modeling
                            (MLM) in which about 15% of the tokens in the
                            input text are masked with a special token [MASK],
                            and the goal is to reconstruct the original text from
                            the masked tokens. Fine-tuning is separately performed for individual tasks as text classification,
                            text matching, text span detection, etc. Usually, the
                            tokens in the input text are fine-grained; for example, they are words or sub-words in English and
                            characters in Chinese. In principle, the tokens can
                            also be coarse-grained, that is, for example, phrases
                            in English and words in Chinese. There are many
                            multi-word expressions in English such as ‘New
                            York’ and ‘ice cream’ and the use of phrases also
                            appears to be reasonable. It is more sensible to use
                            words (including single character words) in Chinese, because they are basic lexical units. In fact,
                            all existing pre-trained language models employ
                            single-grained (usually fine-grained) tokenization.
                            Previous work indicates that the fine-grained approach and the coarse-grained approach have both
                            pros and cons. The tokens in the fine-grained approach are less complete as lexical units but their
                            representations are easier to learn (because there
                            are less token types and more tokens in training
                            data), while the tokens in the coarse-grained approach are more complete as lexical units but their representations are more difficult to learn (because
                            there are more token types and less tokens in training data). Moreover, for the coarse-grained approach there is no guarantee that tokenization (segmentation) is completely correct. Sometimes ambiguity exists and it would be better to retain all
                            possibilities of tokenization. In contrast, for the
                            fine-grained approach tokenization is carried out at
                            the primitive level and there is no risk of ‘incorrect’
                            tokenization.
                            For example, (Li et al., 2019) observe that finegrained models consistently outperform coarsegrained models in deep learning for Chinese language processing. They point out that the reason is
                            that low frequency words (coarse-grained tokens)
                            tend to have insufficient training data and tend to
                            be out of vocabulary, and as a result the learned
                            representations are not sufficiently reliable. On the
                            other hand, previous work also demonstrates that
                            masking of coarse-grained tokens in pre-training
                            of language models is helpful (Cui et al., 2019;
                            Joshi et al., 2020). That is, although the model
                            itself is fine-grained, masking on consecutive tokens (phrases in English and words in Chinese)
                            can lead to learning of a more accurate model. In
                            Appendix A, we give examples of attention maps
                            in BERT to further support the assertion.
                            In this paper, we propose A Multi-grained
                            BERT model (AMBERT), which employs both
                            fine-grained and coarse-grained tokenizations. For
                            English, AMBERT extends BERT by simultaneously constructing representations for both words
                            and phrases in the input text using two encoders.
                            Specifically, AMBERT first conducts tokenization
                            at both word and phrase levels. It then takes the embeddings of words and phrases as input to the two
                            encoders with the shared parameters. Finally it obtains a contextualized representation for the word
                            and a contextualized representation for the phrase
                            at each position. Note that the number of parameters in AMBERT is comparable to that of BERT, because the parameters in the two encoders are shared.
                            There are only additional parameters from multigrained embeddings. AMBERT can represent the
                            input text at both word-level and phrase-level, to
                            leverage the advantages of the two approaches of
                            tokenization, and create richer representations for
                            the input text at multiple granularity.
                            AMBERT consists of two encoders and thus its
                            computational cost is roughly doubled compared
                            with BERT. We also develop a method for improving the efficiency of AMBERT in inference,
                            which only uses one of the two encoders. One
                            can choose either the fine-grained encoder or the
                            coarse-grained encoder for a specific task using a
                            development dataset.
                            We conduct extensive experiments to make a
                            comparison between AMBERT and the baselines
                            as well as alternatives to AMBERT, using the
                            benchmark datasets in English and Chinese. The results show that AMBERT significantly outperforms
                            single-grained BERT models with a large margin
                            in both Chinese and English. In English, compared to Google BERT, AMBERT achieves 2.0%
                            higher GLUE score, 2.5% higher RACE score, and
                            5.1% more SQuAD score. In Chinese, AMBERT
                            improves average score by over 2.7% in CLUE.
                            Furthermore, AMBERT with only one encoder can
                            preform much better than the single-grained BERT
                            models with a similar amount of inference time.
                            We make the following contributions.
                            • Study of multi-grained pre-trained language
                            models,
                            • Proposal of a new pre-trained language model
                            called AMBERT as an extension of BERT,
                            • Empirical verification of AMBERT on the English and Chinese benchmark datasets GLUE,
                            SQuAD, RACE, and CLUE,
                            • Proposal of an efficient inference method for
                            AMBERT."""
    },

    {
        "method_name" : "SesameBERT-Base",
        "score" : 78.6,
        "arxiv_link" : "https://arxiv.org/abs/1910.03176",
        "title" : """SesameBERT: Attention for Anywhere""",
        "abstract" : """Fine-tuning with pre-trained models has achieved exceptional results for many
                        language tasks. In this study, we focused on one such self-attention network
                        model, namely BERT, which has performed well in terms of stacking layers
                        across diverse language-understanding benchmarks. However, in many downstream tasks, information between layers is ignored by BERT for fine-tuning. In
                        addition, although self-attention networks are well-known for their ability to capture global dependencies, room for improvement remains in terms of emphasizing
                        the importance of local contexts. In light of these advantages and disadvantages,
                        this paper proposes SesameBERT, a generalized fine-tuning method that (1) enables the extraction of global information among all layers through Squeeze and
                        Excitation and (2) enriches local information by capturing neighboring contexts
                        via Gaussian blurring. Furthermore, we demonstrated the effectiveness of our approach in the HANS dataset, which is used to determine whether models have
                        adopted shallow heuristics instead of learning underlying generalizations. The experiments revealed that SesameBERT outperformed BERT with respect to GLUE
                        benchmark and the HANS evaluation set.""",
        "introduction" : """In recent years, unsupervised pretrained models have dominated the field of natural language processing (NLP). The construction of a framework for such a model involves two steps: pretraining
                            and fine-tuning. During pretraining, an encoder neural network model is trained using large-scale
                            unlabeled data to learn word embeddings; parameters are then fine-tuned with labeled data related
                            to downstream tasks.
                            Traditionally, word embeddings are vector representations learned from large quantities of unstructured textual data such as those from Wikipedia corpora (Mikolov et al., 2013). Each word is represented by an independent vector, even though many words are morphologically similar. To solve this
                            problem, techniques for contextualized word representation (Peters et al., 2018; Devlin et al., 2019)
                            have been developed; some have proven to be more effective than conventional word-embedding
                            techniques, which extract only local semantic information of individual words. By contrast, pretrained contextual representations learn sentence-level information from sentence encoders and can
                            generate multiple word embeddings for a word. Pretraining methods related to contextualized word
                            representation, such as BERT (Devlin et al., 2019), OpenAI GPT (Radford et al., 2018), and ELMo
                            (Peters et al., 2018), have attracted considerable attention in the field of NLP and have achieved
                            high accuracy in GLUE tasks such as single-sentence, similarity and paraphrasing, and inference
                            tasks (Wang et al., 2019). Among the aforementioned pretraining methods, BERT, a state-of-the-art
                            network, is the leading method that applies the architecture of the Transformer encoder, which outperforms other models with respect to the GLUE benchmark. BERT’s performance suggests that
                            self-attention is highly effective in extracting the latent meanings of sentence embeddings.
                            This study aimed to improve contextualized word embeddings, which constitute the output of encoder layers to be fed into a classifier. We used the original method of the pretraining stage in the
                            BERT model. During the fine-tuning process, we introduced a new architecture known as Squeeze
                            and Excitation alongside Gaussian blurring with symmetrically SAME padding (”SESAME” hereafter). First, although the developer of the BERT model initially presented several options for its
                            use, whether the selective layer approaches involved information contained in all layers was unclear.
                            In a previous study, by investigating relationships between layers, we observed that the Squeeze and Excitation method (Hu et al., 2018) is key for focusing on information between layer weights. This
                            method enables the network to perform feature recalibration and improves the quality of representations by selectively emphasizing informative features and suppressing redundant ones. Second, the
                            self-attention mechanism enables a word to analyze other words in an input sequence; this process
                            can lead to more accurate encoding. The main benefit of the self-attention mechanism method is
                            its high ability to capture global dependencies. Therefore, this paper proposes the strategy, namely
                            Gaussian blurring, to focus on local contexts. We created a Gaussian matrix and performed convolution alongside a fixed window size for sentence embedding. Convolution helps a word to focus on
                            not only its own importance but also its relationships with neighboring words. Through such focus,
                            each word in a sentence can simultaneously maintain global and local dependencies.
                            We conducted experiments with our proposed method to determine whether the trained model could
                            outperform the BERT model. We observed that SesameBERT yielded marked improvement across
                            most GLUE tasks. In addition, we adopted a new evaluation set called HANS (McCoy et al., 2019),
                            which was designed to diagnose the use of fallible structural heuristics, namely the lexical overlap
                            heuristic, subsequent heuristic, and constituent heuristic. Models that apply these heuristics are
                            guaranteed to fail in the HANS dataset. For example, although BERT scores highly in the given test
                            set, it performs poorly in the HANS dataset; BERT may label an example correctly not based on
                            reasoning regarding the meanings of sentences but rather by assuming that the premise entails any
                            hypothesis whose words all appear in the premise (Dasgupta et al., 2018). By contrast, SesameBERT
                            performs well in the HANS dataset; this implies that this model does not merely rely on heuristics.
                            In summary, our final model proved to be competitive on multiple downstream tasks."""
    },

    {
        "method_name" : "GenSen",
        "score" : 66.1,
        "arxiv_link" : "https://arxiv.org/abs/1804.00079",
        "title" : """Learning General Purpose Distributed Sentence Representations via Large Scale Multi-task Learning""",
        "abstract" : """A lot of the recent success in natural language processing (NLP) has been driven
                        by distributed vector representations of words trained on large amounts of text
                        in an unsupervised manner. These representations are typically used as general
                        purpose features for words across a range of NLP problems. However, extending this success to learning representations of sequences of words, such as sentences, remains an open problem. Recent work has explored unsupervised as well
                        as supervised learning techniques with different training objectives to learn general purpose fixed-length sentence representations. In this work, we present a
                        simple, effective multi-task learning framework for sentence representations that
                        combines the inductive biases of diverse training objectives in a single model. We
                        train this model on several data sources with multiple training objectives on over
                        100 million sentences. Extensive experiments demonstrate that sharing a single recurrent sentence encoder across weakly related tasks leads to consistent improvements over previous methods. We present substantial improvements in the context
                        of transfer learning and low-resource settings using our learned general-purpose
                        representations.""",
        "introduction" : """Transfer learning has driven a number of recent successes in computer vision and NLP. Computer vision tasks like image captioning (Xu et al., 2015) and visual question answering typically use CNNs
                            pretrained on ImageNet (Krizhevsky et al., 2012; Simonyan & Zisserman, 2014) to extract representations of the image, while several natural language tasks such as reading comprehension and
                            sequence labeling (Lample et al., 2016) have benefited from pretrained word embeddings (Mikolov
                            et al., 2013; Pennington et al., 2014) that are either fine-tuned for a specific task or held fixed.
                            Many neural NLP systems are initialized with pretrained word embeddings but learn their representations of words in context from scratch, in a task-specific manner from supervised learning signals.
                            However, learning these representations reliably from scratch is not always feasible, especially in
                            low-resource settings, where we believe that using general purpose sentence representations will be
                            beneficial.
                            Some recent work has addressed this by learning general-purpose sentence representations (Kiros
                            et al., 2015; Wieting et al., 2015; Hill et al., 2016; Conneau et al., 2017; McCann et al., 2017; Jernite
                            et al., 2017; Nie et al., 2017; Pagliardini et al., 2017). However, there exists no clear consensus yet
                            on what training objective or methodology is best suited to this goal.
                            Understanding the inductive biases of distinct neural models is important for guiding progress in
                            representation learning. Shi et al. (2016) and Belinkov et al. (2017) demonstrate that neural machine translation (NMT) systems appear to capture morphology and some syntactic properties. Shi
                            et al. (2016) also present evidence that sequence-to-sequence parsers (Vinyals et al., 2015) more
                            strongly encode source language syntax. Similarly, Adi et al. (2016) probe representations extracted
                            by sequence autoencoders, word embedding averages, and skip-thought vectors with a multi-layer
                            perceptron (MLP) classifier to study whether sentence characteristics such as length, word content
                            and word order are encoded.
                            To generalize across a diverse set of tasks, it is important to build representations that encode several
                            aspects of a sentence. Neural approaches to tasks such as skip-thoughts, machine translation, natural language inference, and constituency parsing likely have different inductive biases. Our work
                            exploits this in the context of a simple one-to-many multi-task learning (MTL) framework, wherein
                            a single recurrent sentence encoder is shared across multiple tasks. We hypothesize that sentence
                            representations learned by training on a reasonably large number of weakly related tasks will generalize better to novel tasks unseen during training, since this process encodes the inductive biases
                            of multiple models. This hypothesis is based on the theoretical work of Baxter et al. (2000). While
                            our work aims at learning fixed-length distributed sentence representations, it is not always practical
                            to assume that the entire “meaning” of a sentence can be encoded into a fixed-length vector. We
                            merely hope to capture some of its characteristics that could be of use in a variety of tasks.
                            The primary contribution of our work is to combine the benefits of diverse sentence-representation
                            learning objectives into a single multi-task framework. To the best of our knowledge, this is the first
                            large-scale reusable sentence representation model obtained by combining a set of training objectives with the level of diversity explored here, i.e. multi-lingual NMT, natural language inference,
                            constituency parsing and skip-thought vectors. We demonstrate through extensive experimentation
                            that representations learned in this way lead to improved performance across a diverse set of novel
                            tasks not used in the learning of our representations. Such representations facilitate low-resource
                            learning as exhibited by significant improvements to model performance for new tasks in the low
                            labelled data regime - achieving comparable performance to a few models trained from scratch using
                            only 6% of the available training set on the Quora duplicate question dataset."""
    },

    {
        "method_name" : "InferSent",
        "score" : 63.9,
        "arxiv_link" : "https://arxiv.org/abs/1705.02364",
        "title" : """Supervised Learning of Universal Sentence Representations from Natural Language Inference Data""",
        "abstract" : """Many modern NLP systems rely on word
                        embeddings, previously trained in an unsupervised manner on large corpora, as
                        base features. Efforts to obtain embeddings for larger chunks of text, such as
                        sentences, have however not been so successful. Several attempts at learning unsupervised representations of sentences have
                        not reached satisfactory enough performance to be widely adopted. In this paper,
                        we show how universal sentence representations trained using the supervised data of
                        the Stanford Natural Language Inference
                        datasets can consistently outperform unsupervised methods like SkipThought vectors (Kiros et al., 2015) on a wide range
                        of transfer tasks. Much like how computer vision uses ImageNet to obtain features, which can then be transferred to
                        other tasks, our work tends to indicate the
                        suitability of natural language inference
                        for transfer learning to other NLP tasks.
                        Our encoder is publicly available.""",
        "introduction" : """Distributed representations of words have shown to provide useful features for
                            various tasks in natural language processing and
                            computer vision. While there seems to be a consensus concerning the usefulness of word embeddings and how to learn them, this is not yet clear
                            with regard to representations that carry the meaning of a full sentence. That is, how to capture the
                            relationships among multiple words and phrases in
                            a single vector remains an question to be solved. In this paper, we study the task of learning
                            universal representations of sentences, i.e., a sentence encoder model that is trained on a large corpus and subsequently transferred to other tasks.
                            Two questions need to be solved in order to build
                            such an encoder, namely: what is the preferable neural network architecture; and how and
                            on what task should such a network be trained.
                            Following existing work on learning word embeddings, most current approaches consider learning sentence encoders in an unsupervised manner
                            like SkipThought (Kiros et al.
                            , 2015) or FastSent
                            (Hill et al.
                            , 2016). Here, we investigate whether
                            supervised learning can be leveraged instead, taking inspiration from previous results in computer
                            vision, where many models are pretrained on the
                            ImageNet (Deng et al.
                            , 2009) before being transferred. We compare sentence embeddings trained
                            on various supervised tasks, and show that sentence embeddings generated from models trained
                            on a natural language inference (NLI) task reach
                            the best results in terms of transfer accuracy. We
                            hypothesize that the suitability of NLI as a training task is caused by the fact that it is a high-level
                            understanding task that involves reasoning about
                            the semantic relationships within sentences.
                            Unlike in computer vision, where convolutional
                            neural networks are predominant, there are multiple ways to encode a sentence using neural networks. Hence, we investigate the impact of the
                            sentence encoding architecture on representational
                            transferability, and compare convolutional, recurrent and even simpler word composition schemes.
                            Our experiments show that an encoder based on a
                            bi-directional LSTM architecture with max pooling, trained on the Stanford Natural Language
                            Inference (SNLI) dataset (Bowman et al.
                            , 2015),
                            yields state-of-the-art sentence embeddings compared to all existing alternative unsupervised approaches like SkipThought or FastSent, while being much faster to train. We establish this finding
                            on a broad and diverse set of transfer tasks that
                            measures the ability of sentence representations to
                            capture general and useful information."""
    },


    {
        "method_name" : "DisSent",
        "score" : 61.9,
        "arxiv_link" : "https://arxiv.org/abs/1710.04334",
        "title" : """DisSent: Sentence Representation Learning from Explicit Discourse Relations""",
        "abstract" : """Learning effective representations of sentences is one of the core missions of natural language understanding. Existing models
                        either train on a vast amount of text, or require costly, manually curated sentence relation datasets. We show that with dependency
                        parsing and rule-based rubrics, we can curate
                        a high quality sentence relation task by leveraging explicit discourse relations. We show
                        that our curated dataset provides an excellent
                        signal for learning vector representations of
                        sentence meaning, representing relations that
                        can only be determined when the meanings
                        of two sentences are combined. We demonstrate that the automatically curated corpus allows a bidirectional LSTM sentence encoder
                        to yield high quality sentence embeddings and
                        can serve as a supervised fine-tuning dataset
                        for larger models such as BERT. Our fixed sentence embeddings achieve high performance
                        on a variety of transfer tasks, including SentEval, and we achieve state-of-the-art results
                        on Penn Discourse Treebank’s implicit relation prediction task.""",
        "introduction" : """Developing general models to represent the meaning of a sentence is a key task in natural language
                            understanding. The applications of such generalpurpose representations of sentence meaning are
                            many — paraphrase detection, summarization,
                            knowledge-base population, question-answering,
                            automatic message forwarding, and metaphoric
                            language, to name a few.
                            We propose to leverage a high-level relationship
                            between sentences that is both frequently and systematically marked in natural language: the discourse relations between sentences. Human writers naturally use a small set of very common transition words between sentences (or sentence-like phrases) to identify the relations between adjacent
                            ideas. These words, such as because, but, and,
                            which mark the conceptual relationship between
                            two sentences, have been widely studied in linguistics, both formally and computationally, and
                            have many different names. We use the name “discourse markers”.
                            Learning flexible meaning representations requires a sufficiently demanding, yet tractable,
                            training task. Discourse markers annotate deep
                            conceptual relations between sentences. Learning
                            to predict them may thus represent a strong training task for sentence meanings. This task is an interesting intermediary between two recent trends.
                            On the one hand, models like InferSent (Conneau
                            et al., 2017) are trained to predict entailment—a
                            strong conceptual relation that must be hand annotated. On the other hand, models like BERT
                            (Devlin et al., 2018) are trained to predict random
                            missing words in very large corpora (see Table 1
                            for the data requirements of the models we compare). Discourse marker prediction may permit
                            learning from relatively little data, like entailment,
                            but can rely on naturally occurring data rather than
                            hand annotation, like word-prediction.
                            We thus propose the DisSent task, which uses
                            the Discourse Prediction Task to train sentence
                            embeddings. Using a data preprocessing procedure based on dependency parsing, we are able to
                            automatically curate a sizable training set of sentence pairs. We then train a sentence encoding
                            model to learn embeddings for each sentence in a
                            pair such that a classifier can identify, based on the
                            embeddings, which discourse marker was used to
                            link the sentences. We also use the DisSent task to
                            fine-tune larger pre-trained models such as BERT.
                            We evaluate our sentence embedding model’s
                            performance on the standard fixed embedding
                            evaluation framework developed by Conneau et al.
                            (2017), where during evaluation, the sentence embedding model’s weights are not updated. We
                            further evaluate both the DisSent model and a
                            BERT model fine-tuned on DisSent on two classification tasks from the Penn Discourse Treebank
                            (PDTB) (Rashmi et al., 2008).
                            We demonstrate that the resulting DisSent embeddings achieve comparable results to InferSent
                            on some evaluation tasks, and superior on others.
                            The BERT model fine-tuned on the DisSent tasks
                            achieved state-of-the-art on PDTB classification
                            tasks compared to other fine-tuning strategies."""
    },


    {
        "method_name" : "ALBERT",
        "score" : 55,
        "arxiv_link" : "https://arxiv.org/abs/1909.11942",
        "title" : """ALBERT: A Lite BERT for Self-supervised Learning of Language Representations""",
        "abstract" : """Increasing model size when pretraining natural language representations often results in improved performance on downstream tasks. However, at some point further model increases become harder due to GPU/TPU memory limitations and
                        longer training times. To address these problems, we present two parameterreduction techniques to lower memory consumption and increase the training
                        speed of BERT (Devlin et al., 2019). Comprehensive empirical evidence shows
                        that our proposed methods lead to models that scale much better compared to
                        the original BERT. We also use a self-supervised loss that focuses on modeling
                        inter-sentence coherence, and show it consistently helps downstream tasks with
                        multi-sentence inputs. As a result, our best model establishes new state-of-the-art
                        results on the GLUE, RACE, and SQuAD benchmarks while having fewer parameters compared to BERT-large.""",
        "introduction" : """Full network pre-training (Dai & Le, 2015; Radford et al., 2018; Devlin et al., 2019; Howard &
                            Ruder, 2018) has led to a series of breakthroughs in language representation learning. Many nontrivial NLP tasks, including those that have limited training data, have greatly benefited from these
                            pre-trained models. One of the most compelling signs of these breakthroughs is the evolution of machine performance on a reading comprehension task designed for middle and high-school English
                            exams in China, the RACE test (Lai et al., 2017): the paper that originally describes the task and formulates the modeling challenge reports then state-of-the-art machine accuracy at 44.1%; the latest
                            published result reports their model performance at 83.2% (Liu et al., 2019); the work we present
                            here pushes it even higher to 89.4%, a stunning 45.3% improvement that is mainly attributable to
                            our current ability to build high-performance pretrained language representations.
                            Evidence from these improvements reveals that a large network is of crucial importance for achieving state-of-the-art performance (Devlin et al., 2019; Radford et al., 2019). It has become common
                            practice to pre-train large models and distill them down to smaller ones (Sun et al., 2019; Turc et al.,
                            2019) for real applications. Given the importance of model size, we ask: Is having better NLP
                            models as easy as having larger models?
                            An obstacle to answering this question is the memory limitations of available hardware. Given that
                            current state-of-the-art models often have hundreds of millions or even billions of parameters, it is
                            easy to hit these limitations as we try to scale our models. Training speed can also be significantly
                            hampered in distributed training, as the communication overhead is directly proportional to the
                            number of parameters in the model.
                            Existing solutions to the aforementioned problems include model parallelization (Shazeer et al.,
                            2018; Shoeybi et al., 2019) and clever memory management (Chen et al., 2016; Gomez et al., 2017). These solutions address the memory limitation problem, but not the communication overhead. In
                            this paper, we address all of the aforementioned problems, by designing A Lite BERT (ALBERT)
                            architecture that has significantly fewer parameters than a traditional BERT architecture.
                            ALBERT incorporates two parameter reduction techniques that lift the major obstacles in scaling
                            pre-trained models. The first one is a factorized embedding parameterization. By decomposing
                            the large vocabulary embedding matrix into two small matrices, we separate the size of the hidden
                            layers from the size of vocabulary embedding. This separation makes it easier to grow the hidden
                            size without significantly increasing the parameter size of the vocabulary embeddings. The second
                            technique is cross-layer parameter sharing. This technique prevents the parameter from growing
                            with the depth of the network. Both techniques significantly reduce the number of parameters for
                            BERT without seriously hurting performance, thus improving parameter-efficiency. An ALBERT
                            configuration similar to BERT-large has 18x fewer parameters and can be trained about 1.7x faster.
                            The parameter reduction techniques also act as a form of regularization that stabilizes the training
                            and helps with generalization.
                            To further improve the performance of ALBERT, we also introduce a self-supervised loss for
                            sentence-order prediction (SOP). SOP primary focuses on inter-sentence coherence and is designed
                            to address the ineffectiveness (Yang et al., 2019; Liu et al., 2019) of the next sentence prediction
                            (NSP) loss proposed in the original BERT.
                            As a result of these design decisions, we are able to scale up to much larger ALBERT configurations
                            that still have fewer parameters than BERT-large but achieve significantly better performance. We
                            establish new state-of-the-art results on the well-known GLUE, SQuAD, and RACE benchmarks
                            for natural language understanding. Specifically, we push the RACE accuracy to 89.4%, the GLUE
                            benchmark to 89.4, and the F1 score of SQuAD 2.0 to 92.2."""
    }
]