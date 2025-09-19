

papers = [
    {
        "method_name" : "AISO",
        "joint_f1" : 72.00,
        "arxiv_link" : "https://arxiv.org/abs/2109.06747",
        "title" : """Adaptive Information Seeking for Open-Domain Question Answering""",
        "abstract" : """Information seeking is an essential step for
                        open-domain question answering to efficiently
                        gather evidence from a large corpus. Recently,
                        iterative approaches have been proven to be effective for complex questions, by recursively
                        retrieving new evidence at each step. However, almost all existing iterative approaches
                        use predefined strategies, either applying the
                        same retrieval function multiple times or fixing the order of different retrieval functions,
                        which cannot fulfill the diverse requirements
                        of various questions. In this paper, we propose
                        a novel adaptive information-seeking strategy
                        for open-domain question answering, namely
                        AISO. Specifically, the whole retrieval and
                        answer process is modeled as a partially observed Markov decision process, where three
                        types of retrieval operations (e.g., BM25, DPR,
                        and hyperlink) and one answer operation are
                        defined as actions. According to the learned
                        policy, AISO could adaptively select a proper
                        retrieval action to seek the missing evidence
                        at each step, based on the collected evidence
                        and the reformulated query, or directly output the answer when the evidence set is sufficient for the question. Experiments on SQuAD
                        Open and HotpotQA fullwiki, which serve
                        as single-hop and multi-hop open-domain QA
                        benchmarks, show that AISO outperforms all
                        baseline methods with predefined strategies in
                        terms of both retrieval and answer evaluations.""",
        "introduction" : """Open-domain question answering (QA) (Voorhees
                            et al., 1999) is a task of answering questions using
                            a large collection of texts (e.g., Wikipedia). It relies on a powerful information-seeking method to
                            efficiently retrieve evidence from the given large
                            corpus.
                            Traditional open-domain QA approaches mainly
                            follow the two-stage retriever-reader pipeline (Chen et al., 2017; Yang et al., 2018; Karpukhin
                            et al., 2020), in which the retriever uses a determinate sparse or dense retrieval function to retrieve
                            evidence, independently from the reading stage.
                            But these approaches have limitations in answering complex questions, which need multi-hop or
                            logical reasoning (Xiong et al., 2021).
                            To tackle this issue, iterative approaches have
                            been proposed to recurrently retrieve passages
                            and reformulate the query based on the original
                            question and the previously collected passages.
                            Nevertheless, all of these approaches adopt fixed
                            information-seeking strategies in the iterative process. For example, some works employ a single
                            retrieval function multiple times (Das et al., 2019a;
                            Qi et al., 2019; Xiong et al., 2021), and the other
                            works use a pre-defined sequence of retrieval functions (Asai et al., 2020; Dhingra et al., 2020).
                            However, the fixed information-seeking strategies cannot meet the diversified requirements of
                            various problems. Taking Figure 1 as an example,
                            the answer to the question is ‘Catwoman’ in P3.
                            Due to the lack of essential supporting passages,
                            simply applying BM25/dense retrieval (DR) multiple times (strategy 1 (Qi et al., 2019) or 2 (Xiong
                            et al., 2021)), or using the mixed but fixed strategy
                            (strategy 3 (Asai et al., 2020)) cannot answer the
                            question. Specifically, it is hard for Qi et al. (2019)
                            to generate the ideal query ‘Catwoman game’ by
                            considering P1 or P2, thus BM25 (Robertson and
                            Zaragoza, 2009) suffers from the mismatch problem and fails to find the next supporting passage
                            P3. The representation learning of salient but rare
                            phrases (e.g. ‘Pitof’) still remains a challenging
                            problem (Karpukhin et al., 2020), which may affect the effectiveness of dense retrieval, i.e., the
                            supporting passage P3 is ranked 65, while P1 and
                            P2 do not appear in the top-1000 list at the first step.
                            Furthermore, link retrieval functions fail when the
                            current passage, e.g., P2, has no valid entity links.
                            Motivated by the above observations, we propose an Adaptive Information-Seeking approach
                            for Open-domain QA, namely AISO. Firstly, the
                            task of open-domain QA is formulated as a partially observed Markov decision process (POMDP)
                            to reflect the interactive characteristics between the
                            QA model (i.e., agent) and the intractable largescale corpus (i.e., environment). The agent is asked
                            to perform an action according to its state (belief
                            module) and the policy it learned (policy module).
                            Specifically, the belief module of the agent maintains a set of evidence to form its state. Moreover,
                            there are two groups of actions for the policy module to choose, 1) retrieval action that consists of
                            the type of retrieval function and the reformulated
                            query for requesting evidence, and 2) answer action
                            that returns a piece of text to answer the question,
                            then completes the process. Thus, in each step, the
                            agent emits an action to the environment, which returns a passage as the observation back to the agent.
                            The agent updates the evidence set and generates
                            the next action, step by step, until the evidence set
                            is sufficient to trigger the answer action to answer
                            the question. To learn such a strategy, we train the
                            policy in imitation learning by cloning the behavior of an oracle online, which avoids the hassle of
                            designing reward functions and solves the POMDP
                            in the fashion of supervised learning.
                            Our experimental results show that our approach achieves better retrieval and answering
                            performance than the state-of-the-art approaches
                            on SQuAD Open and HotpotQA fullwiki, which
                            are the representative single-hop and multi-hop
                            datasets for open-domain QA. Furthermore, AISO
                            significantly reduces the number of reading steps
                            in the inference stage.
                            In summary, our contributions include:
                            • To the best of our knowledge, we are the first
                            to introduce the adaptive information-seeking
                            strategy to the open-domain QA task;
                            • Modeling adaptive information-seeking as a
                            POMDP, we propose AISO, which learns the
                            policy via imitation learning and has great
                            potential for expansion.
                            • The proposed AISO achieves state-of-theart performance on two public dataset and
                            wins the first place on the HotpotQA fullwiki
                            leaderboard"""
    },

    {
        "method_name" : "IRRR+",
        "joint_f1" : 69.6,
        "arxiv_link" : "https://arxiv.org/abs/2010.12527",
        "title" : """Answering Open-Domain Questions of Varying Reasoning Steps from Text""",
        "abstract" : """We develop a unified system to answer directly from text open-domain questions that
                        may require a varying number of retrieval
                        steps. We employ a single multi-task transformer model to perform all the necessary
                        subtasks—retrieving supporting facts, reranking them, and predicting the answer from all
                        retrieved documents—in an iterative fashion.
                        We avoid crucial assumptions of previous work
                        that do not transfer well to real-world settings,
                        including exploiting knowledge of the fixed
                        number of retrieval steps required to answer
                        each question or using structured metadata like
                        knowledge bases or web links that have limited availability. Instead, we design a system that can answer open-domain questions
                        on any text collection without prior knowledge
                        of reasoning complexity. To emulate this setting, we construct a new benchmark, called
                        BeerQA, by combining existing one- and twostep datasets with a new collection of 530 questions that require three Wikipedia pages to answer, unifying Wikipedia corpora versions in
                        the process. We show that our model demonstrates competitive performance on both existing benchmarks and this new benchmark. We
                        make the new benchmark available at https:
                        //beerqa.github.io/.""",
        "introduction" : """Using knowledge to solve problems is a hallmark
                            of intelligence. Since human knowledge is often
                            containned in large text collections, open-domain
                            question answering (QA) is an important means for
                            intelligent systems to make use of the knowledge in
                            large text collections. With the help of large-scale
                            datasets based on Wikipedia (Rajpurkar et al., 2016,
                            2018) and other large corpora (Trischler et al., 2016;
                            Dunn et al., 2017; Talmor and Berant, 2018), the
                            research community has made substantial progress
                            on tackling this problem in recent years, including in the direction of complex reasoning over multiple
                            pieces of evidence, or multi-hop reasoning (Yang
                            et al., 2018; Welbl et al., 2018; Chen et al., 2020).
                            Despite this success, most previous systems are
                            developed with, and evaluated on, datasets that
                            contain exclusively single-hop questions (ones that
                            require a single document or paragraph to answer)
                            or two-hop ones. As a result, their design is often
                            tailored exclusively to single-hop (e.g., Chen et al.,
                            2017; Wang et al., 2018b) or multi-hop questions
                            (e.g., Nie et al., 2019; Min et al., 2019; Feldman
                            and El-Yaniv, 2019; Zhao et al., 2020a; Xiong
                            et al., 2021). Even when the model is designed to
                            work with both, it is often trained and evaluated on
                            exclusively single-hop or multi-hop settings (e.g.,
                            Asai et al., 2020). In practice, not only can we
                            not expect open-domain QA systems to receive
                            exclusively single- or multi-hop questions from
                            users, but it is also non-trivial to judge reliably
                            whether a question requires one or multiple pieces
                            of evidence to answer a priori. For instance, “In
                            which U.S. state was Facebook founded?” appears
                            to be single-hop, but its answer cannot be found in
                            the main text of a single English Wikipedia page.
                            Besides the impractical assumption about reasoning hops, previous work often also assumes access
                            to non-textual metadata such as knowledge bases,
                            entity linking, and Wikipedia hyperlinks when retrieving supporting facts, especially in answering
                            complex questions (Nie et al., 2019; Feldman and
                            El-Yaniv, 2019; Zhao et al., 2019; Asai et al., 2020;
                            Dhingra et al., 2020; Zhao et al., 2020a). While
                            this information is helpful, it is not always available in text collections we might be interested in
                            getting answers from, such as news or academic
                            research articles, besides being labor-intensive and
                            time-consuming to collect and maintain. It is therefore desirable to design a system that is capable of
                            extracting knowledge from text without using such
                            metadata, to maximally emphasize using knowledge
                            available to us in the form of text. To address these limitations, we propose Iterative
                            Retriever, Reader, and Reranker (IRRR), which
                            features a single neural network model that performs
                            all of the subtasks required to answer questions
                            from a large collection of text (see Figure 1). IRRR
                            is designed to leverage off-the-shelf information
                            retrieval systems by generating natural language
                            search queries, which allows it to easily adapt to
                            arbitrary collections of text without requiring welltuned neural retrieval systems or extra metadata.
                            This further allows users to understand and control
                            IRRR, if necessary, to facilitate trust. Moreover,
                            IRRR iteratively retrieves more context to answer
                            the question, which allows it to easily accommodate
                            questions of different number of reasoning steps.
                            To evaluate the performance of open-domain QA
                            systems in a more realistic setting, we construct
                            a new benchmark called BeerQA1 by combining
                            the questions from the single-hop SQuAD Open
                            (Rajpurkar et al., 2016; Chen et al., 2017) and the
                            two-hop HotpotQA (Yang et al., 2018) with a new
                            collection of 530 human-annotated questions that
                            require information from at least three Wikipedia
                            pages to answer. We map all questions to a unified
                            version of the English Wikipedia to reduce stylistic
                            differences that might provide statistical shortcuts
                            to models. As a result, BeerQA provides a more realistic evaluation of open-ended question answering
                            systems in their ability to answer questions without knowledge of the number of reasoning steps
                            required ahead of time. We show that IRRR not only achieves competitive performance with stateof-the-art models on the original SQuAD Open and
                            HotpotQA datasets, but also establishes a strong
                            baseline for this new dataset.
                            To recap, our contributions in this paper are: (1)
                            a new open-domain QA benchmark, BeerQA, that
                            features questions requiring variable steps of reasoning to answer on a unified Wikipedia corpus. (2) A
                            single unified neural network model that performs
                            all essential subtasks in open-domain QA purely
                            from text (retrieval, reranking, and reading comprehension), which not only achieves strong results
                            on SQuAD and HotpotQA, but also establishes a
                            strong baseline on this new benchmark"""
    },


    {
        "method_name" : "Recursive Dense Retriever",
        "joint_f1" : 66.55,
        "arxiv_link" : "https://arxiv.org/abs/2009.12756",
        "title" : """Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval""",
        "abstract" : """We propose a simple and efficient multi-hop dense retrieval approach for answering
                        complex open-domain questions, which achieves state-of-the-art performance on
                        two multi-hop datasets, HotpotQA and multi-evidence FEVER. Contrary to previous work, our method does not require access to any corpus-specific information,
                        such as inter-document hyperlinks or human-annotated entity markers, and can
                        be applied to any unstructured text corpus. Our system also yields a much better
                        efficiency-accuracy trade-off, matching the best published accuracy on HotpotQA
                        while being 10 times faster at inference time.""",
        "introduction" : """Open domain question answering is a challenging task where the answer to a given question needs to
                            be extracted from a large pool of documents. The prevailing approach (Chen et al., 2017) tackles the
                            problem in two stages. Given a question, a retriever first produces a list of k candidate documents,
                            and a reader then extracts the answer from this set. Until recently, retrieval models were dependent
                            on traditional term-based information retrieval (IR) methods, which fail to capture the semantics of
                            the question beyond lexical matching and remain a major performance bottleneck for the task. Recent
                            work on dense retrieval methods instead uses pretrained encoders to cast the question and documents
                            into dense representations in a vector space and relies on fast maximum inner-product search (MIPS)
                            to complete the retrieval. These approaches (Lee et al., 2019; Guu et al., 2020; Karpukhin et al.,
                            2020) have demonstrated significant retrieval improvements over traditional IR baselines.
                            However, such methods remain limited to simple questions, where the answer to the question
                            is explicit in a single piece of text evidence. In contrast, complex questions typically involve
                            aggregating information from multiple documents, requiring logical reasoning or sequential (multihop) processing in order to infer the answer (see Figure 1 for an example). Since the process for
                            answering such questions might be sequential in nature, single-shot approaches to retrieval are
                            insufficient. Instead, iterative methods are needed to recursively retrieve new information at each
                            step, conditioned on the information already at hand. Beyond further expanding the scope of existing
                            textual open-domain QA systems, answering more complex questions usually involves multi-hop
                            reasoning, which poses unique challenges for existing neural-based AI systems. With its practical and research values, multi-hop QA has been extensively studied recently (Talmor & Berant, 2018;
                            Yang et al., 2018; Welbl et al., 2018) and remains an active research area in NLP (Qi et al., 2019; Nie
                            et al., 2019; Min et al., 2019; Zhao et al., 2020; Asai et al., 2020; Perez et al., 2020).
                            The main problem in answering multi-hop open-domain questions is that the search space grows
                            exponentially with each retrieval hop. Most recent work tackles this issue by constructing a document
                            graph utilizing either entity linking or existing hyperlink structure in the underlying Wikipedia
                            corpus (Nie et al., 2019; Asai et al., 2020). The problem then becomes finding the best path in this
                            graph, where the search space is bounded by the number of hyperlinks in each passage. However,
                            such methods may not generalize to new domains, where entity linking might perform poorly, or
                            where hyperlinks might not be as abundant as in Wikipedia. Moreover, efficiency remains a challenge
                            despite using these data-dependent pruning heuristics, with the best model (Asai et al., 2020) needing
                            hundreds of calls to large pretrained models to produce a single answer.
                            In contrast, we propose to employ dense retrieval to the multi-hop setting with a simple recursive
                            framework. Our method iteratively encodes the question and previously retrieved documents as a
                            query vector and retrieves the next relevant documents using efficient MIPS methods. With highquality, dense representations derived from strong pretrained encoders, our work first demonstrates
                            that the sequence of documents that provide sufficient information to answer the multi-hop question
                            can be accurately discovered from unstructured text, without the help of corpus-specific hyperlinks.
                            When evaluated on two multi-hop benchmarks, HotpotQA (Yang et al., 2018) and a multi-evidence
                            subset of FEVER (Thorne et al., 2018), our approach improves greatly over the traditional linkingbased retrieval methods. More importantly, the better retrieval results also lead to state-of-the-art
                            downstream results on both datasets. On HotpotQA, we demonstrate a vastly improved efficiencyaccuracy trade-off achieved by our system: by limiting the amount of retrieved contexts fed into
                            downstream models, our system can match the best published result while being 10x faster."""
    },


    {
        "method_name" : "DDRQA",
        "joint_f1" : 63.88,
        "arxiv_link" : "https://arxiv.org/abs/2009.07465",
        "title" : """Answering Any-hop Open-domain Questions with Iterative Document Reranking""",
        "abstract" : """Existing approaches for open-domain question answering (QA)
                        are typically designed for questions that require either single-hop
                        or multi-hop reasoning, which make strong assumptions of the
                        complexity of questions to be answered. Also, multi-step document
                        retrieval often incurs higher number of relevant but non-supporting
                        documents, which dampens the downstream noise-sensitive reader
                        module for answer extraction. To address these challenges, we
                        propose a unified QA framework to answer any-hop open-domain
                        questions, which iteratively retrieves, reranks and filters documents,
                        and adaptively determines when to stop the retrieval process. To
                        improve the retrieval accuracy, we propose a graph-based reranking
                        model that perform multi-document interaction as the core of our
                        iterative reranking framework. Our method consistently achieves
                        performance comparable to or better than the state-of-the-art on
                        both single-hop and multi-hop open-domain QA datasets, including
                        Natural Questions Open, SQuAD Open, and HotpotQA.""",
        "introduction" : """Open-domain question answering (QA) requires a system to answer
                            factoid questions using a large text corpus (e.g., Wikipedia or the
                            Web) without any pre-defined knowledge schema. Most state-ofthe-art approaches for open-domain QA follow the retrieve-andread pipeline initiated by Chen et al. [3], using a retriever module
                            to retrieve relevant documents, and then a reader module to extract
                            answer from the retrieved documents. These approaches achieve
                            prominent results on single-hop QA datasets such as SQuAD [27],
                            whose questions can be answered using a single supporting document. However, they are inherently limited to answering simple
                            questions and not able to handle multi-hop questions, which require the system to retrieve and reason over evidence scattered
                            among multiple documents. In the task of open-domain multi-hop
                            QA [41], the documents with the answer can have little lexical overlap with the question and thus are not directly retrievable. Take the
                            question in Figure 1 as an example, the last paragraph contains the
                            correct answer but cannot be directly retrieved using TF-IDF. In
                            this example, the single-hop TF-IDF retriever is not able to retrieve
                            the last supporting paragraph since it has no lexical overlap with
                            the question, but this paragraph contains the answer and plays a
                            critical role in the reasoning chain.
                            Recent studies on multi-hop QA attempt to perform iterative
                            retrievals to improve the answer recall of the retrieved documents.
                            However, several challenges are not solved yet by existing multihop QA methods: 1) The iterative retrieval rapidly increases the
                            total number of retrieved documents and introduces much noise
                            to the downstream reader module for answer extraction. Typically,
                            the downstream reader module is noise-sensitive, which works
                            poorly when taking noisy documents as input or missing critical
                            supporting documents with the answer [23]. This requires the QA
                            system to reduce relevant but non-supporting documents fed into
                            the reader module. However, to answer open-domain multi-hop
                            questions, it is necessary to iteratively retrieve documents to increase the overall recall of supporting documents. This dilemma
                            poses a challenge for the retrieval phase of open-domain QA systems; 2) Existing multi-hop QA methods such as MUPPET [11] and
                            Multi-step Reasoner [4] perform a fixed number of retrieval steps,
                            which make strong assumptions on the complexity of open-domain
                            questions and perform fixed number of retrieval steps. In real-world
                            scenarios, open-domain questions may require different number of
                            reasoning steps; 3) The relevance of each retrieved document to the
                            question is independently considered. As exemplified in Figure 1,
                            ABR stands for ALBERT-base reranker, which serves as a reference of the retrieval performance of existing multi-hop QA methods
                            that independently consider the relevance of each document to the
                            question. Without considering multiple retrieved documents as a
                            whole, these methods can be easily biased to the lexical overlap
                            between each document and the question, and incorrectly classify
                            non-supporting documents as supporting evidence (such as the
                            middle two non-supporting paragraphs in Figure 1, which have
                            decent lexical overlap with the question) and vice versa (such as the
                            bottom paragraph in Figure 1, which has no lexical overlap with
                            the question but is a critical supporting document that contains the
                            answer).
                            To address the challenges above, we introduce a unified QA
                            framework for answering any-hop open-domain questions named
                            Iterative Document Reranking (IDR). Our framework learns to iteratively retrieve documents with updated question, rerank and filter
                            documents, and adaptively determine when to stop the retrieval
                            process. In this way, our method can significantly reduce the noise
                            introduced by multi-round retrievals and handle open-domain questions that require different number of reasoning steps. To avoid the
                            bias of lexical overlap in identifying supporting documents, our
                            framework considers the question and retrieved documents as a
                            whole and models the multi-document interactions to improve the
                            accuracy of classifying supporting documents. graph attention network (GAT). By leveraging the multi-document
                            information, our reranking model has more knowledge to differentiate supporting documents from irrelevant documents. After
                            initial retrieval, our method updates the question at every retrieval
                            step with a text span extracted from the retrieved documents, and
                            then use the updated question as query to retrieve complementary documents, which are added to the document graph for a new
                            round of interaction. The reranking model is reused to score the
                            documents again and filter the most irrelevant ones. The maintained high-quality shortlist of remaining documents are then fed
                            into the Reader Module to determine whether the answer exists in
                            them. If so, the retrieval process ends and the QA system delivers
                            the answer span extracted by the Reader Module as the predicted
                            answer. Otherwise, the retrieval process continues to the next hop.
                            Our contributions are summarized as follows:
                            • Noise control for iterative retrieval: We propose a novel QA method
                            to iteratively retrieve, rerank and filter documents, and adaptively
                            determine when to stop the retrieval process. Our method maintains a high-quality shortlist of remaining documents, which significantly reduces the noise introduced to the downstream reader
                            module for answer extraction. Thus, the downstream reader module can extract the answer span with higher accuracy.
                            • Unified framework for any-hop open-domain QA: We propose a
                            unified framework that does not require to pre-determine the
                            complexity of input questions. Different from existing QA methods that are specifically designed for either single-hop or fixedhop questions, our method can adaptively determine the termination of retrieval and answer any-hop open-domain questions.
                            • Multi-document interaction: We construct entity-linked document
                            graph and employ graph attention network for multi-document
                            interaction, which boosts up the reranking performance. To the
                            best of our knowledge, we are the first to propose graph-based
                            document reranking method for open-domain multi-hop QA."""
    },


    {
        "method_name" : "HGN + SemanticRetrievalMRS IR",
        "joint_f1" : 59.86,
        "arxiv_link" : "https://arxiv.org/abs/1911.03631",
        "title" : """Hierarchical Graph Network for Multi-hop Question Answering""",
        "abstract" : """In this paper, we present Hierarchical Graph
                        Network (HGN) for multi-hop question answering. To aggregate clues from scattered
                        texts across multiple paragraphs, a hierarchical graph is created by constructing nodes
                        on different levels of granularity (questions,
                        paragraphs, sentences, entities), the representations of which are initialized with pre-trained
                        contextual encoders. Given this hierarchical
                        graph, the initial node representations are updated through graph propagation, and multihop reasoning is performed via traversing
                        through the graph edges for each subsequent
                        sub-task (e.g., paragraph selection, supporting
                        facts extraction, answer prediction). By weaving heterogeneous nodes into an integral unified graph, this hierarchical differentiation of
                        node granularity enables HGN to support different question answering sub-tasks simultaneously. Experiments on the HotpotQA benchmark demonstrate that the proposed model
                        achieves new state of the art, outperforming existing multi-hop QA approaches.""",
        "introduction" : """In contrast to one-hop question answering (Rajpurkar et al., 2016; Trischler et al., 2016; Lai et al.,
                            2017) where answers can be derived from a single
                            paragraph (Wang and Jiang, 2017; Seo et al., 2017;
                            Liu et al., 2018; Devlin et al., 2019), many recent
                            studies on question answering focus on multi-hop
                            reasoning across multiple documents or paragraphs.
                            Popular tasks include WikiHop (Welbl et al., 2018),
                            ComplexWebQuestions (Talmor and Berant, 2018),
                            and HotpotQA (Yang et al., 2018).
                            An example from HotpotQA is illustrated in Figure 1. In order to correctly answer the question
                            (“The director of the romantic comedy ‘Big Stone
                            Gap’ is based in what New York city”), the model is required to first identify P1 as a relevant paragraph,
                            whose title contains the keywords that appear in the
                            question (“Big Stone Gap”). S1, the first sentence
                            of P1, is then chosen by the model as a supporting
                            fact that leads to the next-hop paragraph P2. Lastly,
                            from P2, the span “Greenwich Village, New York
                            City” is selected as the predicted answer.
                            Most existing studies use a retriever to find paragraphs that contain the right answer to the question
                            (P1 and P2 in this case). A Machine Reading Comprehension (MRC) model is then applied to the
                            selected paragraphs for answer prediction (Nishida
                            et al., 2019; Min et al., 2019b). However, even after
                            successfully identifying a reasoning chain through
                            multiple paragraphs, it still remains a critical challenge how to aggregate evidence from scattered sources on different granularity levels (e.g., paragraphs, sentences, entities) for joint answer and
                            supporting facts prediction.
                            To better leverage fine-grained evidences, some
                            studies apply entity graphs through query-guided
                            multi-hop reasoning. Depending on the characteristics of the dataset, answers can be selected either
                            from entities in the constructed entity graph (Song
                            et al., 2018; Dhingra et al., 2018; De Cao et al.,
                            2019; Tu et al., 2019; Ding et al., 2019), or from
                            spans in documents by fusing entity representations back into token-level document representation (Xiao et al., 2019). However, the constructed
                            graph is mostly used for answer prediction only,
                            while insufficient for finding supporting facts. Also,
                            reasoning through a simple entity graph (Ding et al.,
                            2019) or paragraph-entity hybrid graph (Tu et al.,
                            2019) lacks the ability to support complicated questions that require multi-hop reasoning.
                            Intuitively, given a question that requires multiple hops through a set of documents to reach
                            the right answer, a model needs to: (i) identify
                            paragraphs relevant to the question; (ii) determine
                            strong supporting evidence in those paragraphs;
                            and (iii) pinpoint the right answer following the
                            garnered evidence. To this end, Graph Neural Network with its inherent message passing mechanism
                            that can pass on multi-hop information through
                            graph propagation, has great potential of effectively
                            predicting both supporting facts and answer simultaneously for complex multi-hop questions.
                            Motivated by this, we propose a Hierarchical
                            Graph Network (HGN) for multi-hop question answering, which empowers joint answer/evidence
                            prediction via multi-level fine-grained graphs in
                            a hierarchical framework. Instead of only using
                            entities as nodes, for each question we construct
                            a hierarchical graph to capture clues from sources
                            with different levels of granularity. Specifically,
                            four types of graph node are introduced: questions,
                            paragraphs, sentences and entities (see Figure 2).
                            To obtain contextualized representations for these
                            hierarchical nodes, large-scale pre-trained language
                            models such as BERT (Devlin et al., 2019) and
                            RoBERTa (Liu et al., 2019) are used for contextual
                            encoding. These initial representations are then
                            passed through a Graph Neural Network for graph
                            propagation. The updated node representations are
                            then exploited for different sub-tasks (e.g., paragraph selection, supporting facts prediction, entity
                            prediction). Since answers may not be entities in
                            the graph, a span prediction module is also introduced for final answer prediction.
                            The main contributions of this paper are threefold: (i) We propose a Hierarchical Graph Network
                            (HGN) for multi-hop question answering, where
                            heterogeneous nodes are woven into an integral
                            hierarchical graph. (ii) Nodes from different granularity levels mutually enhance each other for different sub-tasks, providing effective supervision
                            signals for both supporting facts extraction and
                            answer prediction. (iii) On the HotpotQA benchmark, the proposed model achieves new state of
                            the art in both Distractor and Fullwiki settings."""
    },


    {
        "method_name" : "SemanticRetrievalMRS",
        "joint_f1" : 47.6,
        "arxiv_link" : "https://arxiv.org/abs/1909.08041",
        "title" : """Revealing the Importance of Semantic Retrieval for Machine Reading at Scale""",
        "abstract" : """Machine Reading at Scale (MRS) is a challenging task in which a system is given an
                        input query and is asked to produce a precise output by “reading” information from a
                        large knowledge base. The task has gained
                        popularity with its natural combination of information retrieval (IR) and machine comprehension (MC). Advancements in representation learning have led to separated progress in
                        both IR and MC; however, very few studies
                        have examined the relationship and combined
                        design of retrieval and comprehension at different levels of granularity, for development
                        of MRS systems. In this work, we give general guidelines on system design for MRS by
                        proposing a simple yet effective pipeline system with special consideration on hierarchical
                        semantic retrieval at both paragraph and sentence level, and their potential effects on the
                        downstream task. The system is evaluated on
                        both fact verification and open-domain multihop QA, achieving state-of-the-art results on
                        the leaderboard test sets of both FEVER and
                        HOTPOTQA. To further demonstrate the importance of semantic retrieval, we present ablation and analysis studies to quantify the contribution of neural retrieval modules at both
                        paragraph-level and sentence-level, and illustrate that intermediate semantic retrieval modules are vital for not only effectively filtering
                        upstream information and thus saving downstream computation, but also for shaping upstream data distribution and providing better
                        data for downstream modeling.1""",
        "introduction" : """Extracting external textual knowledge for machine
                            comprehensive systems has long been an important yet challenging problem. Success requires not only precise retrieval of the relevant information sparsely restored in a large knowledge source
                            but also a deep understanding of both the selected
                            knowledge and the input query to give the corresponding output. Initiated by Chen et al. (2017),
                            the task was termed as Machine Reading at Scale
                            (MRS), seeking to provide a challenging situation
                            where machines are required to do both semantic
                            retrieval and comprehension at different levels of
                            granularity for the final downstream task.
                            Progress on MRS has been made by improving individual IR or comprehension sub-modules
                            with recent advancements on representative learning (Peters et al., 2018; Radford et al., 2018; Devlin et al., 2018). However, partially due to the
                            lack of annotated data for intermediate retrieval in
                            an MRS setting, the evaluations were done mainly
                            on the final downstream task and with much less
                            consideration on the intermediate retrieval performance. This led to the convention that upstream
                            retrieval modules mostly focus on getting better
                            coverage of the downstream information such that
                            the upper-bound of the downstream score can be
                            improved, rather than finding more exact information. This convention is misaligned with the
                            nature of MRS where equal effort should be put
                            in emphasizing the models’ joint performance and
                            optimizing the relationship between the semantic
                            retrieval and the downstream comprehension subtasks.
                            Hence, to shed light on the importance of semantic retrieval for downstream comprehension
                            tasks, we start by establishing a simple yet effective hierarchical pipeline system for MRS using Wikipedia as the external knowledge source.
                            The system is composed of a term-based retrieval
                            module, two neural modules for both paragraphlevel retrieval and sentence-level retrieval, and a
                            neural downstream task module. We evaluated
                            the system on two recent large-scale open domain benchmarks for fact verification and multihop QA, namely FEVER (Thorne et al., 2018)
                            and HOTPOTQA (Yang et al., 2018), in which retrieval performance can also be evaluated accurately since intermediate annotations on evidences
                            are provided. Our system achieves the start-ofthe-art results with 45.32% for answer EM and
                            25.14% joint EM on HOTPOTQA (8% absolute
                            improvement on answer EM and doubling the
                            joint EM over the previous best results) and with
                            67.26% on FEVER score (3% absolute improvement over previously published systems).
                            We then provide empirical studies to validate
                            design decisions. Specifically, we prove the necessity of both paragraph-level retrieval and sentencelevel retrieval for maintaining good performance,
                            and further illustrate that a better semantic retrieval module not only is beneficial to achieving high recall and keeping high upper bound for
                            downstream task, but also plays an important role
                            in shaping the downstream data distribution and
                            providing more relevant and high-quality data for
                            downstream sub-module training and inference.
                            These mechanisms are vital for a good MRS system on both QA and fact verification."""
    },


    {
        "method_name" : "DrKIT",
        "joint_f1" : 42.88,
        "arxiv_link" : "https://arxiv.org/abs/2002.10640",
        "title" : """Differentiable Reasoning over a Virtual Knowledge Base""",
        "abstract" : """We consider the task of answering complex multi-hop questions using a corpus
                        as a virtual knowledge base (KB). In particular, we describe a neural module,
                        DrKIT, that traverses textual data like a KB, softly following paths of relations
                        between mentions of entities in the corpus. At each step the module uses a combination of sparse-matrix TFIDF indices and a maximum inner product search
                        (MIPS) on a special index of contextual representations of the mentions. This
                        module is differentiable, so the full system can be trained end-to-end using gradient based methods, starting from natural language inputs. We also describe a
                        pretraining scheme for the contextual representation encoder by generating hard
                        negative examples using existing knowledge bases. We show that DrKIT improves
                        accuracy by 9 points on 3-hop questions in the MetaQA dataset, cutting the gap
                        between text-based and KB-based state-of-the-art by 70%. On HotpotQA, DrKIT
                        leads to a 10% improvement over a BERT-based re-ranking approach to retrieving
                        the relevant passages required to answer a question. DrKIT is also very efficient,
                        processing 10-100x more queries per second than existing multi-hop systems.""",
        "introduction" : """Large knowledge bases (KBs), such as Freebase and WikiData, organize information around entities,
                            which makes it easy to reason over their contents. For example, given a query like “When was the
                            Grateful Dead’s lead singer born?”, one can identify the entity Grateful Dead and the path of
                            relations LeadSinger, BirthDate to efficiently extract the answer—provided that this information
                            is present in the KB. Unfortunately, KBs are often incomplete (Min et al., 2013). While relation
                            extraction methods can be used to populate KBs, this process is inherently error-prone, expensive
                            and slow.
                            Advances in open-domain QA (Moldovan et al., 2002; Yang et al., 2019) suggest an alternative—
                            instead of performing relation extraction, one could treat a large corpus as a virtual KB by answering
                            queries with spans from the corpus. This ensures facts are not lost in the relation extraction process,
                            but also poses challenges. One challenge is that it is relatively expensive to answer questions using
                            QA models which encode each document in a query-dependent fashion (Chen et al., 2017; Devlin
                            et al., 2019)—even with modern hardware (Strubell et al., 2019; Schwartz et al., 2019). The cost of
                            QA is especially problematic for certain complex questions, such as the example question above. If
                            the passages stating that “Jerry Garcia was the lead singer of the Grateful Dead” and “Jerry Garcia
                            was born in 1942” are far apart in the corpus, it is difficult for systems that retrieve and read a single
                            passage to find an answer—even though in this example, it might be easy to answer the question
                            after the relations were explicitly extracted into a KB. More generally, complex questions involving
                            sets of entities or paths of relations may require aggregating information from multiple documents,
                            which is expensive.
                            One step towards efficient QA is the recent work of Seo et al. (2018; 2019) on phrase-indexed question answering (PIQA), in which spans in the text corpus are associated with question-independent contextual representations and then indexed for fast retrieval. Natural language questions are then
                            answered by converting them into vectors that are used to perform maximum inner product search
                            (MIPS) against the index. This can be done efficiently using approximate algorithms (Shrivastava
                            & Li, 2014). However, this approach cannot be directly used to answer complex queries, since by
                            construction, the information stored in the index is about the local context around a span—it can
                            only be used for questions where the answer can be derived by reading a single passage.
                            This paper addresses this limitation of phrase-indexed question answering. We introduce an efficient,
                            end-to-end differentiable framework for doing complex QA over a large text corpus that has been
                            encoded in a query-independent manner. Specifically, we consider “multi-hop” complex queries
                            which can be answered by repeatedly executing a “soft” version of the operation below, defined
                            over a set of entities X and a relation R:
                            Y = X.follow(R) = {x
                            0
                            : ∃x ∈ X s.t. R(x, x0
                            ) holds}
                            In past work soft, differentiable versions of this operation were used to answer multi-hop questions
                            against an explicit KB (Cohen et al., 2019). Here we propose a more powerful neural module which
                            approximates this operation against an indexed corpus (a virtual KB). In our module, the input X is
                            a sparse-vector representing a weighted set of entities, and the relation R is a dense feature vector,
                            e.g. a vector derived from a neural network over a natural language query. X and R are used to
                            construct a MIPS query used for retrieving the top-K spans from the index. The output Y is another
                            sparse-vector representing the weighted set of entities, aggregated over entity mentions in the top-K
                            spans. We discuss pretraining schemes for the index in §2.3.
                            For multi-hop queries, the output entities Y can be recursively passed as input to the next iteration
                            of the same module. The weights of the entities in Y are differentiable w.r.t the MIPS queries, which
                            allows end-to-end learning without any intermediate supervision. We discuss an implementation
                            based on sparse-matrix-vector products, whose runtime and memory depend only on the number of
                            spans K retrieved from the index. This is crucial for scaling up to large corpora, providing up to 15x
                            faster inference than existing state-of-the-art multi-hop and open-domain QA systems. The system
                            we introduce is called DrKIT (for Differentiable Reasoning over a Knowledge base of Indexed
                            Text). We test DrKIT on the MetaQA benchmark for complex question answering, and show that
                            it improves on prior text-based systems by 5 points on 2-hop and 9 points on 3-hop questions,
                            reducing the gap between text-based and KB-based systems by 30% and 70%, respectively. We
                            also test DrKIT on a new dataset of multi-hop slot-filling over Wikipedia articles, and show that it
                            outperforms DrQA (Chen et al., 2017) and PIQA (Seo et al., 2019) adapted to this task. Finally, we
                            apply DrKIT to multi-hop information retrieval on the HotpotQA dataset (Yang et al., 2018), and
                            show that it significantly improves over a BERT-based reranking approach, while being 10x faster."""
    },


    {
        "method_name" : "GoldEn Retriever",
        "joint_f1" : 39.13,
        "arxiv_link" : "https://arxiv.org/abs/1910.07000",
        "title" : """Answering Complex Open-domain Questions Through Iterative Query Generation""",
        "abstract" : """It is challenging for current one-step retrieveand-read question answering (QA) systems to
                        answer questions like “Which novel by the author of ‘Armada’ will be adapted as a feature
                        film by Steven Spielberg?” because the question seldom contains retrievable clues about
                        the missing entity (here, the author). Answering such a question requires multi-hop reasoning where one must gather information about
                        the missing entity (or facts) to proceed with
                        further reasoning. We present GOLDEN (Gold
                        Entity) Retriever, which iterates between reading context and retrieving more supporting
                        documents to answer open-domain multi-hop
                        questions. Instead of using opaque and computationally expensive neural retrieval models,
                        GOLDEN Retriever generates natural language
                        search queries given the question and available
                        context, and leverages off-the-shelf information retrieval systems to query for missing entities. This allows GOLDEN Retriever to scale
                        up efficiently for open-domain multi-hop reasoning while maintaining interpretability. We
                        evaluate GOLDEN Retriever on the recently
                        proposed open-domain multi-hop QA dataset,
                        HOTPOTQA, and demonstrate that it outperforms the best previously published model despite not using pretrained language models
                        such as BERT.""",
        "introduction" : """Open-domain question answering (QA) is an important means for us to make use of knowledge
                            in large text corpora and enables diverse queries
                            without requiring a knowledge schema ahead of
                            time. Enabling such systems to perform multistep inferencecan further expand our capability to
                            explore the knowledge in these corpora (e.g., see
                            Figure 1). Fueled by the recently proposed large-scale QA
                            datasets such as SQuAD (Rajpurkar et al., 2016,
                            2018) and TriviaQA (Joshi et al., 2017), much
                            progress has been made in open-domain question
                            answering. Chen et al. (2017) proposed a twostage approach of retrieving relevant content with
                            the question, then reading the paragraphs returned
                            by the information retrieval (IR) component to arrive at the final answer. This “retrieve and read”
                            approach has since been adopted and extended in
                            various open-domain QA systems (Nishida et al.,
                            2018; Kratzwald and Feuerriegel, 2018), but it is
                            inherently limited to answering questions that do
                            not require multi-hop/multi-step reasoning. This
                            is because for many multi-hop questions, not all
                            the relevant context can be obtained in a single retrieval step (e.g., “Ernest Cline” in Figure 1).
                            More recently, the emergence of multi-hop
                            question answering datasets such as QAngaroo
                            (Welbl et al., 2018) and HOTPOTQA (Yang et al.,
                            2018) has sparked interest in multi-hop QA in the
                            research community. Designed to be more challenging than SQuAD-like datasets, they feature
                            questions that require context of more than one document to answer, testing QA systems’ abilities to infer the answer in the presence of multiple pieces of evidence and to efficiently find the
                            evidence in a large pool of candidate documents.
                            However, since these datasets are still relatively
                            new, most of the existing research focuses on the
                            few-document setting where a relatively small set
                            of context documents is given, which is guaranteed to contain the “gold” context documents, all
                            those from which the answer comes (De Cao et al.,
                            2019; Zhong et al., 2019).
                            In this paper, we present GOLDEN (Gold Entity) Retriever. Rather than relying purely on the
                            original question to retrieve passages, the central
                            innovation is that at each step the model also uses
                            IR results from previous hops of reasoning to generate a new natural language query and retrieve
                            new evidence to answer the original question. For
                            the example in Figure 1, GOLDEN would first generate a query to retrieve Armada (novel) based on
                            the question, then query for Ernest Cline based on
                            newly gained knowledge in that article. This allows GOLDEN to leverage off-the-shelf, generalpurpose IR systems to scale open-domain multihop reasoning to millions of documents efficiently,
                            and to do so in an interpretable manner. Combined with a QA module that extends BiDAF++
                            (Clark and Gardner, 2017), our final system outperforms the best previously published system on
                            the open-domain (fullwiki) setting of HOTPOTQA
                            without using powerful pretrained language models like BERT (Devlin et al., 2019).
                            The main contributions of this paper are: (a)
                            a novel iterative retrieve-and-read framework capable of multi-hop reasoning in open-domain
                            QA1
                            (b) a natural language query generation
                            approach that guarantees interpretability in the
                            multi-hop evidence gathering process; (c) an efficient training procedure to enable query generation with minimal supervision signal that significantly boosts recall of gold supporting documents
                            in retrieval."""
    },


    {
        "method_name" : "Cognitive Graph QA",
        "joint_f1" : 34.92,
        "arxiv_link" : "https://arxiv.org/abs/1905.05460",
        "title" : """Cognitive Graph for Multi-Hop Reading Comprehension at Scale""",
        "abstract" : """We propose a new CogQA framework for
                        multi-hop question answering in web-scale
                        documents. Founded on the dual process theory in cognitive science, the framework gradually builds a cognitive graph in an iterative
                        process by coordinating an implicit extraction module (System 1) and an explicit reasoning module (System 2). While giving accurate answers, our framework further provides explainable reasoning paths. Specifically, our implementation1 based on BERT
                        and graph neural network (GNN) efficiently
                        handles millions of documents for multi-hop
                        reasoning questions in the HotpotQA fullwiki
                        dataset, achieving a winning joint F1 score of
                        34.9 on the leaderboard, compared to 23.6 of
                        the best competitor""",
        "introduction" : """Deep learning models have made significant
                            strides in machine reading comprehension and
                            even outperformed human on single paragraph
                            question answering (QA) benchmarks including
                            SQuAD (Wang et al., 2018b; Devlin et al., 2018;
                            Rajpurkar et al., 2016). However, to cross the
                            chasm of reading comprehension ability between
                            machine and human, three main challenges lie
                            ahead: 1) Reasoning ability. As revealed by adversarial tests (Jia and Liang, 2017), models for
                            single paragraph QA tend to seek answers in sentences matched by the question, which does not
                            involve complex reasoning. Therefore, multi-hop
                            QA becomes the next frontier to conquer (Yang
                            et al., 2018). 2) Explainability. Explicit reasoning paths, which enable verification of logical rigor, are vital for the reliability of QA systems. HotpotQA (Yang et al., 2018) requires models to provide supporting sentences, which
                            means unordered and sentence-level explainability, yet humans can interpret answers with step by
                            step solutions, indicating an ordered and entitylevel explainability. 3) Scalability. For any practically useful QA system, scalability is indispensable. Existing QA systems based on machine comprehension generally follow retrievalextraction framework in DrQA (Chen et al., 2017),
                            reducing the scope of sources to a few paragraphs
                            by pre-retrieval. This framework is a simple compromise between single paragraph QA and scalable information retrieval, compared to human’s
                            ability to breeze through reasoning with knowledge in massive-capacity memory (Wang et al.,
                            2003).
                            Therefore, insights on the solutions to these
                            challenges can be drawn from the cognitive process of humans. Dual process theory (Evans,
                            1984, 2003, 2008; Sloman, 1996) suggests that our
                            brains first retrieve relevant information following attention via an implicit, unconscious and intuitive process called System 1, based on which another explicit, conscious and controllable reasoning process, System 2, is then conducted. System
                            1 could provide resources according to requests,
                            while System 2 enables diving deeper into relational information by performing sequential thinking in the working memory, which is slower but
                            with human-unique rationality (Baddeley, 1992).
                            For complex reasoning, the two systems are coordinated to perform fast and slow thinking (Kahneman and Egan, 2011) iteratively.
                            In this paper, we propose a framework, namely
                            Cognitive Graph QA (CogQA), contributing to
                            tackling all challenges above. Inspired by the dual
                            process theory, the framework comprises functionally different System 1 and 2 modules. System 1
                            extracts question-relevant entities and answer candidates from paragraphs and encodes their semantic information. Extracted entities are organized
                            as a cognitive graph (Figure 1), which resembles
                            the working memory. System 2 then conducts the
                            reasoning procedure over the graph, and collects
                            clues to guide System 1 to better extract next-hop
                            entities. The above process is iterated until all
                            possible answers are found, and then the final answer is chosen based on reasoning results from
                            System 2. An efficient implementation based on
                            BERT (Devlin et al., 2018) and graph neural network (GNN) (Battaglia et al., 2018) is introduced.
                            Our contributions are as follows:
                            • We propose the novel CogQA framework
                            for multi-hop reading comprehension QA at
                            scale according to human cognition.
                            • We show that the cognitive graph structure
                            in our framework offers ordered and entitylevel explainability and suits for relational
                            reasoning.
                            • Our implementation based on BERT and
                            GNN surpasses previous works and other
                            competitors substantially on all the metrics."""
    },

    {
        "method_name" : "MUPPET",
        "joint_f1" : 27.01,
        "arxiv_link" : "https://arxiv.org/abs/1906.06606",
        "title" : """Multi-Hop Paragraph Retrieval for Open-Domain Question Answering""",
        "abstract" : """This paper is concerned with the task of
                        multi-hop open-domain Question Answering
                        (QA). This task is particularly challenging
                        since it requires the simultaneous performance
                        of textual reasoning and efficient searching.
                        We present a method for retrieving multiple
                        supporting paragraphs, nested amidst a large
                        knowledge base, which contain the necessary
                        evidence to answer a given question. Our
                        method iteratively retrieves supporting paragraphs by forming a joint vector representation of both a question and a paragraph. The
                        retrieval is performed by considering contextualized sentence-level representations of the
                        paragraphs in the knowledge source. Our
                        method achieves state-of-the-art performance
                        over two well-known datasets, SQuAD-Open
                        and HotpotQA, which serve as our single- and
                        multi-hop open-domain QA benchmarks, respectively.""",
        "introduction" : """Textual Question Answering (QA) is the task of
                            answering natural language questions given a set
                            of contexts from which the answers to these questions can be inferred. This task, which falls under the domain of natural language understanding, has been attracting massive interest due to extremely promising results that were achieved using deep learning techniques. These results were
                            made possible by the recent creation of a variety of
                            large-scale QA datasets, such as TriviaQA (Joshi
                            et al., 2017) and SQuAD (Rajpurkar et al., 2016).
                            The latest state-of-the-art methods are even capable of outperforming humans on certain tasks (Devlin et al., 2018)
                            2
                            .
                            The basic and arguably the most popular task of
                            QA is often referred to as Reading Comprehension (RC), in which each question is paired with a relatively small number of paragraphs (or documents)
                            from which the answer can potentially be inferred.
                            The objective in RC is to extract the correct answer from the given contexts or, in some cases,
                            deem the question unanswerable (Rajpurkar et al.,
                            2018). Most large-scale RC datasets, however, are
                            built in such a way that the answer can be inferred
                            using a single paragraph or document. This kind
                            of reasoning is termed single-hop reasoning, since
                            it requires reasoning over a single piece of evidence. A more challenging task, called multi-hop
                            reasoning, is one that requires combining evidence
                            from multiple sources (Talmor and Berant, 2018;
                            Welbl et al., 2018; Yang et al., 2018). Figure 1
                            provides an example of a question requiring multihop reasoning. To answer the question, one must
                            first infer from the first context that Alex Ferguson
                            is the manager in question, and only then can the
                            answer to the question be inferred with any confidence from the second context.
                            Another setting for QA is open-domain QA, in
                            which questions are given without any accompanying contexts, and one is required to locate the
                            relevant contexts to the questions from a large
                            knowledge source (e.g., Wikipedia), and then extract the correct answer using an RC component.
                            This task has recently been resurged following
                            the work of Chen et al. (2017), who used a TFIDF based retriever to find potentially relevant
                            documents, followed by a neural RC component
                            that extracted the most probable answer from the
                            retrieved documents. While this methodology
                            performs reasonably well for questions requiring
                            single-hop reasoning, its performance decreases
                            significantly when used for open-domain multihop reasoning.
                            We propose a new approach to accomplishing
                            this task, called iterative multi-hop retrieval, in
                            which one iteratively retrieves the necessary evidence to answer a question. We believe this iterative framework is essential for answering multihop questions, due to the nature of their reasoning
                            requirements.
                            Our main contributions are the following:
                            • We propose a novel multi-hop retrieval approach, which we believe is imperative for
                            truly solving the open-domain multi-hop QA
                            task.
                            • We show the effectiveness of our approach,
                            which achieves state-of-the-art results in
                            both single- and multi-hop open-domain QA
                            benchmarks.
                            • We also propose using sentence-level representations for retrieval, and show the possible
                            benefits of this approach over paragraph-level
                            representations.
                            While there are several works that discuss solutions for multi-hop reasoning (Dhingra et al.,
                            2018; Zhong et al., 2019), to the best of our knowledge, this work is the first to propose a viable solution for open-domain multi-hop QA."""
    },


    {
        "method_name" : "KGNN",
        "joint_f1" : 24.66,
        "arxiv_link" : "https://arxiv.org/abs/1911.02170",
        "title" : """Multi-Paragraph Reasoning with Knowledge-enhanced Graph Neural Network""",
        "abstract" : """Multi-paragraph reasoning is indispensable for
                        open-domain question answering (OpenQA),
                        which receives less attention in the current
                        OpenQA systems. In this work, we propose
                        a knowledge-enhanced graph neural network
                        (KGNN), which performs reasoning over multiple paragraphs with entities. To explicitly
                        capture the entities’ relatedness, KGNN utilizes relational facts in knowledge graph to
                        build the entity graph. The experimental results show that KGNN outperforms in both
                        distractor and full wiki settings than baselines
                        methods on HotpotQA dataset. And our further analysis illustrates KGNN is effective and
                        robust with more retrieved paragraphs.""",
        "introduction" : """Open-domain question answering (OpenQA) aims
                            to answer questions based on large-scale knowledge source, such as an unlabelled corpus. Recent
                            years, OpenQA has aroused the interest of many
                            researchers, with the availability of large-scale
                            datasets such as Quasar (Dhingra et al., 2017),
                            SearchQA (Dunn et al., 2017), TriviaQA (Joshi
                            et al., 2017), etc. They proposed lots of OpenQA
                            models (Chen et al., 2017; Clark and Gardner,
                            2018; Wang et al., 2018a,b; Choi et al., 2017; Lin
                            et al., 2018) which achieved promising results in
                            various public benchmarks.
                            However, most questions in previous OpenQA
                            datasets only require reasoning within a single
                            paragraph or a single-hop over paragraphs. The
                            HotpotQA dataset (Yang et al., 2018) was constructed to facilitate the development of OpenQA
                            system in handling multi-paragraph reasoning.
                            Multi-paragraph reasoning is an important and
                            practical problem towards a more intelligent
                            OpenQA. Nevertheless, existing OpenQA systems
                            have not paid enough attention to multi-paragraph
                            reasoning. They generally fall into two categories when dealing with multiple paragraphs: (1) regarding each paragraph as an individual which
                            cannot reason over paragraphs; (2) concatenating
                            all paragraphs into a single long text which leads
                            to time and memory consuming.
                            To achieve a multi-paragraph reasoning system,
                            we propose a knowledge-enhanced graph neural
                            network (KGNN). First, we build an entity graph
                            by all named entities from paragraphs, and add coreference edges to the graph if the entity appears in
                            different paragraphs. After that, to explicitly capture the entities’ relatedness, we further utilize the
                            relational facts in knowledge graph (KG) to build
                            the relational entity graph for reasoning, i.e, add
                            a relation edge to the graph if two entities have
                            a relation in KG. We believe that the reasoning
                            information can be captured through propagation over a relational entity graph. As the example in
                            Figure 1, for the given entity Wildest Dreams, we
                            require two kinds of the one-hop reasoning to obtain the answer: One is Wildest Dreams appears in
                            multi-paragraph and the other is reasoning based
                            on the relational fact (Wildest Dreams,lyrics by,
                            Max martin and Shellback).
                            Our main contribution is that we propose a
                            novel reasoning module combined with knowledge. The experiments show that reasoning over
                            entities can help our model surpass all baseline
                            models significantly on HotpotQA dataset. Our
                            analysis demonstrates that KGNN is robust and
                            has a strong ability to handle massive texts."""
    },


    {
        "method_name" : "QFE",
        "joint_f1" : 23.10,
        "arxiv_link" : "https://arxiv.org/abs/1905.08511",
        "title" : """Answering while Summarizing: Multi-task Learning for Multi-hop QA with Evidence Extraction""",
        "abstract" : """Question answering (QA) using textual
                        sources for purposes such as reading comprehension (RC) has attracted much attention.
                        This study focuses on the task of explainable
                        multi-hop QA, which requires the system to
                        return the answer with evidence sentences
                        by reasoning and gathering disjoint pieces
                        of the reference texts. It proposes the Query
                        Focused Extractor (QFE) model for evidence
                        extraction and uses multi-task learning with
                        the QA model. QFE is inspired by extractive
                        summarization models; compared with the
                        existing method, which extracts each evidence
                        sentence independently, it sequentially extracts evidence sentences by using an RNN
                        with an attention mechanism on the question
                        sentence. It enables QFE to consider the dependency among the evidence sentences and
                        cover important information in the question
                        sentence. Experimental results show that QFE
                        with a simple RC baseline model achieves a
                        state-of-the-art evidence extraction score on
                        HotpotQA. Although designed for RC, it also
                        achieves a state-of-the-art evidence extraction
                        score on FEVER, which is a recognizing
                        textual entailment task on a large textual
                        database.""",
        "introduction" : """Reading comprehension (RC) is a task that uses
                            textual sources to answer any question. It has seen
                            significant progress since the publication of numerous datasets such as SQuAD (Rajpurkar et al.,
                            2016). To achieve the goal of RC, systems must
                            be able to reason over disjoint pieces of information in the reference texts. Recently, multi-hop
                            question answering (QA) datasets focusing on this
                            capability, such as QAngaroo (Welbl et al., 2018)
                            and HotpotQA (Yang et al., 2018), have been released.
                            Multi-hop QA faces two challenges. The first
                            is the difficulty of reasoning. It is difficult for the
                            Figure 1: Concept of explainable multi-hop QA. Given
                            a question and multiple textual sources, the system extracts evidence sentences from the sources and returns
                            the answer and the evidence.
                            system to find the disjoint pieces of information
                            as evidence and reason using the multiple pieces
                            of such evidence. The second challenge is interpretability. The evidence used to reason is not necessarily located close to the answer, so it is difficult for users to verify the answer.
                            Yang et al. (2018) released HotpotQA, an explainable multi-hop QA dataset, as shown in Figure 1. Hotpot QA provides the evidence sentences
                            of the answer for supervised learning. The evidence extraction in multi-hop QA is more difficult
                            than that in other QA problems because the question itself may not provide a clue for finding evidence sentences. As shown in Figure 1, the system finds an evidence sentence (Evidence 2) by relying on another evidence sentence (Evidence 1).
                            The capability of being able to explicitly extract
                            evidence is an advance towards meeting the above
                            two challenges.
                            Here, we propose a Query Focused Extractor
                            (QFE) that is based on a summarization model.
                            We regard the evidence extraction of the explainable multi-hop QA as a query-focused summarization task. Query-focused summarization is the
                            task of summarizing the source document with regard to the given query. QFE sequentially extracts
                            the evidence sentences by using an RNN with an attention mechanism on the question sentence,
                            while the existing method extracts each evidence
                            sentence independently. This query-aware recurrent structure enables QFE to consider the dependency among the evidence sentences and cover the
                            important information in the question sentence.
                            Our overall model uses multi-task learning with a
                            QA model for answer selection and QFE for evidence extraction. The multi-task learning with
                            QFE is general in the sense that it can be combined
                            with any QA model.
                            Moreover, we find that the recognizing textual
                            entailment (RTE) task on a large textual database,
                            FEVER (Thorne et al., 2018), can be regarded as
                            an explainable multi-hop QA task. We confirm
                            that QFE effectively extracts the evidence both on
                            HotpotQA for RC and on FEVER for RTE.
                            Our main contributions are as follows.
                            • We propose QFE for explainable multi-hop
                            QA. We use the multi-task learning of the QA
                            model for answer selection and QFE for evidence extraction.
                            • QFE adaptively determines the number of evidence sentences by considering the dependency among the evidence sentences and the
                            coverage of the question.
                            • QFE achieves state-of-the-art performance
                            on both HotpotQA and FEVER in terms of
                            the evidence extraction score and comparable
                            performance to competitive models in terms
                            of the answer selection score. QFE is the first
                            model that outperformed the baseline on HotpotQA."""
    }
]