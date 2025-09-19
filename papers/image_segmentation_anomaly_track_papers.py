

papers = [
    {
        "method_name" : "FlowCLAS",
        "mean_F1" : 72.97,
        "arxiv_link" : "https://arxiv.org/abs/2411.19888",
        "title" : """FlowCLAS: Enhancing Normalizing Flow Via Contrastive Learning For Anomaly Segmentation""",
        "abstract" : """Anomaly segmentation is a valuable computer vision
                        task for safety-critical applications that need to be aware of
                        unexpected events. Current state-of-the-art (SOTA) scenelevel anomaly segmentation approaches rely on diverse inlier class labels during training, limiting their ability to
                        leverage vast unlabeled datasets and pre-trained vision
                        encoders. These methods may underperform in domains
                        with reduced color diversity and limited object classes.
                        Conversely, existing unsupervised methods struggle with
                        anomaly segmentation with the diverse scenes of less restricted domains. To address these challenges, we introduce FlowCLAS, a novel self-supervised framework that
                        utilizes vision foundation models to extract rich features
                        and employs a normalizing flow network to learn their
                        density distribution. We enhance the model’s discriminative power by incorporating Outlier Exposure and contrastive learning in the latent space. FlowCLAS significantly outperforms all existing methods on the ALLO
                        anomaly segmentation benchmark for space robotics and
                        demonstrates competitive results on multiple road anomaly
                        segmentation benchmarks for autonomous driving, including Fishyscapes Lost&Found and Road Anomaly. These results highlight FlowCLAS’s effectiveness in addressing the
                        unique challenges of space anomaly segmentation while retaining SOTA performance in the autonomous driving domain without reliance on inlier segmentation labels.""",
        "introduction" : """Anomaly segmentation is an essential computer vision
                            task in robotics that focuses on detecting and localizing objects that deviate from expected patterns present in training
                            images. By identifying out-of-distribution (OoD) visual elements that fall outside of predefined classes, anomaly segmentation complements traditional closed-set semantic segmentation, offering a more comprehensive understanding of complex visual scenes. This capability is highly valuable for safety-critical applications, of which we study two
                            examples in this work. In autonomous driving, anomaly
                            segmentation can help prevent collisions with unexpected
                            objects like exotic animals or costumed children, while in
                            space robotics, it can safeguard against potential damage to
                            robotic arms from unexpected objects, thereby enhancing
                            operational safety and efficiency [35]. The development
                            of robust anomaly segmentation methods is of significant
                            interest to many safety-critical robotics applications, as it
                            contributes to risk mitigation and robust robotic operations.
                            State-of-the-art (SOTA) autonomous driving anomaly
                            segmentation methods [16, 38, 40, 54] typically rely on diverse inlier class labels, such as those from the Cityscapes
                            dataset [14] with 19 classes, to learn discriminative feature spaces and class boundaries. However, this approach
                            limits their ability to leverage vast unlabeled datasets and
                            pre-trained vision foundation models that offer generalizable features for various downstream tasks. Moreover, these
                            methods are ill-suited for domains characterized by limited pixel and label spaces, as exists in the space domain. Images in the ALLO dataset [35] feature reduced color variation and fewer object classes, with only foreground and
                            background labels available in the training set. This lack of
                            diversity in both pixel and label spaces renders conventional
                            supervised road anomaly segmentation methods ineffective
                            for direct application in such domains, as we demonstrate
                            in our results.
                            An alternative approach involves the use of unsupervised
                            and self-supervised anomaly segmentation methods. Existing techniques [49, 52, 53], primarily developed for industrial defect inspection and medical imaging, fail to address the challenges of scene-level anomaly segmentation
                            in dynamic environments typical of autonomous driving
                            and space robotics. Recent research has shown that these
                            methods, designed for static, consistent-perspective images,
                            struggle to adapt to the diverse scene changes inherent in
                            space imagery [35]. To our knowledge, no existing unsupervised or self-supervised methods have been proposed for
                            images with highly variable scenes.
                            To address these challenges, we propose a novel
                            self-supervised approach that enhances normalizing Flow
                            via Contrastive Learning for Anomaly Segmentation
                            (FlowCLAS). FlowCLAS leverages a frozen vision foundation model to extract discriminative features and trains a
                            normalizing flow network to estimate feature density. Unlike existing unsupervised flow-based methods [25, 34, 47,
                            52], our self-supervised approach incorporates Outlier Exposure (OE), mixing pseudo-outlier objects from an auxiliary dataset into source anomaly-free images and employs
                            contrastive learning to separate normal and anomalous features in the latent space, improving anomaly segmentation performance on both road and space images. Our
                            method effectively performs scene-level anomaly segmentation without requiring inlier class labels, addressing the
                            limitations of both current unsupervised and supervised approaches. This innovation enables FlowCLAS to tackle the
                            unique challenges posed by variable scenes across different
                            domains. Our key contributions are:
                            • We introduce FlowCLAS, a novel self-supervised flowbased density estimation framework that integrates contrastive learning with pseudo-outliers for enhanced scenelevel anomaly segmentation.
                            • We show how integrating a contrastive learning scheme
                            with standard inlier likelihood maximization into FlowCLAS improves outlier latent separation, leading to enhanced scene-level anomaly segmentation performance.
                            • FlowCLAS demonstrates cross-domain generalizability by achieving SOTA performance on the ALLO
                            benchmark and outperforming existing unsupervised approaches on two road anomaly datasets: Fishyscapes
                            Lost&Found and Road Anomaly. Our method also
                            achieves SOTA AUPRC on Road Anomaly and matches
                            the performance of recent supervised approaches on other
                            metrics without requiring labels."""
    },

    {
        "method_name" : "VL4AD",
        "mean_F1" : 65.4,
        "arxiv_link" : "https://arxiv.org/abs/2409.17330",
        "title" : """VL4AD: Vision-Language Models Improve Pixel-wise Anomaly Detection""",
        "abstract" : """Semantic segmentation networks have achieved significant
                        success under the assumption of independent and identically distributed
                        data. However, these networks often struggle to detect anomalies from
                        unknown semantic classes due to the limited set of visual concepts they
                        are typically trained on. To address this issue, anomaly segmentation
                        often involves fine-tuning on outlier samples, necessitating additional efforts for data collection, labeling, and model retraining. Seeking to avoid
                        this cumbersome work, we take a different approach and propose to incorporate vision-language (VL) encoders into existing anomaly detectors
                        to leverage the semantically broad VL pre-training for improved outlier
                        awareness. Additionally, we propose a new scoring function that enables
                        data- and training-free outlier supervision via textual prompts. The resulting VL4AD model, which includes max-logit prompt ensembling and
                        a class-merging strategy, achieves competitive performance on widely
                        used benchmark datasets, thereby demonstrating the potential of visionlanguage models for pixel-wise anomaly detection.""",
        "introduction" : """Recent advances in deep neural networks (DNNs) have led to significant improvements in semantic segmentation tasks for urban driving scenes [10, 11, 24, 49],
                            especially when the semantic classes of training and testing are well aligned [39].
                            In real-world situations, however, unexpected object types, that were not part
                            of the training data, appear during operations due to long-tailed class distributions. Examples include wild animals on roads or objects falling from cars on
                            highways. Existing semantic segmentation networks often fail to detect such
                            objects, leading to unreliable predictions that could result in collisions and
                            traffic accidents. A standard technique to address this issue is anomaly detection [1,4,9,12,18,31,32,38,47], which differentiates between objects that fall into
                            in-distribution (ID) classes a model knows from training and objects that do not (out-of-distribution (OOD) or outlier). Nevertheless, these anomaly detectors
                            come with certain drawbacks. As illustrated in Fig. 1, compared with DNNs not
                            designated for anomaly detection (left), many anomaly detectors [9,12,18,38,47]
                            (middle) enhance the separability between ID and OOD by fine-tuning on OOD
                            data. A procedure often referred to as outlier supervision guided by negative
                            data. This approach necessitates extensive data collection and labeling as well as
                            retraining of models and often sacrifices a small but non-negligible amount of performance on ID data. Moreover, these models can only reject OOD samples similar to the collected negative data and likely fail on other types of OOD inputs.
                            Seeking to avoid these drawbacks, we present a method called the VisionLanguage Model for Anomaly Detection (VL4AD). It incorporates CLIP’s [41]
                            vision and text encoders into existing anomaly detectors. Vision-language models
                            are typically exposed to a broader range of visual concepts during pre-training
                            compared to the above mentioned vision-only models [41]. Previous work on image classification has shown that frozen CLIP models are as competitive as many
                            sophisticated vision-only models in a zero-shot manner [37]. We aim to leverage these advantageous generalization abilities of CLIP for improved pixel-level
                            OOD-awareness without outlier supervision. Additionally, since vision-language
                            models can handle textual input, we can utilize textual prompts to achieve dataand training-free outlier supervision, thereby increasing flexibility in real-world
                            applications. Our contributions are as follows: (1) we develop a method that applies FC-CLIP-type [52] vision-language models to detect anomalous objects at
                            the pixel level. (2) Subsequently, we introduce a strategy that combines i) maxlogit prompt ensembling for a better alignment between the ID textual and
                            visual embeddings with ii) class merging to reduce the estimated uncertainty of
                            edge pixels between ID class regions. (3) We propose a new scoring function that
                            enables data- and training-free outlier supervision via textual prompts. We evaluate our models on RoadAnomaly19 (RA19) [32], FishyScapes Lost and Found
                            (FS LaF) [6], and the Segment-Me-If-You-Can (SMIYC) dataset [8], achieving
                            competitive performance."""
    },

    {
        "method_name" : "EAM",
        "mean_F1" : 60.86,
        "arxiv_link" : "https://arxiv.org/abs/2301.03407",
        "title" : """On Advantages of Mask-level Recognition for Outlier-aware Segmentation""",
        "abstract" : """Most dense recognition approaches bring a separate decision in each particular pixel. These approaches deliver
                        competitive performance in usual closed-set setups. However, important applications in the wild typically require
                        strong performance in presence of outliers. We show that
                        this demanding setup greatly benefit from mask-level predictions, even in the case of non-finetuned baseline models.
                        Moreover, we propose an alternative formulation of dense
                        recognition uncertainty that effectively reduces false positive responses at semantic borders. The proposed formulation produces a further improvement over a very strong
                        baseline and sets the new state of the art in outlier-aware
                        semantic segmentation with and without training on negative data. Our contributions also lead to performance improvement in a recent panoptic setup. In-depth experiments
                        confirm that our approach succeeds due to implicit aggregation of pixel-level cues into mask-level predictions.""",
        "introduction" : """Emergence of deep learning revolutionized the field of
                            computer vision [34]. Complex yet efficient deep networks
                            advanced the capability of machines to understand scenes
                            [20,61]. Segmentation is a very important form of scene understanding due to its applications in medicine, agriculture,
                            robotics and the automotive industry. In the last decade,
                            segmentation tasks were modelled as per-pixel classification [20, 44]. However, such approach assumes independence of neighbouring pixels, which does not hold in practice. Neighbouring pixels are usually strongly correlated
                            due to belonging to the same object or scene part [39]. Albeit designed and trained with false assumption on independence of neighbouring pixels, the obtained models deliver
                            competitive generalization performance in in-distribution
                            scenes [14,15]. However, their real-world performance still
                            leaves much to be desired due to insufficient handling of the
                            out-of-taxonomy scene parts [6, 11].
                            A recent approach to per-pixel classification decouples
                            localization from recognition [17]. The localization is carried out by assigning pixels to an abundant set of masks,
                            each trained to capture semantically related regions (e.g. a
                            road or a building). The recovered semantic regions are subsequently classified as a whole. The described approach is
                            dubbed mask-level recognition [16]. Decoupling localization from classification further enables utilizing the same
                            model for semantic, instance and panoptic segmentation.
                            The shared architecture performs competitively on standard
                            segmentation benchmarks [18, 39, 64].
                            However, prior work does not consider demanding applications of mask-based approaches. Thus, we investigate
                            the value of mask-level recognition in some of the last major remaining challenges towards scene understanding in
                            the wild - outlier-aware semantic segmentation [7, 11, 29]
                            and outlier-aware panoptic segmentation [29]. Our experiments reveal strong performance of mask-level approaches
                            in these challenges. We investigate the reasons behind such
                            behaviour and contribute improvements that support these
                            important applications.
                            Mask-level recognition has several interesting properties. For instance, masks are classified into K known classes
                            and the class void, while mask assignments are not mutually
                            exclusive [17]. This provides more opportunity to reject
                            predictions than in standard per-pixel approaches. Masklevel approaches can propagate mask-level uncertainty to
                            the pixel-level. This is different from the standard approach
                            which has to estimate independent anomaly scores in each
                            pixel [26]. Obviously, the standard approach can easily ignore the local correlations in a pixel neighborhood, which
                            does not seem desirable. In terms of scalability, mask-level
                            recognition models do not require per-class feature maps at
                            the output resolution. This allows designers to decrease the
                            training footprint [8] and increase the flexibility of training.
                            All these properties make mask-level recognition a compelling research topic.
                            This paper proposes the following contributions. We
                            point out that mask-level recognition delivers strong baseline performance on standard benchmarks for outlier-aware
                            segmentation. Our improvements further exploit the specific bias of mask-level recognition. Combining the proposed EAM outlier detector with negative supervision attains competitive results in outlier-aware semantic and
                            panoptic segmentation. Further improvements can be obtained by combining the proposed approach with negative
                            supervision. The resulting models set the new state of the
                            art in outlier-aware segmentation on two tracks from the
                            Segment Me If You Can (SMIYC) benchmark and adapted
                            MS COCO."""
    },

    {
        "method_name" : "SOTA-RbA",
        "mean_F1" : 55.38,
        "arxiv_link" : "https://arxiv.org/abs/2504.19183",
        "title" : """Segmenting Objectiveness and Task-awareness Unknown Region for Autonomous Driving""",
        "abstract" : """With the emergence of transformer-based architectures and large
                        language models (LLMs), the accuracy of road scene perception has
                        substantially advanced. Nonetheless, current road scene segmentation approaches are predominantly trained on closed-set data,
                        resulting in insufficient detection capabilities for out-of-distribution
                        (OOD) objects. To overcome this limitation, road anomaly detection
                        methods have been proposed. However, existing methods primarily depend on image inpainting and OOD distribution detection
                        techniques, facing two critical issues: (1) inadequate consideration of the objectiveness attributes of anomalous regions, causing incomplete segmentation when anomalous objects share similarities
                        with known classes, and (2) insufficient attention to environmental constraints, leading to the detection of anomalies irrelevant
                        to autonomous driving tasks. In this paper, we propose a novel
                        framework termed Segmenting Objectiveness and Task-Awareness
                        (SOTA) for autonomous driving scenes. Specifically, SOTA enhances
                        the segmentation of objectiveness through a Semantic Fusion Block
                        (SFB) and filters anomalies irrelevant to road navigation tasks using
                        a Scene-understanding Guided Prompt-Context Adaptor (SG-PCA).
                        Extensive empirical evaluations on multiple benchmark datasets, including Fishyscapes Lost and Found, Segment-Me-If-You-Can, and
                        RoadAnomaly, demonstrate that the proposed SOTA consistently
                        improves OOD detection performance across diverse detectors,
                        achieving robust and accurate segmentation outcomes.""",
        "introduction" : """The integration of transformer-based architectures [10, 11, 38, 40,
                            51]and large-scale pretrained models [15, 21, 30] has significantly
                            advanced semantic segmentation in road scene perception. These
                            methods have shown outstanding performance in closed-set environments, characterized by predefined object categories like vehicles and pedestrians [12]. However, real-world environments are
                            inherently open, often containing unexpected out-of-distribution
                            (OOD) objects—from accident debris to rare natural phenomena—that
                            significantly challenge traditional segmentation models. Misclassifying these anomalies as background or assigning them to known
                            categories introduces substantial safety hazards, particularly in
                            autonomous driving. For instance, failing to detect a fallen tree
                            or a misplaced construction sign can result in severe accidents.
                            Consequently, reliably identifying and segmenting OOD objects
                            without compromising in-distribution (ID) segmentation accuracy
                            has emerged as a critical challenge, termed road anomaly detection,
                            for robust and safe scene understanding [4, 47].
                            Current road anomaly detection methods follow two main technical routes. The first paradigm relies primarily on reconstructionbased approaches [26, 27, 45, 46, 48], which use generative models
                            to learn normal scene distributions and detect anomalies via high
                            reconstruction errors. These methods proved effective in controlled
                            environments but faced inherent challenges in complex real-world
                            scenarios. Recently, the field has shifted towards score-based methods [13, 14, 28, 35, 37, 42], framing anomaly detection as a confidence estimation problem. These approaches typically leverage
                            prediction uncertainty or feature-space distances to compute pixellevel anomaly scores. Representative works include RPL [28]and
                            RbA [37]. RPL [28] uses residual pattern learning to identify deviations from expected feature distributions, while RbA [35] employs a
                            multi-head rejection mechanism, treating mask classification results
                            as multiple one-vs-all classifiers for robust anomaly identification.
                            Despite significant advancements, prior approaches often treat
                            road anomaly detection as a general anomaly detection problem,
                            where all outliers are deemed anomalous. This leads to two fundamental limitations hindering their practical deployment in autonomous driving systems, as shown in Fig. 1. First, they struggle
                            with incomplete segmentation of partial anomalies. Current methods fail to capture all OOD-specific pixels, especially when anomalies share partial visual characteristics with in-distribution objects.
                            For example, state-of-the-art RbA [35] (Fig. 1, column 1–2) detects
                            anomalies like damaged tires but fails to segment them entirely, leaving residual regions misclassified as background. This issue arises
                            because its multi-head rejection mechanism suppresses uncertain
                            pixels without explicitly modeling OOD shape continuity, leading
                            to fragmented masks with undetected anomaly parts. Second, they
                            suffer from task-agnostic overdetection. Existing methods overlook
                            road scene constraints, leading to false alarms in irrelevant regions. As shown in Fig. 1 (column 3–5), non-critical objects outside
                            drivable areas, such as roadside vegetation and distant buildings,
                            are erroneously flagged as OOD. This happens because RbA’s[35]
                            threshold-based scoring system prioritizes anomaly likelihood without considering autonomous driving task requirements, resulting
                            in overdetection in areas unrelated to navigation safety. These two
                            issues—partial under-segmentation and task-irrelevant overdetection—highlight the need for anomaly detection frameworks that
                            explicitly model OOD objective attributes while incorporating scene
                            constraints to enhance autonomous driving reliability.
                            In this paper, we propose Segmenting Objectiveness and Taskawareness (SOTA), a unified framework integrating semantic feature fusion with scene-understanding guided prompt learning. It
                            comprises two modules: the Semantic Fusion Block (SFB) and the
                            Scene-understanding Guided Prompt-Context Adapter (SG-PCA).
                            For SFB, the pixel-wise segmentor’s vanilla OOD prediction is
                            aligned to the latent vision space via projection and alignment
                            blocks. Fusing the OOD map with vision features improves objectiveness segmentation precision by refining OOD-specific attribute detection and reducing partial anomaly errors. SG-PCA, conversely, extracts road scene priors (e.g., lane topology, drivable areas) through task-aware aggregation, adapting Erosion and Dilation
                            operations to resolve partial occlusion issues. Subsequently, scene
                            priors and the vanilla OOD prediction combine to generate taskaware prompts via multi-aware cross-attention. By suppressing
                            navigation-irrelevant anomalies through prompt learning, SG-PCA
                            ensures focus on safety-critical regions, overcoming environmental constraints inherent in distribution-based methods. Moreover,
                            we introduce parameter-efficient adaptation of SAM’s [21] mask
                            decoder using Low-Rank Adaptation (LoRA) [19], enabling seamless integration of enriched OOD embeddings without additional
                            manual threshold tuning or postprocessing. Our contributions are
                            summarized as follows:
                            • We are the first to incorporate both objectiveness and taskawareness into road anomaly detection, significantly improving its practical utility for real-world autonomous driving
                            applications.
                            • We propose Segmenting Objectiveness and Task-awareness
                            (SOTA), comprising two key modules—Semantic Fusion Block
                            (SFB) and Scene-understanding Guided Prompt-Context Adapter
                            (SG-PCA)—that together achieve spatial and semantic completeness in anomaly detection.
                            • Extensive experiments on benchmark datasets demonstrate
                            that our approach significantly outperforms the baseline and
                            state-of-the-art method in both pixel-level and componentlevel anomaly segmentation results, achieving marked improvements across key evaluation metrics."""
    },

    {
        "method_name" : "Maskomaly",
        "mean_F1" : 49.9,
        "arxiv_link" : "https://arxiv.org/abs/2305.16972",
        "title" : """Maskomaly:Zero-Shot Mask Anomaly Segmentation""",
        "abstract" : """We present a simple and practical framework for anomaly segmentation called Maskomaly.
                        It builds upon mask-based standard semantic segmentation networks by adding a simple
                        inference-time post-processing step which leverages the raw mask outputs of such networks.
                        Maskomaly does not require additional training and only adds a small computational overhead
                        to inference. Most importantly, it does not require anomalous data at training. We show top
                        results for our method on SMIYC, RoadAnomaly, and StreetHazards. On the most central
                        benchmark, SMIYC1
                        , Maskomaly outperforms all directly comparable approaches. Further,
                        we introduce a novel metric that benefits the development of robust anomaly segmentation
                        methods and demonstrate its informativeness on RoadAnomaly.""",
        "introduction" : """Anomaly detection is the task of identifying whether one data point belongs to a set of inlier
                            classes that have been seen during the training. Recently, Fan et al. [18] have demonstrated
                            how difficult anomaly detection is from a theoretical viewpoint. Nevertheless, it is an essential
                            component of many real-world systems operating in safety-critical settings, such as autonomous
                            cars. In order to achieve fully automated driving, autonomous cars need to understand when an
                            anomaly is present and where the anomaly is located in the scene. The latter task is significantly
                            more challenging, as a model needs not only to output one single anomaly score per scene but
                            a dense map of pixel-level scores. With more diverse datasets [4, 7, 32, 39] and more complex
                            training paradigms [1, 8, 38, 39], current approaches have already achieved high scores on relevant
                            benchmarks. However, the current state of the art has not yet reached a level of accuracy that would
                            allow deployment in real-world settings. Given these facts, one might expect that an even more
                            complex training pipeline is needed to achieve results of practical utility.
                            Nonetheless, we propose a very simple yet effective zero-shot method which requires no training on anomalous data. Our method, Maskomaly, builds upon mask-based semantic segmentation
                            networks, such as Mask2Former [12], by post-processing their raw mask predictions to compute
                            a dense anomaly heatmap, only adding a small computational overhead at inference. To the best
                            of our knowledge, we are the first, along with the concurrent work in [26, 43], to explore the utility
                            of mask-based semantic segmentation networks for anomaly segmentation. Our key insight is that mask-based networks trained for standard semantic segmentation already learn to assign certain
                            masks to anomalies. Even though such masks are discarded by default when generating semantic
                            predictions, we show that they can be leveraged for inference on images potentially containing
                            anomalies to achieve state-of-the-art results in anomaly segmentation. Although Maskomaly is an
                            intuitive extension of Mask2Former, we show that computing the anomaly heatmap from the mask
                            outputs of the latter is nontrivial and justify our proposed components through proper ablations.
                            On multiple anomaly segmentation benchmarks, Maskomaly beats all state-of-the-art methods
                            that do not train with auxiliary data and most methods that perform such training. We evidence
                            the generality of our key insight and method across different backbones. Finally, we present a new
                            metric that promotes robust anomaly prediction methods tailored for real-world settings."""
    },

    {
        "method_name" : "ObsNet",
        "mean_F1" : 45.08,
        "arxiv_link" : "https://arxiv.org/abs/2108.01634",
        "title" : """Triggering Failures: Out-Of-Distribution detection by learning from local adversarial attacks in Semantic Segmentation""",
        "abstract" : """In this paper, we tackle the detection of out-of-distribution
                        (OOD) objects in semantic segmentation. By analyzing the
                        literature, we found that current methods are either accurate or fast but not both which limits their usability in real
                        world applications. To get the best of both aspects, we propose to mitigate the common shortcomings by following four
                        design principles: decoupling the OOD detection from the
                        segmentation task, observing the entire segmentation network instead of just its output, generating training data for
                        the OOD detector by leveraging blind spots in the segmentation network and focusing the generated data on localized
                        regions in the image to simulate OOD objects. Our main contribution is a new OOD detection architecture called ObsNet
                        associated with a dedicated training scheme based on Local
                        Adversarial Attacks (LAA). We validate the soundness of our
                        approach across numerous ablation studies. We also show it
                        obtains top performances both in speed and accuracy when
                        compared to ten recent methods of the literature on three
                        different datasets.""",
        "introduction" : """For real-world decision systems such as autonomous vehicles, accuracy is not the only performance requirement
                            and it often comes second to reliability, robustness, and
                            safety concerns [40], as any failure carries serious consequences. Component modules of such systems frequently
                            rely on Deep Neural Networks (DNNs) which have emerged
                            as a dominating approach across numerous tasks and benchmarks [59, 21, 20]. Yet, a major source of concern is related
                            to the data-driven nature of DNNs as they do not always
                            generalize to objects unseen in the training data. Simple
                            uncertainty estimation techniques, e.g., entropy of softmax
                            predictions [11], are less effective since modern DNNs are
                            consistently overconfident on both in-domain [19] and outof-distribution (OOD) data samples [46, 25, 23]. This hinders further the performance of downstream components
                            relying on their predictions. Dealing successfully with the
                            “unknown unknown”, e.g., by launching an alert or failing
                            gracefully, is crucial.
                            In this work we address OOD detection for semantic
                            segmentation, an essential and common task for visual
                            perception in autonomous vehicles. We consider “Out-ofdistribution”, pixels from a region that has no training labels associated with. This encompasses unseen objects, but
                            also noise or image alterations. The most effective methods
                            for OOD detection task stem from two major categories of
                            approaches: ensembles and auxiliary error prediction modules. DeepEnsemble (DE) [30] is a prominent and simple
                            ensemble method that exposes potentially unreliable predictions by measuring the disagreement between individual
                            DNNs. In spite of the outstanding performance, DE is computationally demanding for both training and testing and
                            prohibitive for real-time on-vehicle usage. For the latter category, given a trained main task network, a simple model is
                            trained in a second stage to detect its errors or estimate its
                            confidence [10, 22, 4]. Such approaches are computationally
                            lighter, yet, in the context of DNNs, an unexpected drawback is related to the lack of sufficient negative samples, i.e.,
                            failures, to properly train the error detector [10]. This is due
                            to an accumulation of causes: reduced size of the training
                            set for this module (essentially a mini validation set to withhold a sufficient amount for training the main predictor), few
                            mistakes made by the main DNNs, hence few negatives.
                            In this work, we propose to revisit the two-stage approach
                            with modern deep learning tools in a semantic segmentation
                            context. Given the application context, i.e., limited hardware
                            and high performance requirements, we aim for reliable
                            OOD detection (see Figure 1) without compromising on
                            predictive accuracy and computational time. To that end
                            we introduce four design principles aimed at mitigating the
                            most common pitfalls and covering two main aspects, (i)
                            architecture and (ii) training:
                            (i.a) The pitfall of trading accuracy in the downstream
                            segmentation task for robustness to OOD can be alleviated
                            by decoupling OOD detection from segmentation.
                            (i.b) Since the processing performed by the segmentation
                            network aims to recognize known objects and is not adapted
                            to OOD objects, the accuracy of the OOD detection can be
                            improved significantly by observing the entire segmentation
                            network instead of just its output.
                            (ii.a) Training an OOD detector requires additional
                            data that can be generated by leveraging blind spots in the
                            segmentation network.
                            (ii.b) Generated data should focus on localized regions in
                            the image to mimic unknown objects that are OOD.
                            Following these principles, we propose a new OOD detection architecture called ObsNet and its associated training
                            scheme based on Local Adversarial Attacks (LAA). We experimentally show that our ObsNet+LAA method achieves
                            top performance in OOD detection on three semantic segmentation datasets (CamVid [9], StreetHazards [24] and
                            BDD-Anomaly [24]), compared to a large set of methods1
                            .
                            Contributions. To summarize, our contributions are as
                            follows: We propose a new OOD detection method for semantic segmentation based on four design principles: (i.a)
                            decoupling OOD detection from the segmentation task; (i.b)
                            observing the full segmentation network instead of just the
                            output; (ii.a) generating training data for the OOD detector
                            using blind spots of the segmentation network; (ii.b) focusing the adversarial attacks in localized region of the image
                            to simulate unknown objects. We implement these four
                            principles in a new architecture called ObsNet and its associated training scheme using Local Adversarial Attacks
                            (LAA). We perform extensive ablation studies on these
                            principles to validate them empirically. We compare our
                            method to 10 diverse methods from the literature on three
                            datasets (CamVid OOD, StreetHazards, BDD Anomaly)
                            and we show it obtains top performances both in accuracy and in speed.
                            Strength and weakness. The strengths and weaknesses of
                            our approach are:
                            ✓ It can be used with any pre-trained segmentation network without altering their performances and without
                            fine-tuning them (we train only the auxiliary module).
                            ✓ It is fast since only one extra forward pass is required.
                            ✓ It is very effective since we show it performs best compared to 10 very diverse methods from the literature on
                            three different datasets.
                            ✗ The pre-trained segmentation network has to allow for
                            adversarial attacks, which is the case of commonly used
                            deep neural networks.
                            ✗ Our observer network has a memory/computation overhead equivalent to that of the segmentation network,
                            which is not ideal for real time applications, but far less
                            than that of MC Dropout or deep ensemble methods.
                            In the next section, we position our work with respect to
                            the existing literature."""
    },

    {
        "method_name" : "RbA",
        "mean_F1" : 42.04,
        "arxiv_link" : "https://arxiv.org/abs/2211.14293",
        "title" : """RbA: Segmenting Unknown Regions Rejected by All""",
        "abstract" : """Standard semantic segmentation models owe their success
                        to curated datasets with a fixed set of semantic categories,
                        without contemplating the possibility of identifying unknown
                        objects from novel categories. Existing methods in outlier
                        detection suffer from a lack of smoothness and objectness
                        in their predictions, due to limitations of the per-pixel classification paradigm. Furthermore, additional training for
                        detecting outliers harms the performance of known classes.
                        In this paper, we explore another paradigm with region-level
                        classification to better segment unknown objects. We show
                        that the object queries in mask classification tend to behave
                        like one vs. all classifiers. Based on this finding, we propose
                        a novel outlier scoring function called RbA by defining the
                        event of being an outlier as being rejected by all known
                        classes. Our extensive experiments show that mask classification improves the performance of the existing outlier
                        detection methods, and the best results are achieved with
                        the proposed RbA. We also propose an objective to optimize
                        RbA using minimal outlier supervision. Further fine-tuning
                        with outliers improves the unknown performance, and unlike
                        previous methods, it does not degrade the inlier performance.""",
        "introduction" : """We address the problem of semantic segmentation of unknown categories. Detecting novel objects, for example, in
                            front of a self-driving vehicle, is crucial for safety yet very
                            challenging. The distribution of potential objects on the road
                            has a long tail of unknowns such as wild animals, vehicle
                            debris, litter, etc., manifesting in small quantities on the existing datasets [73, 6, 17]. The diversity of unknowns in terms
                            of appearance, size, and location adds to the difficulty. In
                            addition to the challenges of data, deep learning has evolved
                            around the closed-set assumption. Most existing models for
                            category prediction owe their success to curated datasets
                            with a fixed set of semantic categories. These models fail in
                            the open-set case by over-confidently assigning the labels of
                            known classes to unknowns [33, 58]. The existing approaches to segmenting unknowns can be
                            divided into two depending on whether they use supervision for unknown objects or not. In either case, the model
                            has access to known classes during training, i.e. inlier or
                            in-distribution, and the goal is to identify the pixels belonging to an unknown class, i.e. anomalous, outlier, or out-ofdistribution (OoD). Earlier approaches resort to an ensemble
                            of models [40] or Monte Carlo dropout [22] which require
                            multiple forward passes, therefore costly in practice. More
                            recent approaches use the maximum class probability [35]
                            predicted by the model as a measure of its confidence. However, this approach requires the probability predictions to
                            be calibrated, which is not guaranteed [64, 58, 26, 54, 38].
                            In the supervised case, the model can utilize outlier data to
                            learn a discriminative representation, however, outlier data is
                            limited. Typically, another dataset from a different domain
                            is used for this purpose [11], or outlier objects are artificially
                            added to driving images [25, 65].
                            The existing methods in outlier detection suffer from a
                            lack of smoothness and objectness in the OoD predictions
                            as shown in Fig. 1. This is mainly due to the limitations of the per-pixel classification paradigm that previous OoD
                            methods are built on. In this paper, we explore another
                            paradigm with region-level classification to better segment
                            objects. To that end, we use mask-classification models,
                            such as Mask2Former [14] that are trained to predict regions
                            and then classify each region rather than individual pixels.
                            This endows our method with spatial smoothness, learned
                            by region-level supervision. We discover the properties of
                            this family of models which allow better calibration of confidence values. Then, we exploit these properties to boost the
                            performance of the existing OoD methods that rely on predicted class scores such as max logit [34] and energy-based
                            ones [25, 65, 49].
                            The existing methods also suffer from high false positive
                            rates due to failing to separate the sources of uncertainty, especially on datasets in the wild such as Road Anomaly [48].
                            For example, on the boundaries, segmentation models typically predict weak scores for the two inlier classes separated
                            by the boundary, causing these regions to be confused as
                            OoD by score-based methods [34]. Based on exploring the
                            behavior of object queries in mask classification, we find
                            that most of the object queries tend to behave like one vs.
                            all classifiers. Consequently, we propose a novel outlier
                            scoring function based on this one vs. all behavior of object
                            queries. We define the event of a pixel being an outlier as being rejected by all known classes. In other words, we define
                            being an outlier as a complementary event whose probability
                            can be expressed in terms of the known class probabilities.
                            We show that this scoring function can eliminate irrelevant
                            sources of uncertainty as in the case of boundaries, resulting
                            in a considerably lower false positive rate on all datasets.
                            The state-of-the-art methods in OoD [25, 65] utilize outlier data for supervision. While better unknown segmentation can be achieved, it comes at the expense of lower
                            closed-set performance. Unfortunately, this unintended consequence is not desirable since the primary objective of
                            unknown segmentation is to identify unknowns while still
                            accurately recognizing known classes without compromising
                            the inlier performance.
                            We propose an objective to optimize the proposed outlier scoring function using a limited amount of outlier data.
                            By fine-tuning a very small portion of the model with this
                            objective, our method outperforms the state-of-the-art on
                            challenging datasets with high distribution shifts such as
                            Road Anomaly [48] and SMIYC [10]. Notably, we achieve
                            this without affecting the closed-set performance. Our contributions can be summarized as follows:
                            • We postulate and study the inherent ability of mask
                            classification models to express uncertainty, and use this
                            strength to boost the performance of several existing
                            OoD segmentation methods.
                            • Based on our finding that object queries behave approximately as one vs. all classifiers, we propose a novel
                            outlier scoring function that represents the probability of being an outlier as not being any of the known
                            classes. The proposed scoring function helps to eliminate uncertainty in ambiguous inlier regions such as
                            semantic boundaries.
                            • We propose a loss function that directly optimizes our
                            proposed scoring function using minimal outlier data.
                            The proposed objective exceeds the state-of-the-art by
                            only fine-tuning a very small portion of the model without affecting the closed-set performance."""
    },

    {
        "method_name" : "DenseHybrid",
        "mean_F1" : 31.08,
        "arxiv_link" : "https://arxiv.org/abs/2207.02606",
        "title" : """DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition""",
        "abstract" : """Anomaly detection can be conceived either through generative modelling of regular training data or by discriminating with respect to negative training data. These two approaches exhibit different
                        failure modes. Consequently, hybrid algorithms present an attractive research goal. Unfortunately, dense anomaly detection requires translational equivariance and very large input resolutions. These requirements
                        disqualify all previous hybrid approaches to the best of our knowledge.
                        We therefore design a novel hybrid algorithm based on reinterpreting
                        discriminative logits as a logarithm of the unnormalized joint distribution ˆp(x, y). Our model builds on a shared convolutional representation from which we recover three dense predictions: i) the closedset class posterior P(y|x), ii) the dataset posterior P(din|x), iii) unnormalized data likelihood ˆp(x). The latter two predictions are trained
                        both on the standard training data and on a generic negative dataset.
                        We blend these two predictions into a hybrid anomaly score which allows dense open-set recognition on large natural images. We carefully
                        design a custom loss for the data likelihood in order to avoid backpropagation through the untractable normalizing constant Z(θ). Experiments evaluate our contributions on standard dense anomaly detection benchmarks as well as in terms of open-mIoU - a novel metric for dense open-set performance. Our submissions achieve state-ofthe-art performance despite neglectable computational overhead over
                        the standard semantic segmentation baseline. Official implementation:
                        https://github.com/matejgrcic/DenseHybrid""",
        "introduction" : """High accuracy, fast inference and small memory footprint of modern neural networks steadily expand the horizon of downstream applications. Many exciting
                            applications require advanced image understanding functionality provided by semantic segmentation [17]. These models associate each pixel with a class from
                            a predefined taxonomy. They can accurately segment two megapixel images in real-time on low-power embedded hardware [11,43,26]. However, the standard
                            training procedures assume the closed-world setup which may raise serious safety
                            issues in real-world deployments. For example, if a segmentation model missclassifies an unknown object (e.g. lost cargo) as road, the autonomous car may
                            experience a serious accident. Such hazards can be alleviated by complementing semantic segmentation with dense anomaly detection. The resulting dense
                            open-set recognition models are more suitable for real-world applications due to
                            ability to decline the decision in anomalous pixels.
                            Previous approaches for dense anomaly detection either use a generative or
                            a discriminative perspective. Generative approaches are based on density estimation [6] or image resynthesis [36,4]. Discriminative approaches use classification confidence [23], a binary classifier [2] or Bayesian inference [29]. These two
                            perspectives exhibit different failure modes. Generative detectors inaccurately
                            disperse the probability volume [41,47,38,53] or rely on risky image resynthesis.
                            On the other hand, discriminative detectors assume training on full span of the
                            input space, even including unknown unknowns [25].
                            In this work we combine the two perspectives into a hybrid anomaly detector.
                            The proposed approach complements a standard semantic segmentation model
                            with two additional predictions: i) unnormalized dense data likelihood ˆp(x) [6],
                            and ii) dense data posterior P(din|x) [2]. Both predictions require training with
                            negative data [25,2,4,10]. Joining these two outputs yields an accurate yet efficient dense anomaly detector which we refer to as DenseHybrid.
                            We summarize our contributions as follows. We propose the first hybrid
                            anomaly detector which allows end-to-end training and operates at pixel level.
                            Our approach combines likelihood evaluation and discrimination with respect
                            to an off-the-shelf negative dataset. Our experiments reveal accurate anomaly
                            detection despite minimal computational overhead. We complement semantic
                            segmentation with DenseHybrid to achieve dense open-set recognition. We report state-of-the-art dense open-set recognition performance according to a novel
                            performance metric which we refer to as open-mIoU."""
    },

    {
        "method_name" : "Maximized Entropy",
        "mean_F1" : 28.72,
        "arxiv_link" : "https://arxiv.org/abs/2012.06575",
        "title" : """Entropy Maximization and Meta Classification for Out-Of-Distribution Detection in Semantic Segmentation""",
        "abstract" : """Deep neural networks (DNNs) for the semantic segmentation of images are usually trained to operate on a predefined closed set of object classes. This is in contrast to
                        the “open world” setting where DNNs are envisioned to be
                        deployed to. From a functional safety point of view, the ability to detect so-called “out-of-distribution” (OoD) samples,
                        i.e., objects outside of a DNN’s semantic space, is crucial
                        for many applications such as automated driving. A natural baseline approach to OoD detection is to threshold on
                        the pixel-wise softmax entropy. We present a two-step procedure that significantly improves that approach. Firstly,
                        we utilize samples from the COCO dataset as OoD proxy
                        and introduce a second training objective to maximize the
                        softmax entropy on these samples. Starting from pretrained
                        semantic segmentation networks we re-train a number of
                        DNNs on different in-distribution datasets and consistently
                        observe improved OoD detection performance when evaluating on completely disjoint OoD datasets. Secondly, we
                        perform a transparent post-processing step to discard false
                        positive OoD samples by so-called “meta classification.”
                        To this end, we apply linear models to a set of hand-crafted
                        metrics derived from the DNN’s softmax probabilities. In
                        our experiments we consistently observe a clear additional
                        gain in OoD detection performance, cutting down the number of detection errors by 52% when comparing the best
                        baseline with our results. We achieve this improvement sacrificing only marginally in original segmentation performance. Therefore, our method contributes to safer DNNs
                        with more reliable overall system performance.""",
        "introduction" : """In recent years spectacular advances in the computer
                            vision task semantic segmentation have been achieved by
                            deep learning [47, 51]. Deep convolutional neural networks
                            (CNNs) are envisioned to be deployed to real world applications, where they are likely to be exposed to data that is substantially different from the model’s training data. We
                            consider data samples that are not included in the set of a
                            model’s semantic space as out-of-distribution (OoD) samples. State-of-the-art neural networks for semantic segmentation, however, are trained to recognize a predefined closed
                            set of object classes [13, 32], e.g. for the usage in environment perception systems of autonomous vehicles [24]. In
                            open world settings there are countless possibly occurring
                            objects. Defining additional classes requires a large amount
                            of annotated data (cf. [12, 52]) and may even lead to performance drops [15]. One natural approach is to introduce a
                            none-of-the-known output for objects not belonging to any
                            of the predefined classes [49]. In other words, one uses a
                            set of object classes that is sufficient for most scenarios and
                            treats OoD objects by enforcing an alternative model output
                            for such samples. From a functional safety point of view, it
                            is a crucial but missing prerequisite that neural networks are capable of reliably indicating when they are operating out of
                            their proper domain, i.e., detecting OoD objects, in order to
                            initiate a fallback policy.
                            As images from everyday scenes usually contain many
                            different objects, of which only some could be out-ofdistribution, knowing the location where the OoD object
                            occurs is desired for practical application. Therefore, we
                            address the problem of detecting anomalous regions in an
                            image, which is the case if an OoD object is present (see
                            figure 1) and which is a research area of high interest
                            [6, 20, 33, 42]. This so-called anomaly segmentation [5, 20]
                            can be pursued, for instance, by incorporating sophisticated
                            uncertainty estimates [3, 18] or by adding an extra class to
                            the model’s learnable set of classes [49].
                            In this work, we detect OoD objects in semantic segmentation with a different approach which is composed of
                            two steps: As first step, we re-train the segmentation CNN
                            to predict class labels with low confidence scores on OoD
                            inputs, by enforcing the model to output high prediction uncertainty. In order to quantify uncertainty, we compute the
                            softmax entropy which is maximized when a model outputs
                            uniform probability scores over all classes [29]. By deliberately including annotated OoD objects as known unknowns
                            into the re-training process and employing a modified multiobjective loss function, we observe that the segmentation
                            CNN generalizes learned uncertainty to unseen OoD samples (unknown unknowns) without significantly sacrificing
                            in original performance on the primary task, see figure 1.
                            The initial model for semantic segmentation is trained on
                            the Cityscapes data [13]. As proxy for OoD samples we randomly pick images from the COCO dataset [32] excluding
                            the ones with instances that are also available in Cityscapes,
                            cf. [19, 22, 37] for a related approach in image classification. We evaluate the pixel-wise OoD detection performance via entropy thresholding for OoD samples from the
                            LostAndFound [42] and Fishyscapes [6] dataset, respectively. Both datasets share the same setup as Cityscapes but
                            include OoD objects.
                            The second step incorporates a meta classifier flagging
                            incorrect class predictions at segment level, similar as proposed in [34, 44, 45] for the detection of false positive instances in semantic segmentation. After increasing the sensitivity towards predicting OoD objects, we aim at removing
                            false predictions which are produced due to the preceding
                            entropy boost (cf. [9]). The removal of false positive OoD
                            object predictions is based on aggregated dispersion measures and geometry features within segments (connected
                            components of pixels), with all information derived solely
                            from the CNN’s softmax output. As meta classifier we employ a simple linear model which allows us to track and
                            understand the impact of each metric.
                            To sum up our contributions, we are the first to successfully modify the training of segmentation CNNs to make
                            them much more efficient at detecting OoD samples in LostAndFound and Fishyscapes. Re-training the CNNs with
                            a specific choice of OoD images from COCO [32] clearly
                            outperforms the natural baseline approach of plain softmax
                            entropy thresholding [21] as well as many state-of-the-art
                            approaches from image classification. In addition, we are
                            the first to demonstrate that entropy based OoD object predictions in semantic segmentation can be meta classified
                            reliably, i.e., classified whether one considered OoD prediction is true positive or false positive without having access to the ground truth. For this meta task we employ
                            simple logistic regression. Combining entropy maximization and meta classification therefore is an efficient and yet
                            lightweight method, which is particularly suitable as an integrated monitoring system of safety-critical real world applications based on deep learning."""
    },

    {
        "method_name" : "ATTA",
        "mean_F1" : 20.64,
        "arxiv_link" : "https://arxiv.org/abs/2309.05994",
        "title" : """ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation""",
        "abstract" : """Recent advancements in dense out-of-distribution (OOD) detection have primarily
                        focused on scenarios where the training and testing datasets share a similar domain,
                        with the assumption that no domain shift exists between them. However, in realworld situations, domain shift often exits and significantly affects the accuracy of
                        existing out-of-distribution (OOD) detection models. In this work, we propose a
                        dual-level OOD detection framework to handle domain shift and semantic shift
                        jointly. The first level distinguishes whether domain shift exists in the image by
                        leveraging global low-level features, while the second level identifies pixels with
                        semantic shift by utilizing dense high-level feature maps. In this way, we can
                        selectively adapt the model to unseen domains as well as enhance model’s capacity
                        in detecting novel classes. We validate the efficacy of our method on several OOD
                        segmentation benchmarks, including those with significant domain shifts and those
                        without, observing consistent performance improvements across various baseline
                        models. Code is available at https://github.com/gaozhitong/ATTA.""",
        "introduction" : """Semantic segmentation, a fundamental computer vision task, has witnessed remarkable progress
                            thanks to the expressive representations learned by deep neural networks [33]. Despite the advances,
                            most deep models are trained under a close-world assumption, and hence do not possess knowledge
                            of what they do not know, leading to over-confident and inaccurate predictions for the unknown
                            objects [18]. To address this, the task of dense out-of-distribution (OOD) detection [1, 15], which
                            aims to generate pixel-wise identification of the unknown objects, has attracted much attention as it
                            plays a vital role in a variety of safety-critical applications such as autonomous driving.
                            Recent efforts in dense OOD detection have primarily focused on the scenarios where training
                            and testing data share a similar domain, assuming no domain shift (or covariant shift) between
                            them [50, 3, 1, 15, 38]. However, domain shift widely exists in real-world situations [39] and can
                            also be observed in common dense OOD detection benchmarks [29]. In view of this, we investigate
                            the performance of existing dense OOD detection methods under the test setting with domain-shift
                            and observe significant performance degradation in comparison with the setting without domain-shift
                            (cf. Figure 1). In particular, the state-of-the-art detection models typically fail to distinguish the
                            distribution shift in domain and the distribution shift in semantics, and thus tend to predict high
                            uncertainty scores for inlier-class pixels.
                            A promising strategy to tackle such domain shift is to adapt a model during test (known as test-time
                            adaptation (TTA) [46]), which utilizes unlabeled test data to finetune the model without requiring
                            prior information of test domain. However, applying the existing test-time domain adaption (TTA)
                            techniques [46, 7, 31, 47] to the task of general dense OOD detection faces two critical challenges.
                            First, traditional TTA methods often assume the scenarios where all test data are under domain shift while our dense OOD detection task addresses a more realistic setting where test data can come
                            from seen or unseen domains without prior knowledge. In such a scenario, TTA techniques like
                            the transductive batch normalization (TBN) [36, 40, 2], which substitutes training batch statistics
                            with those of the test batch, could inadvertently impair OOD detection performance on images from
                            seen domains due to inaccurate normalization parameter estimation (cf. Figure 1(b)). On the other
                            hand, the existence of novel classes in test images further complicates the problem. Unsupervised
                            TTA losses like entropy minimization [46, 12, 32, 45] often indiscriminately reduce the uncertainty
                            or OOD scores of these novel classes, leading to poor OOD detection accuracy (cf. Figure 1(b)).
                            Consequently, how to design an effective test-time adaptation strategy for the general dense OOD
                            detection in wild remains an open problem.
                            In this work, we aim to address the aforementioned limitations and tackle the problem of dense OOD
                            detection in wild with both domain- and semantic-level distribution shift. To this end, we propose a
                            novel dual-level test-time adaptation framework that simultaneously detects two types of distribution
                            shift and performs online model adaptation in a selective manner. Our core idea is to leverage
                            low-level feature statistics of input image to detect whether domain-level shift exists while utilizing
                            dense semantic representations to identify pixels with semantic-level shift. Based on this dual-level
                            distribution-shift estimation, we design an anomaly-aware self-training procedure to compensate for
                            the potential image-level domain shift and to enhance its novel-class detection capacity based on
                            re-balanced uncertainty minimization of model predictions. Such a selective test-time adaptation
                            strategy allows us to adapt an open-set semantic segmentation model to a new environment with
                            complex distribution shifts.
                            Specifically, we develop a cascaded modular TTA framework for any pretrained segmentation model
                            with OOD detection head. Our framework consists of two main stages, namely a selective Batch
                            Normalization (BN) stage and an anomaly-aware self-training stage. Given a test image (or batch),
                            we first estimate the probability of domain-shift based on the statistics of the model’s BN activations
                            and update the normalization parameters accordingly to incorporate new domain information. Subsequently, our second stage performs an online self-training for the entire segmentation model based
                            on an anomaly-aware entropy loss, which jointly minimizes a re-balanced uncertainty of inlier-class
                            prediction and outlier detection. As the outlier-class labels are unknown, we design a mixture model
                            in the OOD score space to generate the pseudo-labels of pixels for the entropy loss estimation.
                            We validate the efficacy of our proposed method on several OOD segmentation benchmarks, including
                            those with significant domain shifts and those without, based on FS Static [1], FS Lost&Found [1],
                            RoadAnomaly [29] and SMIYC [3]. The results show that our method consistently improves the
                            performance of dense OOD detection across various baseline models especially on the severe domain
                            shift settings, and achieves new state-of-the-arts performance on the benchmarks.
                            To summarize, our main contribution is three-folds: (i) We propose the problem of dense OOD
                            detection under domain shift (or covariance shift), revealing the limitations of existing dense OOD
                            detection methods in wild. (ii) We introduce an anomaly-aware test-time adaptation method that
                            jointly tackles domain and semantic shifts. (iii) Our extensive experiments validate our approach,
                            demonstrating significant performance gains on various OOD segmentation benchmarks, especially
                            those with notable domain shifts."""
    },

    {
        "method_name" : "NFlowJS",
        "mean_F1" : 14.89,
        "arxiv_link" : "https://arxiv.org/abs/2112.12833",
        "title" : """Dense Out-of-Distribution Detection by Robust Learning on Synthetic Negative Data""",
        "abstract" : """Standard machine learning is unable to accommodate inputs which do not belong to the training distribution. The
                        resulting models often give rise to confident incorrect predictions
                        which may lead to devastating consequences. This problem is
                        especially demanding in the context of dense prediction since
                        input images may be only partially anomalous. Previous work has
                        addressed dense out-of-distribution detection by discriminative
                        training with respect to off-the-shelf negative datasets. However,
                        real negative data are unlikely to cover all modes of the
                        entire visual world. To this end, we extend this approach by
                        generating synthetic negative patches along the border of the
                        inlier manifold. We leverage a jointly trained normalizing flow
                        due to coverage-oriented learning objective and the capability to
                        generate samples at different resolutions. We detect anomalies
                        according to a principled information-theoretic criterion which
                        can be consistently applied through training and inference. The
                        resulting models set the new state of the art on benchmarks for
                        out-of-distribution detection in road-driving scenes and remote
                        sensing imagery, in spite of minimal computational overhead.""",
        "introduction" : """Image understanding involves recognizing objects and
                            localizing them down to the pixel level [1]. In its basic
                            form, the task is to classify each pixel into one of K predefined
                            classes, which is also known as semantic segmentation [2].
                            Recent work improves perception quality through instance
                            recognition [3], depth reconstruction [4], semantic forecasting
                            [5], and competence in the open world [6].
                            Modern semantic segmentation approaches [2] are based
                            on deep learning. A deep model for semantic segmentation
                            maps the input RGB image x3×H×W into the corresponding
                            prediction yK×H×W . Typically, the model parameters θ are
                            obtained by gradient optimization of a supervised discriminative objective based on maximum likelihood. Recent approaches produce high-fidelity segmentations of large images
                            in real time even when inferring on a modest GPU [7].
                            However, standard learning is susceptible to overconfidence in
                            incorrect predictions [8], which may make the model unusable
                            in presence of semantic outliers [9] and domain shift [10]. This
                            poses a threat to models deployed in the real world [11], [12].
                            We study ability of deep models for natural image understanding to deal with OOD input. We desire to correctly
                            segment the scene while simultaneously detecting anomalous
                            objects which are unlike any scenery from the training dataset Such capability is important in real-world applications
                            like road driving [14], [15] and remote sensing [16], [17].
                            Previous approaches to dense OOD detection rely on
                            Bayesian modeling [18], image resynthesis [14], [19], [20],
                            recognition in the latent space [12], or auxiliary negative
                            training data [21]. However, all these approaches have significant shortcomings. Bayesian approaches and image resynthesis
                            require extraordinary computational resources that hamper
                            development and makes them unsuitable for real-time applications. Recognition in the latent space [12] may be sensitive
                            to feature collapse [22], [23] due to relying on pre-trained
                            features. Training on auxiliary negative data may give rise to
                            undesirable bias and over-optimistic evaluation. Moreover, appropriate negative data may be unavailable in some application
                            areas such as medical diagnostics [24] or remote sensing [16],
                            [25]. Our experiments suggest that synthetic negatives may
                            come to aid in such cases.
                            This work addresses dense out-of-distribution detection by
                            encouraging the chosen standard dense prediction model to
                            emit uniform predictions in outliers [26]. We propose to
                            perform the training on mixed-content images [21] which we
                            craft by pasting synthetic negatives into inlier training images.
                            We learn to generate synthetic negatives by jointly optimizing
                            high inlier likelihood, and uniform discriminative prediction
                            [26]. We argue that normalizing flows are better than GANs
                            for the task at hand due to much better distribution coverage
                            and more stable training. Additionally, normalizing flows can
                            generate samples of variable spatial dimensions [27] which
                            makes them suitable for mimicking anomalies of varying size.
                            This paper proposes five major improvements over our
                            preliminary report [28]. First, we show that Jensen-Shannon
                            divergence is a criterion of choice for robust joint learning
                            in presence of noisy synthetic negatives. We use the same
                            criterion during inference, as a score for OOD detection. Second, we propose to discourage overfitting the discriminative
                            model to synthetic outliers through separate pre-training of
                            the discriminative model and the generative flow. Third, we
                            offer theoretical evidence for the advantage of our coverageoriented synthetic negatives with respect to their adversarial
                            counterparts. Fourth, we demonstrate utility of synthetic outliers by performing experiments within the domain of remote
                            sensing. These experiments show that of-the-shelf negative
                            datasets such as ImageNet, COCO or Ade20k do not represent
                            a suitable source of negative content for all possible domains.
                            Fifth, we show that training with synthetic negatives increases
                            the separation between knowns and unknowns in the logit
                            space, which makes our method a prominent component of
                            future dense open-set recognition systems. We refer to the consolidated method as NFlowJS. NFlowJS achieves state-ofthe-art performance on benchmarks for dense OOD detection
                            in road driving scenes [11], [12] and remote sensing images
                            [16], despite abstaining from auxiliary negative data [21],
                            image resynthesis [14], [19] and Bayesian modelling [18]. Our
                            method has a very low overhead over the standard discriminative model, making it suitable for real-time applications."""
    },

    {
        "method_name" : "Image Resynthesis",
        "mean_F1" : 12.51,
        "arxiv_link" : "https://arxiv.org/abs/1904.07595",
        "title" : """Detecting the Unexpected via Image Resynthesis""",
        "abstract" : """Classical semantic segmentation methods, including the
                        recent deep learning ones, assume that all classes observed
                        at test time have been seen during training. In this paper, we
                        tackle the more realistic scenario where unexpected objects
                        of unknown classes can appear at test time. The main trends
                        in this area either leverage the notion of prediction uncertainty to flag the regions with low confidence as unknown, or
                        rely on autoencoders and highlight poorly-decoded regions.
                        Having observed that, in both cases, the detected regions
                        typically do not correspond to unexpected objects, in this
                        paper, we introduce a drastically different strategy: It relies on the intuition that the network will produce spurious
                        labels in regions depicting unexpected objects. Therefore,
                        resynthesizing the image from the resulting semantic map
                        will yield significant appearance differences with respect to
                        the input image. In other words, we translate the problem
                        of detecting unknown classes to one of identifying poorlyresynthesized image regions. We show that this outperforms
                        both uncertainty- and autoencoder-based methods.""",
        "introduction" : """Semantic segmentation has progressed tremendously in
                            recent years and state-of-the-art methods rely on deep learning [4, 5, 47, 45]. Therefore, they typically operate under
                            the assumption that all classes encountered at test time have
                            been seen at training time. In reality, however, guaranteeing that all classes that can ever be found are represented
                            in the database is impossible when dealing with complex
                            outdoors scenes. For instance, in an autonomous driving
                            scenario, one should expect to occasionally find the unexpected, in the form of animals, snow heaps, or lost cargo
                            on the road, as shown in Fig. 1. Note that the corresponding labels are absent from standard segmentation training
                            datasets [7, 46, 14]. Nevertheless, a self-driving vehicle
                            should at least be able to detect that some image regions
                            cannot be labeled properly and warrant further attention.
                            Recent approaches to addressing this problem follow two trends. The first one involves reasoning about the prediction uncertainty of the deep networks used to perform
                            the segmentation [18, 24, 19, 12]. In the driving scenario,
                            we have observed that the uncertain regions tend not to coincide with unknown objects, and, as illustrated by Fig. 1,
                            these methods therefore fail to detect the unexpected. The
                            second trend consists of leveraging autoencoders to detect
                            anomalies [8, 33, 1], assuming that never-seen-before objects will be decoded poorly. We found, however, that autoencoders tend to learn to simply generate a lower-quality
                            version of the input image. As such, as shown in Fig. 1,
                            they also fail to find the unexpected objects.
                            In this paper, we therefore introduce a radically different approach to detecting the unexpected. Fig. 2 depicts our
                            pipeline, built on the following intuition: In regions containing unknown classes, the segmentation network will make
                            spurious predictions. Therefore, if one tries to resynthesize
                            the input image from the semantic label map, the resynthesized unknown regions will look significantly different from
                            the original ones. In other words, we reformulate the problem of segmenting unknown classes as one of identifying the differences between the original input image and the one
                            resynthesized from the predicted semantic map. To this end,
                            we leverage a generative network [42] to learn a mapping
                            from semantic maps back to images. We then introduce a
                            discrepancy network that, given as input the original image,
                            the resynthesized one, and the predicted semantic map, produces a binary mask indicating unexpected objects. To train
                            this network without ever observing unexpected objects, we
                            simulate such objects by changing the semantic label of
                            known object instances to other, randomly chosen classes.
                            This process, described in Section 3.2, does not require seeing the unknown classes during training, which makes our
                            approach applicable to detecting never-seen-before classes
                            at test time.
                            Our contribution is therefore a radically new approach
                            to identifying regions that have been misclassified by a
                            given semantic segmentation method, based on comparing
                            the original image with a resynthesized one. We demonstrate the ability of our approach to detect unexpected objects using the Lost and Found dataset [35]. This dataset,
                            however, only depicts a limited set of unexpected objects
                            in a fairly constrained scenario. To palliate this lack of
                            data, we create a new dataset depicting unexpected objects,
                            such as animals, rocks, lost tires and construction equipment, on roads. Our method outperforms uncertainty-based
                            baselines, as well as the state-of-the-art autoencoder-based
                            method specifically designed to detect road obstacles [8].
                            Furthermore, our approach to detecting anomalies by
                            comparing the original image with a resynthesized one is
                            generic and applies to other tasks than unexpected object
                            detection. For example, deep learning segmentation algorithms are vulnerable to adversarial attacks [44, 6], that is,
                            maliciously crafted images that look normal to a human but
                            cause the segmentation algorithm to fail catastrophically.
                            As in the unexpected object detection case, re-synthesizing
                            the image using the erroneous labels results in a synthetic
                            image that looks nothing like the original one. Then, a
                            simple non-differentiable detector, thus less prone to attacks, is sufficient to identify the attack. As shown by our
                            experiments, our approach outperforms the state-of-the-art
                            one of [43] for standard attacks"""
    },

    {
        "method_name" : "SynBoost",
        "mean_F1" : 9.99,
        "arxiv_link" : "https://arxiv.org/abs/2103.05445",
        "title" : """Pixel-wise Anomaly Detection in Complex Driving Scenes""",
        "abstract" : """The inability of state-of-the-art semantic segmentation
                        methods to detect anomaly instances hinders them from being deployed in safety-critical and complex applications,
                        such as autonomous driving. Recent approaches have
                        focused on either leveraging segmentation uncertainty to
                        identify anomalous areas or re-synthesizing the image from
                        the semantic label map to find dissimilarities with the input image. In this work, we demonstrate that these two
                        methodologies contain complementary information and can
                        be combined to produce robust predictions for anomaly
                        segmentation. We present a pixel-wise anomaly detection
                        framework that uses uncertainty maps to improve over existing re-synthesis methods in finding dissimilarities between
                        the input and generated images. Our approach works as
                        a general framework around already trained segmentation
                        networks, which ensures anomaly detection without compromising segmentation accuracy, while significantly outperforming all similar methods. Top-2 performance across
                        a range of different anomaly datasets shows the robustness
                        of our approach to handling different anomaly instances.""",
        "introduction" : """Recent advances in deep learning have shown significant improvements in the field of computer vision. Neural
                            networks have become the de-facto methodology for classification, object detection, and semantic segmentation due
                            to their high accuracy in comparison to previous methods
                            [35, 34, 41]. However, while the predictions of these networks are highly accurate, they usually fail when encountering anomalous inputs (i.e. instances outside the training
                            distribution of the network). With this work, we focus on the inability of existing semantic segmentation models to localize anomaly instances
                            and how this limitation hinders them from being deployed
                            in safety-critical, in-the-wild scenarios. Consider the case
                            of a self-driving vehicle that uses a semantic segmentation
                            model. If the agent encounters an anomalous object (i.e. a
                            wooden box in the middle of the street), the model could
                            wrongly classify this object as part of the road and lead the
                            vehicle to crash.
                            To detect such anomalies in the input, we build our
                            approach upon two established groups of methods. The
                            first group uses uncertainty estimation to detect anomalies. Their intuition follows that a low-confidence prediction is likely an anomaly. However, uncertainty estimation
                            methods themselves are still noisy and inaccurate. Previous works [24, 4] have shown that these models fail to detect many unexpected objects. Example failure cases are
                            shown in Figure 1 (top and bottom) where the anomaly object is either detected but miss-classified or non-detected
                            and blended with the background. In both cases, the segmentation network is overconfident about its prediction and,
                            thus, the estimated uncertainty (softmax entropy) is low.
                            The second group focuses on re-synthesizing the input
                            image from the predicted semantic map and then comparing
                            the two images (input and generated) to find the anomaly.
                            These models have shown promising results when dealing
                            with segmentation overconfidence but fail when the segmentation outputs a noisy prediction for the unknown object, as shown in Figure 1 (middle). This failure is explained
                            by the inability of the synthesis model to reconstruct noisy
                            patches of the semantic map, which complicates finding the
                            differences between input and synthesized images.
                            In this paper, we propose a novel pixel-level anomaly
                            framework that combines uncertainty and re-synthesis approaches in order to produce robust predictions for the different anomaly scenarios. Our experiments show that uncertainty and re-synthesis approaches are complementary to
                            each other, and together they cover the different outcomes
                            when a segmentation network encounters an anomaly.
                            Our framework builds upon previous re-synthesis methods [24, 12, 38] of reformulating the problem of segmenting unknown classes as one of identifying differences between the input image and the re-synthesised image from
                            a predicted semantic map. We improve over those frameworks by integrating different uncertainty measures, such
                            as softmax entropy [10, 21], softmax difference [31], and
                            perceptual differences [16, 8] to assist the dissimilarity network in differentiating the input and generated images. The
                            proposed framework successfully generalizes to all anomalies scenarios, as shown in Figure 1, with minimal additional computation effort and without the need to jeopardize
                            the segmentation network accuracy (no re-training necessary), which is one common flaw of other anomaly detectors [3, 26, 27]. Besides maintaining state-of-the-art performance in segmentation, eliminating the need for re-training
                            also reduces the complexity of adding an anomaly detector
                            to future segmentation networks, as training these networks
                            is non-trivial.
                            We evaluate our framework in public benchmarks for
                            anomaly detection, where we compare to methods similar
                            to ours that not compromise segmentation accuracy, as well
                            as those requiring full retraining. We also demonstrate that
                            our framework is able to generalize to different segmentation and synthesis networks, even when these models have
                            lower performance. We replace the segmentation and synthesis models with lighter architectures to prioritize speed
                            in time-critical scenarios like autonomous driving.
                            In summary, our contributions are the following:
                            – We present a novel pixel-wise anomaly detection
                            framework that leverages the best features of existing
                            uncertainty and re-synthesis methodologies.
                            – Our approach is robust to the different anomaly scenarios, achieving state-of-the-art performance on the
                            Fishyscapes benchmark while maintaining state-ofthe-art segmentation accuracy.
                            – Our proposed framework is able to generalize to different segmentation and synthesis networks, serving
                            as a wrapper methodology to existing segmentation
                            pipelines."""
    },

    {
        "method_name" : "Maximum Softmax",
        "mean_F1" : 5.37,
        "arxiv_link" : "https://arxiv.org/abs/1610.02136",
        "title" : """A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks""",
        "abstract" : """We consider the two related problems of detecting if an example is misclassified or
                        out-of-distribution. We present a simple baseline that utilizes probabilities from
                        softmax distributions. Correctly classified examples tend to have greater maximum softmax probabilities than erroneously classified and out-of-distribution examples, allowing for their detection. We assess performance by defining several tasks in computer vision, natural language processing, and automatic speech
                        recognition, showing the effectiveness of this baseline across all. We then show
                        the baseline can sometimes be surpassed, demonstrating the room for future research on these underexplored detection tasks.""",
        "introduction" : """When machine learning classifiers are employed in real-world tasks, they tend to fail when the
                            training and test distributions differ. Worse, these classifiers often fail silently by providing highconfidence predictions while being woefully incorrect (Goodfellow et al., 2015; Amodei et al.,
                            2016). Classifiers failing to indicate when they are likely mistaken can limit their adoption or
                            cause serious accidents. For example, a medical diagnosis model may consistently classify with
                            high confidence, even while it should flag difficult examples for human intervention. The resulting
                            unflagged, erroneous diagnoses could blockade future machine learning technologies in medicine.
                            More generally and importantly, estimating when a model is in error is of great concern to AI Safety
                            (Amodei et al., 2016).
                            These high-confidence predictions are frequently produced by softmaxes because softmax probabilities are computed with the fast-growing exponential function. Thus minor additions to the softmax
                            inputs, i.e. the logits, can lead to substantial changes in the output distribution. Since the softmax function is a smooth approximation of an indicator function, it is uncommon to see a uniform
                            distribution outputted for out-of-distribution examples. Indeed, random Gaussian noise fed into an
                            MNIST image classifier gives a “prediction confidence” or predicted class probability of 91%, as we
                            show later. Throughout our experiments we establish that the prediction probability from a softmax
                            distribution has a poor direct correspondence to confidence. This is consistent with a great deal of
                            anecdotal evidence from researchers (Nguyen & O’Connor, 2015; Yu et al., 2010; Provost et al.,
                            1998; Nguyen et al., 2015).
                            However, in this work we also show the prediction probability of incorrect and out-of-distribution
                            examples tends to be lower than the prediction probability for correct examples. Therefore, capturing prediction probability statistics about correct or in-sample examples is often sufficient for
                            detecting whether an example is in error or abnormal, even though the prediction probability viewed
                            in isolation can be misleading.
                            These prediction probabilities form our detection baseline, and we demonstrate its efficacy through
                            various computer vision, natural language processing, and automatic speech recognition tasks.
                            While these prediction probabilities create a consistently useful baseline, at times they are less effective, revealing room for improvement. To give ideas for future detection research, we contribute one method which outperforms the baseline on some (but not all) tasks. This new method evaluates
                            the quality of a neural network’s input reconstruction to determine if an example is abnormal.
                            In addition to the baseline methods, another contribution of this work is the designation of standard
                            tasks and evaluation metrics for assessing the automatic detection of errors and out-of-distribution
                            examples. We use a large number of well-studied tasks across three research areas, using standard
                            neural network architectures that perform well on them. For out-of-distribution detection, we provide ways to supply the out-of-distribution examples at test time like using images from different
                            datasets and realistically distorting inputs. We hope that other researchers will pursue these tasks in
                            future work and surpass the performance of our baselines.
                            In summary, while softmax classifier probabilities are not directly useful as confidence estimates,
                            estimating model confidence is not as bleak as previously believed. Simple statistics derived from
                            softmax distributions provide a surprisingly effective way to determine whether an example is misclassified or from a different distribution from the training data, as demonstrated by our experimental
                            results spanning computer vision, natural language processing, and speech recognition tasks. This
                            creates a strong baseline for detecting errors and out-of-distribution examples which we hope future
                            research surpasses."""
    },

    {
        "method_name" : "Mahalanobis",
        "mean_F1" : 2.68,
        "arxiv_link" : "https://arxiv.org/abs/1807.03888",
        "title" : """A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks""",
        "abstract" : """Detecting test samples drawn sufficiently far away from the training distribution
                        statistically or adversarially is a fundamental requirement for deploying a good
                        classifier in many real-world machine learning applications. However, deep neural networks with the softmax classifier are known to produce highly overconfident
                        posterior distributions even for such abnormal samples. In this paper, we propose
                        a simple yet effective method for detecting any abnormal samples, which is applicable to any pre-trained softmax neural classifier. We obtain the class conditional
                        Gaussian distributions with respect to (low- and upper-level) features of the deep
                        models under Gaussian discriminant analysis, which result in a confidence score
                        based on the Mahalanobis distance. While most prior methods have been evaluated for detecting either out-of-distribution or adversarial samples, but not both,
                        the proposed method achieves the state-of-the-art performances for both cases in
                        our experiments. Moreover, we found that our proposed method is more robust
                        in harsh cases, e.g., when the training dataset has noisy labels or small number of
                        samples. Finally, we show that the proposed method enjoys broader usage by applying it to class-incremental learning: whenever out-of-distribution samples are
                        detected, our classification rule can incorporate new classes well without further
                        training deep models.""",
        "introduction" : """Deep neural networks (DNNs) have achieved high accuracy on many classification tasks, e.g.,
                            speech recognition [1], object detection [9] and image classification [12]. However, measuring the
                            predictive uncertainty still remains a challenging problem [20, 21]. Obtaining well-calibrated predictive uncertainty is indispensable since it could be useful in many machine learning applications
                            (e.g., active learning [8] and novelty detection [18]) as well as when deploying DNNs in real-world
                            systems [2], e.g., self-driving cars and secure authentication system [6, 30].
                            The predictive uncertainty of DNNs is closely related to the problem of detecting abnormal samples that are drawn far away from in-distribution (i.e., distribution of training samples) statistically
                            or adversarially. For detecting out-of-distribution (OOD) samples, recent works have utilized the
                            confidence from the posterior distribution [13, 21]. For example, Hendrycks & Gimpel [13] proposed the maximum value of posterior distribution from the classifier as a baseline method, and it
                            is improved by processing the input and output of DNNs [21]. For detecting adversarial samples,
                            confidence scores were proposed based on density estimators to characterize them in feature spaces
                            of DNNs [7]. More recently, Ma et al. [22] proposed the local intrinsic dimensionality (LID) and
                            empirically showed that the characteristics of test samples can be estimated effectively using the LID. However, most prior works on this line typically do not evaluate both OOD and adversarial
                            samples. To best of our knowledge, no universal detector is known to work well on both tasks.
                            Contribution. In this paper, we propose a simple yet effective method, which is applicable to
                            any pre-trained softmax neural classifier (without re-training) for detecting abnormal test samples
                            including OOD and adversarial ones. Our high-level idea is to measure the probability density of test
                            sample on feature spaces of DNNs utilizing the concept of a “generative” (distance-based) classifier.
                            Specifically, we assume that pre-trained features can be fitted well by a class-conditional Gaussian
                            distribution since its posterior distribution can be shown to be equivalent to the softmax classifier
                            under Gaussian discriminant analysis (see Section 2.1 for our justification). Under this assumption,
                            we define the confidence score using the Mahalanobis distance with respect to the closest classconditional distribution, where its parameters are chosen as empirical class means and tied empirical
                            covariance of training samples. To the contrary of conventional beliefs, we found that using the
                            corresponding generative classifier does not sacrifice the softmax classification accuracy. Perhaps
                            surprisingly, its confidence score outperforms softmax-based ones very strongly across multiple
                            other tasks: detecting OOD samples, detecting adversarial samples and class-incremental learning.
                            We demonstrate the effectiveness of the proposed method using deep convolutional neural networks,
                            such as DenseNet [14] and ResNet [12] trained for image classification tasks on various datasets
                            including CIFAR [15], SVHN [28], ImageNet [5] and LSUN [32]. First, for the problem of detecting
                            OOD samples, the proposed method outperforms the current state-of-the-art method, ODIN [21], in
                            all tested cases. In particular, compared to ODIN, our method improves the true negative rate (TNR),
                            i.e., the fraction of detected OOD (e.g., LSUN) samples, from 45.6% to 90.9% on ResNet when
                            95% of in-distribution (e.g., CIFAR-100) samples are correctly detected. Next, for the problem
                            of detecting adversarial samples, e.g., generated by four attack methods such as FGSM [10], BIM
                            [16], DeepFool [26] and CW [3], our method outperforms the state-of-the-art detection measure,
                            LID [22]. In particular, compared to LID, ours improves the TNR of CW from 82.9% to 95.8% on
                            ResNet when 95% of normal CIFAR-10 samples are correctly detected.
                            We also found that our proposed method is more robust in the choice of its hyperparameters as well
                            as against extreme scenarios, e.g., when the training dataset has some noisy, random labels or a
                            small number of data samples. In particular, Liang et al. [21] tune the hyperparameters of ODIN
                            using validation sets of OOD samples, which is often impossible since the knowledge about OOD
                            samples is not accessible a priori. We show that hyperparameters of the proposed method can be
                            tuned only using in-distribution (training) samples, while maintaining its performance. We further
                            show that the proposed method tuned on a simple attack, i.e., FGSM, can be used to detect other
                            more complex attacks such as BIM, DeepFool and CW.
                            Finally, we apply our method to class-incremental learning [29]: new classes are added progressively
                            to a pre-trained classifier. Since the new class samples are drawn from an out-of-training distribution,
                            it is natural to expect that one can classify them using our proposed metric without re-training the
                            deep models. Motivated by this, we present a simple method which accommodates a new class at
                            any time by simply computing the class mean of the new class and updating the tied covariance of all
                            classes. We show that the proposed method outperforms other baseline methods, such as Euclidean
                            distance-based classifier and re-trained softmax classifier. This evidences that our approach have a
                            potential to apply to many other related machine learning tasks, such as active learning [8], ensemble
                            learning [19] and few-shot learning [31]."""
    }

]