

papers = [
    {
        "method_name" : "UniDepthV2",
        "SILog" : 7.74,
        "arxiv_link" : "https://arxiv.org/abs/2502.20110",
        "title" : """UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler""",
        "abstract" : """Accurate monocular metric depth estimation
                        (MMDE) is crucial to solving downstream tasks in 3D perception and modeling. However, the remarkable accuracy of
                        recent MMDE methods is confined to their training domains.
                        These methods fail to generalize to unseen domains even in the
                        presence of moderate domain gaps, which hinders their practical
                        applicability. We propose a new model, UniDepthV2, capable
                        of reconstructing metric 3D scenes from solely single images
                        across domains. Departing from the existing MMDE paradigm,
                        UniDepthV2 directly predicts metric 3D points from the input
                        image at inference time without any additional information, striving for a universal and flexible MMDE solution. In particular,
                        UniDepthV2 implements a self-promptable camera module predicting a dense camera representation to condition depth features.
                        Our model exploits a pseudo-spherical output representation,
                        which disentangles the camera and depth representations. In
                        addition, we propose a geometric invariance loss that promotes
                        the invariance of camera-prompted depth features. UniDepthV2
                        improves its predecessor UniDepth model via a new edge-guided
                        loss which enhances the localization and sharpness of edges
                        in the metric depth outputs, a revisited, simplified and more
                        efficient architectural design, and an additional uncertainty-level
                        output which enables downstream tasks requiring confidence.
                        Thorough evaluations on ten depth datasets in a zero-shot
                        regime consistently demonstrate the superior performance and
                        generalization of UniDepthV2. Code and models are available at:
                        github.com/lpiccinelli-eth/UniDepth.""",
        "introduction" : """Precise pixel-wise depth estimation is crucial to understanding the geometric scene structure, with applications
                            in 3D modeling [1], robotics [2], [3], and autonomous vehicles [4], [5]. However, delivering reliable metric scaled depth
                            outputs is necessary to perform 3D reconstruction effectively,
                            thus motivating the challenging and inherently ill-posed task
                            of Monocular Metric Depth Estimation (MMDE).
                            While existing MMDE methods [6]–[12] have demonstrated
                            remarkable accuracy across different benchmarks, they require
                            training and testing on datasets with similar camera intrinsics
                            and scene scales. Moreover, the training datasets typically have
                            a limited size and contain little diversity in scenes and cameras.
                            These characteristics result in poor generalization to realworld inference scenarios [13], where images are captured in uncontrolled, arbitrarily structured environments and cameras
                            with arbitrary intrinsics. What makes the situation even worse
                            is the imperfect nature of actual ground-truth depth which is
                            used to supervise MMDE models, namely its sparsity and its
                            incompleteness near edges, which results in blurry predictions
                            with inaccurate fine-grained geometric details.
                            Only a few methods [14]–[16] have addressed the challenging task of generalizable MMDE. However, these methods assume controlled setups at test time, including camera
                            intrinsics. While this assumption simplifies the task, it has
                            two notable drawbacks. Firstly, it does not address the full
                            application spectrum, e.g. in-the-wild video processing and
                            crowd-sourced image analysis. Secondly, the inherent camera
                            parameter noise is directly injected into the model, leading to
                            large inaccuracies in the high-noise case.
                            In this work, we address the more demanding task of generalizable MMDE without any reliance on additional external
                            information, such as camera parameters, thus defining the
                            universal MMDE task. Our approach, named UniDepthV2, extends UniDepth [17] and is the first that attempts to solve this
                            challenging task without restrictions on scene composition and
                            setup and distinguishes itself through its general and adaptable
                            nature. Unlike existing methods, UniDepthV2 delivers metric
                            3D predictions for any scene solely from a single image,
                            waiving the need for extra information about scene or camera.
                            Furthermore, UniDepthV2 flexibly allows for the incorporation
                            of additional camera information at test time. Simultaneously,
                            UniDepthV2 achieves sharper depth predictions with betterlocalized depth discontinuities than the original UniDepth
                            model thanks to a novel edge-guided loss that enhances the consistency of the local structure of depth predictions around
                            edges with the respective structure in the ground truth.
                            The design of UniDepthV2 introduces a camera module
                            that outputs a non-parametric, i.e. dense camera representation, serving as the prompt to the depth module. However,
                            relying only on this single additional module clearly results in
                            challenges related to training stability and scale ambiguity.
                            We propose an effective pseudo-spherical representation of
                            the output space to disentangle the camera and depth dimensions of this space. This representation employs azimuth
                            and elevation angle components for the camera and a radial
                            component for the depth, forming a perfect orthogonal space
                            between the camera plane and the depth axis. Moreover, the
                            pinhole-based camera representation is positionally encoded
                            via a sine encoding in UniDepthV2, leading to a substantially more efficient computation compared to the spherical
                            harmonic encoding of the pinhole-based representation of the
                            original UniDepth. Figure 1 depicts our camera self-prompting
                            mechanism and the output space. Additionally, we introduce a
                            geometric invariance loss to enhance the robustness of depth
                            estimation. The underlying idea is that the camera-conditioned
                            depth outputs from two views of the same image should
                            exhibit reciprocal consistency. In particular, we sample two
                            geometric augmentations, creating different views for each
                            training image, thus simulating different apparent cameras for
                            the original scene. Besides the aforementioned consistencyoriented invariance loss, UniDepthV2 features an additional
                            uncertainty output and respective loss. These pixel-level uncertainties are supervised with the differences between the
                            respective depth predictions and their corresponding groundtruth values, and enable the utilization of our MMDE model
                            in downstream tasks such as control which require confidenceaware perception inputs [18]–[21] for certifiability.
                            The overall contributions of the present, extended journal
                            version of our work are the first universal MMDE methods,
                            the original UniDepth and the newer UniDepthV2, which
                            predict a point in metric 3D space for each pixel without any
                            input other than a single image. An earlier version of this
                            work has appeared in the Conference on Computer Vision
                            and Pattern Recognition [17] and has introduced our original
                            UniDepth model. In [17], we have first designed a promptable
                            camera module, an architectural component that learns a dense
                            camera representation and allows for non-parametric camera
                            conditioning. Second, we have proposed a pseudo-spherical
                            representation of the output space, thus solving the intertwined
                            nature of camera and depth prediction. In addition, we have
                            introduced a geometric invariance loss to disentangle the camera information from the underlying 3D geometry of the scene.
                            Moreover, in the conference version, we have extensively
                            evaluated and compared UniDepth on ten different datasets in
                            a fair and comparable zero-shot setup to lay the ground for our
                            novel generalized MMDE task. Owing to its design, UniDepth
                            consistently set the state of the art even compared with nonzero-shot methods, ranking first at the time of its appearance in
                            the competitive official KITTI Depth Prediction Benchmark.
                            Compared to the aforementioned conference version, this
                            article makes the following additional contributions:
                            1) A revisited architectural design of the camera-conditioned
                            monocular metric depth estimator network, which makes
                            UniDepthV2 simpler, substantially more efficient in computation time and parameters, and at the same time more
                            accurate than UniDepth. This design upgrade pertains to
                            the simplification of the connections between the Camera
                            Module and the Depth Module of the network, the more
                            economic sinusoidal embedding of the pinhole-based
                            dense camera representations fed to the Depth Module
                            that we newly adopt, the inclusion of multi-resolution
                            features and convolutional layers in our depth decoder,
                            and the application of the geometric invariance loss solely
                            on output-space features.
                            2) A novel edge-guided scale-shift-invariant loss, which is
                            computed from the predicted and ground-truth depth
                            maps around geometric edges of the input, encourages
                            UniDepthV2 to preserve the local structure of the depth
                            map better, and thus enhances the sharpness of depth outputs substantially compared to UniDepth even on camera
                            and scene domains which are unseen during training.
                            3) An improved practical training strategy that presents the
                            network with a greater diversity of input image shapes
                            and resolutions within each mini-batch and hence with
                            a larger range of intrinsic parameters of the assumed
                            pinhole camera model, leading to increased robustness
                            to the specific input distribution during inference.
                            4) An additional, uncertainty-level output, which requires no
                            additional supervisory signal during training yet allows
                            to quantify confidence during inference reliably and thus
                            enables downstream applications to geometric perception,
                            e.g. control, which require confidence-aware depth inputs.
                            The methodological novelties introduced lead to improved performance, robustness, and efficiency of UniDepthV2 compared
                            to UniDepth across a wide range of camera and scene domains.
                            This is demonstrated through an extensive set of comparisons
                            to the latest state-of-the-art methods as well as ablation studies
                            on 10 depth estimation benchmarks, both in the challenging
                            zero-shot evaluation setting and in the practical supervised
                            fine-tuning setting. UniDepthV2 sets the overall new state of
                            the art in MMDE and ranks first among published methods
                            in the competitive official public KITTI Depth Prediction
                            Benchmark."""
    },

    {
        "method_name" : "UniDepth",
        "SILog" : 8.13,
        "arxiv_link" : "https://arxiv.org/abs/2403.18913",
        "title" : """UniDepth: Universal Monocular Metric Depth Estimation""",
        "abstract" : """Accurate monocular metric depth estimation (MMDE)
                        is crucial to solving downstream tasks in 3D perception
                        and modeling. However, the remarkable accuracy of recent
                        MMDE methods is confined to their training domains. These
                        methods fail to generalize to unseen domains even in the presence of moderate domain gaps, which hinders their practical
                        applicability. We propose a new model, UniDepth, capable
                        of reconstructing metric 3D scenes from solely single images
                        across domains. Departing from the existing MMDE methods, UniDepth directly predicts metric 3D points from the
                        input image at inference time without any additional information, striving for a universal and flexible MMDE solution.
                        In particular, UniDepth implements a self-promptable camera module predicting dense camera representation to condition depth features. Our model exploits a pseudo-spherical
                        output representation, which disentangles camera and depth
                        representations. In addition, we propose a geometric invariance loss that promotes the invariance of camera-prompted
                        depth features. Thorough evaluations on ten datasets in a
                        zero-shot regime consistently demonstrate the superior performance of UniDepth, even when compared with methods
                        directly trained on the testing domains. Code and models
                        are available at: github.com/lpiccinelli-eth/unidepth.""",
        "introduction" : """The precise pixel-wise depth estimation is crucial to understanding the geometric scene structure, with applications in
                            3D modeling [10], robotics [11, 63], and autonomous vehicles [38, 51]. However, delivering reliable metric scaled
                            depth outputs is necessary to perform 3D reconstruction effectively, thus motivating the challenging and inherently illposed task of Monocular Metric Depth Estimation (MMDE).
                            While existing MMDE methods [3, 14, 16, 40, 41, 43,
                            61] have demonstrated remarkable accuracy across different benchmarks, they require training and testing on datasets
                            with similar camera intrinsics and scene scales. Moreover,
                            the training datasets typically have a limited size and contain little diversity in scenes and cameras. These characteristics result in poor generalization to real-world inference scenarios [52], where images are captured in uncontrolled, arbitrarily structured environments and cameras with arbitrary
                            intrinsics.
                            Only a few methods [21, 59] have addressed the challenging task of generalizable MMDE. However, these methods
                            assume controlled setups at test time, including camera intrinsics. While this assumption simplifies the task, it has
                            two notable drawbacks. Firstly, it does not address the full
                            application spectrum, e.g. in-the-wild video processing and
                            crowd-sourced image analysis. Secondly, the inherent camera parameter noise is directly injected into the model, leading to large inaccuracies in the high-noise case.
                            In this work, we address the more demanding task of generalizable MMDE without any reliance on additional external information, such as camera parameters, thus defining
                            the universal MMDE task. Our approach, named UniDepth,
                            is the first that attempts to solve this challenging task without restrictions on scene composition and setup and distinguishes itself through its general and adaptable nature. Unlike existing methods, UniDepth delivers metric 3D predictions for any scene solely from a single image, waiving the
                            need for extra information about scene or camera. Furthermore, UniDepth flexibly allows for the incorporation of additional camera information at test time.
                            Our design introduces a camera module that outputs a
                            non-parametric, i.e. dense camera representation, serving as
                            the prompt to the depth module. However, relying only on
                            this single additional module clearly results in challenges related to training stability and scale ambiguity. We propose
                            an effective pseudo-spherical representation of the output
                            space to disentangle the camera and depth dimensions of
                            this space. This representation employs azimuth and elevation angle components for the camera and a radial component for the depth, forming a perfect orthogonal space between the camera plane and the depth axis. Moreover, the
                            camera components are embedded through Laplace spherical harmonic encoding. Figure 1 depicts our camera selfprompting mechanism and the output space. Additionally,
                            we introduce a geometric invariance loss to enhance the robustness of depth estimation. The underlying idea is that the
                            camera-conditioned depth features from two views of the
                            same image should exhibit reciprocal consistency. In particular, we sample two geometric augmentations, creating a
                            pair of different views for each training image, thus simulating different apparent cameras for the original scene.
                            Our overall contribution is the first universal MMDE
                            method, UniDepth, that predicts a point in metric 3D space
                            for each pixel without any input other than a single image. In
                            particular, first, we design a promptable camera module, an
                            architectural component that learns a dense camera representation and allows for non-parametric camera conditioning.
                            Second, we propose a pseudo-spherical representation of the
                            output space, thus solving the intertwined nature of camera
                            and depth prediction. In addition, we introduce a geometric
                            invariance loss to disentangle the camera information from
                            the underlying 3D geometry of the scene. Moreover, we extensively test UniDepth and re-evaluate seven MMDE Stateof-the-Art (SotA) methods on ten different datasets in a fair
                            and comparable zero-shot setup to lay the ground for the generalized MMDE task. Owing to its design, UniDepth consistently sets the new state of the art even compared with nonzero-shot methods, ranking first in the competitive official
                            KITTI Depth Prediction Benchmark."""
    },

    {
       "method_name" : "DCDepth",
        "SILog" : 9.60,
        "arxiv_link" : "https://arxiv.org/abs/2410.14980",
        "title" : """DCDepth: Progressive Monocular Depth Estimation in Discrete Cosine Domain""",
        "abstract" : """In this paper, we introduce DCDepth, a novel framework for the long-standing
                        monocular depth estimation task. Moving beyond conventional pixel-wise depth
                        estimation in the spatial domain, our approach estimates the frequency coefficients
                        of depth patches after transforming them into the discrete cosine domain. This
                        unique formulation allows for the modeling of local depth correlations within each
                        patch. Crucially, the frequency transformation segregates the depth information
                        into various frequency components, with low-frequency components encapsulating
                        the core scene structure and high-frequency components detailing the finer aspects.
                        This decomposition forms the basis of our progressive strategy, which begins with
                        the prediction of low-frequency components to establish a global scene context,
                        followed by successive refinement of local details through the prediction of higherfrequency components. We conduct comprehensive experiments on NYU-DepthV2, TOFDC, and KITTI datasets, and demonstrate the state-of-the-art performance
                        of DCDepth. Code is available at https://github.com/w2kun/DCDepth.""",
        "introduction" : """Monocular Depth Estimation (MDE) is a cornerstone topic within computer vision communities,
                            tasked with predicting the distance–or depth–of each pixel’s corresponding object from the camera
                            based solely on single image. As a pivotal technology for interpreting 3D scenes from 2D representations, MDE is extensively applied across various fields such as autonomous driving, robotics, and
                            3D modeling [45, 49, 9, 43], etc. However, MDE is challenged by the inherent ill-posed nature of
                            inferring 3D structures from 2D images, making it a particularly daunting task for traditional methodologies, which often hinge on particular physical assumptions or parametric models [40, 59, 31, 32].
                            Over the past decade, the field of computer vision has witnessed a substantial surge in the integration
                            of deep learning techniques. Many studies have endeavored to harness the robust learning capabilities
                            of end-to-end deep neural networks for MDE task, propelling the estimation accuracy to new heights.
                            Researchers have investigated a variety of methodologies, including regression-based [11, 19, 55],
                            classification-based [5, 12], and classification-regression based approaches [3, 20], to predict depth
                            on a per-pixel basis within the spatial domain. Despite these significant strides in enhancing accuracy,
                            current methods encounter two primary limitations: the first is the tendency to predict depth for
                            individual pixels in isolation, thus neglecting the crucial local inter-pixel correlations. The second
                            limitation is the reliance on a singular forward estimation process, which may not sufficiently capture
                            the complexities of 3D scene structures, thereby constraining their predictive performance.
                            To address the identified limitations, we propose to transfer depth estimation from the spatial domain
                            to the frequency domain. Instead of directly predicting metric depth values, our method focuses
                            on estimating the frequency coefficients of depth patches transformed using the Discrete Cosine Transform (DCT) [2, 6]. This strategy offers dual benefits: firstly, the DCT’s basis functions inherently capture the inter-pixel correlations within depth patches, thereby facilitating the model’s
                            learning of local structures. Secondly, the DCT decomposes depth information into distinct frequency components, where low-frequency components reflect the overall scene architecture, and
                            high-frequency components capture intricate local details. This dichotomy underpins our progressive
                            estimation methodology, which commences with the prediction of low-frequency coefficients to
                            grasp the macroscopic scene layout, subsequently refining the local geometries by inferring higherfrequency coefficients predicated on previous predictions. The spatial depth map is then accurately
                            reconstructed via the inverse DCT. We illustrate this progress in Fig. 1. To implement our progressive estimation, we introduce a Progressive Prediction Head (PPH) that conditions on previous
                            predictions from both spatial and frequency domains, and facilitates the sequential prediction of
                            higher-frequency components using a GRU-based mechanism. Furthermore, recognizing the DCT’s
                            energy compaction property–indicative of the concentration of signal data within low-frequency
                            components–we introduce a DCT-inspired downsampling technique to mitigate information loss
                            during the downsampling process. This technique is embedded within a Pyramid Feature Fusion
                            (PFF) module, ensuring effective fusion of multi-scale image features for accurate depth estimation.
                            Our contributions can be succinctly summarized in three key aspects:
                            • To the best of our knowledge, we are the first to formulate MDE as a progressive regression
                            task in the discrete cosine domain. Our proposed method not only models local correlations
                            effectively but also enables global-to-local depth estimation.
                            • We introduce a framework called DCDepth, comprising two novel modules: the PPH module
                            progressively estimates higher-frequency coefficients based on previous predictions, and the
                            PFF module incorporates a DCT-based downsampling technique to mitigate information loss
                            during downsampling and ensures effective integration of multi-scale features.
                            • We evaluate our approach through comprehensive experiments on NYU-Depth-V2 [36], TOFDC
                            [52], and KITTI [13] datasets. The results demonstrate the superior performance of DCDepth
                            compared to existing state-of-the-art methods."""
    }, 

    {
       "method_name" : "DiffusionDepth-I",
        "SILog" : 9.85,
        "arxiv_link" : "https://arxiv.org/abs/2303.05021",
        "title" : """DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation""",
        "abstract" : """Monocular depth estimation is a challenging task that
                        predicts the pixel-wise depth from a single 2D image. Current methods typically model this problem as a regression
                        or classification task. We propose DiffusionDepth, a new
                        approach that reformulates monocular depth estimation as
                        a denoising diffusion process. It learns an iterative denoising process to ‘denoise’ random depth distribution into a
                        depth map with the guidance of monocular visual conditions. The process is performed in the latent space encoded by a dedicated depth encoder and decoder. Instead
                        of diffusing ground truth (GT) depth, the model learns to reverse the process of diffusing the refined depth of itself into
                        random depth distribution. This self-diffusion formulation
                        overcomes the difficulty of applying generative models to
                        sparse GT depth scenarios. The proposed approach benefits
                        this task by refining depth estimation step by step, which is
                        superior for generating accurate and highly detailed depth
                        maps. Experimental results on KITTI and NYU-Depth-V2
                        datasets suggest that a simple yet efficient diffusion approach could reach state-of-the-art performance in both indoor and outdoor scenarios with acceptable inference time.
                        Codes are available through link.""",
        "introduction" : """Monocular depth estimation is a fundamental vision task
                            with numerous applications such as autonomous driving,
                            robotics, and augmented reality. Along with the rise of convolutional neural networks (CNNs) [20, 55, 12], numerous
                            mainstream methods employ it as dense per-pixel regression problems, such as RAP [65], DAV [24], and BTS [29].
                            Follow-up approaches such as UnetDepth [19], CANet [59],
                            and BANet [2], concentrate on enhancing the visual feature
                            by modifying the backbone structure. Transformer structures [57, 11, 36, 62] is introduced by DPT [43], and PixelFormer [1] pursue the performance to a higher level by
                            replacing CNNs for better visual representation. However,
                            pure regression methods suffer from severe overfitting and
                            unsatisfactory object details.
                            Estimating depth from a single image is challenging
                            due to the inherent ambiguity in the mapping between the
                            2D image and the 3D scene. To increase the robustness,
                            the following methods utilizing constructed additional constraints such as uncertainty (UCRDepth [47]), and piecewise planarity prior (P3Depth [41]). The NewCRFs [64]
                            introduces window-separated Conditional Random Fields
                            (CRF) to enhance local space relation with neighbor pixels. DORN [15], and Soft Ordinary [10] propose to discretize continuous depth into several intervals and reformulate the task as a classification problem on low-resolution
                            feature maps. Follow-up methods (AdaBins [5, 26], BinsFormer [33]) merge regression results with classification
                            prediction from bin centers. However, the discretization
                            depth values from bin centers result in lower visual quality with discontinuities and blur.
                            We solve the depth estimation task by reformulating it
                            as an iterative denoising process that generates the depth
                            map from random depth distribution. The brief process is
                            described in Fig. 1. Intuitively, the iterative refinement enables the framework to capture both coarse and fine details
                            in the scene at different steps. Meanwhile, by denoising
                            with extracted monocular guidance on large latent space,
                            this framework enables accurate depth prediction in high
                            resolution. Diffusion models have shown remarkable success in generation tasks [23, 56], or more recently, on detection [7] and segmentation [8, 7] tasks. To the best of our
                            knowledge, this is the first work introducing the diffusion
                            model into depth estimation.
                            This paper proposes DiffusionDepth, a novel framework
                            for monocular depth estimation as described in Fig. 2. The
                            framework takes in a random depth distribution as input
                            and iteratively refines it through denoising steps guided by
                            visual conditions. By performing the diffusion-denoising
                            process in latent depth space [45], DiffusionDepth is able to
                            achieve more accurate depth estimation with higher resolution. The depth latent is composed of a subtle encoder and
                            decoder. The denoising process is guided by visual conditions by merging it with the denoising block through a
                            hierarchical structure (Fig. 3). The visual backbone extracts
                            multi-scale features from monocular visual input and aggregated it through a feature pyramid (FPN [34]). We aggregated both global and local correlations to construct a strong
                            monocular condition.
                            One severe problem of adopting generative methods into
                            depth prediction is the sparse ground truth (GT) depth problem , which can lead to mode collapse in normal generative
                            training. To address this issue, DiffusionDepth introduces a
                            self-diffusion process. During training, instead of directly
                            diffusing on sparse GT depth values, the model gradually
                            adds noise to refined depth latent from the current denoising
                            output. The supervision is achieved by aligning the refined
                            depth predictions with the sparse GT values in both depth
                            latent space and pixel-wise depth through a sparse valid
                            mask. With the help of random crop, jitter, and flip augmentation in training, this process lets the generative model
                            organize the entire depth map instead of just regressing on
                            known parts, which largely improves the visual quality of
                            the depth prediction.
                            The proposed DiffusionDepth framework is evaluated
                            on widely used public benchmarks KITTI [16] and NYUDepth-V2 [44], covering both indoor and outdoor scenarios.
                            It could reach 0.298 and 1.452 RMSE on official offline test
                            split respectively on NYU-Depth-V2 and KITTI datasets,
                            which exceeds state-of-the-art (SOTA) performance. To
                            better understand the effectiveness and properties of the
                            diffusion-based approach for 3D perception tasks, we conduct a detailed ablation study. It discusses the impact of
                            different components and design choices on introducing the
                            diffusion approach to 3D perception, providing valuable
                            insights as references for related tasks such as stereo and
                            depth completion. The contribution of this paper could be
                            summarized in threefold.
                            • This work proposes a novel approach to monocular
                            depth estimation by reformulating it as an iterative
                            diffusion-denoising problem with visual guidance.• Experimental results suggest DiffusionDepth achieves
                            state-of-the-art performance on both offline and offline
                            evaluations with affordable inference costs.
                            • This is the first work introducing the diffusion model
                            into depth estimation, providing extensive ablation
                            component analyses, and valuable insights for potentially related 3D vision tasks"""
    },

    {
       "method_name" : "PixelFormer",
        "SILog" : 10.28,
        "arxiv_link" : "https://arxiv.org/abs/2210.09071", 
        "title" : """Attention Attention Everywhere: Monocular Depth Prediction with Skip Attention""", 
        "abstract" : """Monocular Depth Estimation (MDE) aims to predict
                        pixel-wise depth given a single RGB image. For both, the
                        convolutional as well as the recent attention-based models,
                        encoder-decoder-based architectures have been found to be
                        useful due to the simultaneous requirement of global context and pixel-level resolution. Typically, a skip connection module is used to fuse the encoder and decoder features, which comprises of feature map concatenation followed by a convolution operation. Inspired by the demonstrated benefits of attention in a multitude of computer vision problems, we propose an attention-based fusion of encoder and decoder features. We pose MDE as a pixel query
                        refinement problem, where coarsest-level encoder features
                        are used to initialize pixel-level queries, which are then
                        refined to higher resolutions by the proposed Skip Attention Module (SAM). We formulate the prediction problem
                        as ordinal regression over the bin centers that discretize
                        the continuous depth range and introduce a Bin Center
                        Predictor (BCP) module that predicts bins at the coarsest
                        level using pixel queries. Apart from the benefit of image
                        adaptive depth binning, the proposed design helps learn
                        improved depth embedding in initial pixel queries via direct supervision from the ground truth. Extensive experiments on the two canonical datasets, NYUV2 and KITTI,
                        show that our architecture outperforms the state-of-the-art
                        by 5.3% and 3.9%, respectively, along with an improved
                        generalization performance by 9.4% on the SUNRGBD
                        dataset. Code is available at https://github.com/
                        ashutosh1807/PixelFormer.git.""", 
        "introduction" : """Monocular Depth Estimation (MDE) is a well-studied
                            topic in computer vision. State-of-the-art (SOTA) techniques for MDE are based on encoder-decoder style Convolutional Neural Network (CNN) architectures [16, 3, 4, 33,
                            20, 17]. Due to the inherently local nature of a convolution kernel, early-stage feature maps have higher resolution but
                            lack a global receptive field. The feature pyramidal-based
                            decoder mitigates the issue by fusing low-resolution, semantically rich decoder features with the higher resolution
                            but semantically weaker encoder features via a top-down
                            path-way and lateral connections called skip connections
                            [18]. Inline with the recent success of transformers, many
                            latest works have used a self-attention based architectures
                            for MDE [1, 32, 34]. Self-attention increases the receptive
                            field and allows to capture long-range dependencies in feature maps. Practically, it is challenging to use self-attention
                            for high-resolution feature maps due to memory and computational constraints. Hence, the current SOTA [34] uses
                            window based attention using Swin transformer-based encoder backbone [19] to improve efficiency.
                            We observe that SOTA [1, 34] techniques are highly
                            accurate in aligning depth edges with the object boundaries. However, there exists a confusion in giving the
                            depth label to a pixel (c.f. Fig. 1). We posit this due to
                            the inability of current techniques to effectively fuse highresolution local features from the encoder and global contextual features from the decoder. Typically such a fusion is
                            achieved through a skip connection module implementing feature concatenation followed by a convolution operation.
                            Weights of the convolution kernels are highly localized,
                            which restricts the flow of semantic information from long
                            ranges affecting the ability of the model to predict the correct depth label for a pixel. To mitigate the constraint, we introduce a skip-attention module (SAM) that helps integrate
                            information using window-based cross-attention. SAM calculates self-similarity between pixel queries based on decoder features and their corresponding neighbors from the
                            encoder features in a predefined window to attend to and
                            aggregate information at a longer range. We implement the
                            overall architecture as a pixel query refinement problem.
                            We use the coarsest feature map from the encoder with maximum global information to initialize pixel queries using a
                            Pixel Query Initialiser Module. The pixel queries are then
                            refined with the help of a SAM module to finer scales.
                            Recent MDE techniques [1] formulate the problem as
                            a classification-regression one, in which the depth is predicted by a linear combination of bin centers discretized
                            over the depth range. The bin centers are predicted adaptively per image, allowing the network to concentrate on
                            the depth range regions that are more likely to occur in
                            the scene of the input image. A vision transformer that
                            aggregates global information from the output of another
                            encoder-decoder-based transformer model is typically used
                            to generate the bin centers. Since we pose MDE as a pixel
                            query refinement problem starting from the coarsest resolution, we propose a lightweight Bin Center Module (BCP)
                            that predicts bin centers based on the initial pixel queries.
                            This is more efficient than decoding features and then attending again in current SOTA [1]. The proposed design
                            also helps embed the depth information into the initial pixel
                            queries via direct ground truth supervision.
                            Contributions: The specific contributions of this work are
                            as follows: (1) We propose a novel strategy for predicting
                            depth using a single image by viewing it as a pixel query refinement problem. (2) We introduce a Skip Attention Module (SAM) that uses a window-based cross-attention module to refine pixel queries from the decoder feature maps
                            for cross-attending to higher resolution encoder features.
                            (3) We present a Bin Center Predictor (BCP) Module that
                            estimates bin centers adaptively per image using the global
                            information from the coarsest-level feature maps. This
                            helps to provide direct supervision to initial pixel queries
                            from ground truth depth, leading to better query embedding. (4) We combine the novel design elements in an
                            encoder-decoder framework comprised of a vision transformer backbone. The proposed architecture called PixelFormer achieves state-of-the-art (SOTA) performance on
                            indoor NYUV2 and outdoor KITTI datasets, improving the
                            current SOTA by 5.3% and 3.9%, in terms of absolute relative error and square relative error, respectively. Additionally, PixelFormer improves the generalization performance
                            by 9.4% over SOTA on the SUNRGBD dataset in terms of
                            absolute relative error."""
    },

    {
       "method_name" : "SideRT", 
        "SILog" : 11.42, 
        "arxiv_link" : "https://arxiv.org/abs/2204.13892", 
        "title" : """SideRT: A Real-time Pure Transformer Architecture for Single Image Depth Estimation""", 
        "abstract" : """Since context modeling is critical for estimating
                        depth from a single image, researchers put tremendous effort into obtaining global context. Many
                        global manipulations are designed for traditional
                        CNN-based architectures to overcome the locality
                        of convolutions. Attention mechanisms or transformers originally designed for capturing longrange dependencies might be a better choice, but
                        usually complicates architectures and could lead
                        to a decrease in inference speed. In this work,
                        we propose a pure transformer architecture called
                        SideRT that can attain excellent predictions in realtime. In order to capture better global context,
                        Cross-Scale Attention (CSA) and Multi-Scale Refinement (MSR) modules are designed to work collaboratively to fuse features of different scales efficiently. CSA modules focus on fusing features
                        of high semantic similarities, while MSR modules aim to fuse features at corresponding positions. These two modules contain a few learnable
                        parameters without convolutions, based on which
                        a lightweight yet effective model is built. This architecture achieves state-of-the-art performances in
                        real-time (51.3 FPS) and becomes much faster with
                        a reasonable performance drop on a smaller backbone Swin-T (83.1 FPS). Furthermore, its performance surpasses the previous state-of-the-art by a
                        large margin, improving AbsRel metric 6.9% on
                        KITTI and 9.7% on NYU. To the best of our knowledge, this is the first work to show that transformerbased networks can attain state-of-the-art performance in real-time in the single image depth estimation field. Code will be made available soon.""", 
        "introduction" : """Single image depth estimation (SIDE) has a pivotal role in extracting 3D geometry, which has a wide range of practical applications, including automatic driving, robotics navigation,
                            and augmented reality. The main difficulty of SIDE is that:
                            unlike other 3D vision problems, multiple views are missing
                            to establish the geometric relationship as 3D geometric clues
                            can only be dug from a single image. In order to solve this ill-posed problem, the ability to
                            extract global context is paid tremendous attention, which
                            largely relies on the powerful learning capability of modern deep neural networks. CNN-based architectures [Eigen
                            et al., 2014; Fu et al., 2018; Lee et al., 2019; Qiao et al.,
                            2021] once dominate the SIDE field, due to the intrinsic locality of convolution, global context is only obtained near the
                            bottleneck. The global context is usually maintained in lowresolution feature maps. Essential clues for 3D structures like
                            local details are lost after consecutive convolutional operations. To obtain high-resolution global context, there has been
                            a trend in SIDE field to enlarge receptive field via large backbones [Huang et al., 2017; Xie et al., 2017; Sun et al., 2019],
                            feature pyramid [Lin et al., 2017], spatial pyramid pooling
                            [He et al., 2015] and atrous convolution [Chen et al., 2017;
                            Yang et al., 2018].
                            Another paradigm to extract global context is taking advantage of the long-range dependency modeling capability of the
                            attention mechanisms. The attention module [Vaswani et al.,
                            2017; Wang et al., 2018] computes the responses at each position by estimating matching scores to all positions and gathering the corresponding embeddings accordingly, so a global
                            receptive field is guaranteed. Using attention as the main
                            component, transformers which are initially designed for natural language processing, are found more and more applications in the computer vision field [Dosovitskiy et al., 2020;
                            Liu et al., 2021; Carion et al., 2020]. Thanks to the powerful ability to establish long-range dependencies of attention mechanisms and transformers, integrating them into fully
                            convolutional architectures [Ranftl et al., 2021; Bhat et al.,
                            2021; Yang et al., 2021] has pushed state-of-the-art performance forward a lot.
                            Since the attention mechanism is usually time- and
                            memory-consuming, inference speed has to be compromised
                            when using transformers or attention mechanisms. Many
                            works have been devised for more efficient implementation,
                            but similar works are rare in the SIDE field.
                            This paper explores how to achieve state-of-the-art performance in real-time when using transformers and attention
                            mechanisms. We introduce the SIDE Real-time Transformer
                            (SideRT) based on an encoder-decoder architecture. Swin
                            transformers are used as the encoder. The decoder is built
                            on a novel attention mechanism named Cross-Scale Attention
                            (CSA) and a Multi-Scale Refinement module (MSR). Both
                            CSA and MSR modules are global operations and work collaboratively. In CSA modules, finer-resolution features are
                            augmented by coarser-resolution features according to attention scores defined by semantic similarity. In MSR modules, coarser-resolution features are merged to spatially corresponding finer-resolution features. Since a few learnable
                            parameters are used in the proposed modules, feature maps at
                            different scales are fused with a fair computational overhead.
                            Based on CSA and MSR modules, we build a lightweight
                            decoder that conducts hierarchical depth optimization progressively to get the final prediction in a coarse-to-fine manner. Furthermore, Multi-Stage Supervision (MSS) is added at
                            each stage to ease the training process.
                            As depicted in Figure 1, the proposed SideRT significantly outperforms the previous state-of-the-art at a speed
                            of 51.3 FPS. It improves the AbsRel metric from 0.058 to
                            0.054 on KITTI and from 0.103 to 0.093 on NYU. Moreover,
                            SideRT can achieve 0.060 AbsRel on KITTI, and 0.124 AbsRel on NYU on smaller backbone Swin-T [Liu et al., 2021]
                            at a speed of 83.1 FPS and 84.4 FPS respectively. To the
                            best of our knowledge, this is the first work to show that
                            transformer-based networks can attain state-of-the-art performance in real-time in the single image depth estimation field."""
    },

    {
       "method_name" :  "PFANet", 
        "SILog" : 11.84, 
        "arxiv_link" : "https://arxiv.org/abs/2403.01440", 
        "title" : """PYRAMID FEATURE ATTENTION NETWORK FOR MONOCULAR DEPTH PREDICTION""", 
        "abstract" : """Deep convolutional neural networks (DCNNs) have achieved
                        great success in monocular depth estimation (MDE). However, few existing works take the contributions for MDE of
                        different levels feature maps into account, leading to inaccurate spatial layout, ambiguous boundaries and discontinuous object surface in the prediction. To better tackle these
                        problems, we propose a Pyramid Feature Attention Network
                        (PFANet) to improve the high-level context features and lowlevel spatial features. In the proposed PFANet, we design
                        a Dual-scale Channel Attention Module (DCAM) to employ
                        channel attention in different scales, which aggregate global
                        context and local information from the high-level feature
                        maps. To exploit the spatial relationship of visual features, we
                        design a Spatial Pyramid Attention Module (SPAM) which
                        can guide the network attention to multi-scale detailed information in the low-level feature maps. Finally, we introduce
                        scale-invariant gradient loss to increase the penalty on errors
                        in depth-wise discontinuous regions. Experimental results
                        show that our method outperforms state-of-the-art methods
                        on the KITTI dataset.""", 
        "introduction" : """Monocular depth estimation (MDE) is an important task that
                            aims to predict pixel-wise depth from a single RGB image,
                            and has many applications in computer vision, such as 3D reconstruction, scene understanding, autonomous driving and
                            intelligent robots [1]. In the meanwhile, MDE is a technically
                            ill-posed problem as a single image can be projected from an
                            infinite number of different 3D scenes. To solve this inherent ambiguity, one possibility is to leverage prior auxiliary
                            information, such as texture information, occlusion, object
                            locations, perspective, and defocus [2], but it is not easy to
                            effectively extract useful prior information.
                            More recently, some works on MDE based on encoderdecoder architecture have shown significant improvements
                            in performance by using deep convolutional neural networks
                            (DCNNs) [3]. As backbone for encoder, very powerful deep networks such as ResNet [4], DenseNet [5] or ResNext [6]
                            are widely adopted. These networks cascade multiple convolutions and spatial pooling layers to gradually increase the
                            receptive field and generate the high-level depth information.
                            In decoder phase, state-of-the-art methods are based on upsampling layer with global context module [7], skip connection, depth-to-space [8], multi-scale local planar guidance for
                            upsampling operation [3]. These methods directly fuse different scale features without considering their different contributions for MDE, which leads to ambiguous boundaries and discontinuous object surface in predicted depth (see Fig.1 (c)).
                            To tackle these problems, logarithmic discretization for ordinal regression [2] and attention module with structural awareness [9] are introduced to MDE network. However, the highlevel and low-level features play different roles in MDE. The
                            existing methods did not consider this aspect, which may affect the effective extraction of depth information.
                            In this paper, we propose a novel monocular depth estimation network named Pyramid Feature Attention Network
                            (PFANet). In order to enhance the global structural information in high-level features, we introduced the Dense version of
                            Atrous Spatial Pyramid Pooling (Dense ASPP) [10], which is
                            generally utilized in pixel-level semantic segmentation. Since
                            Dense ASPP applies sparse convolutions with various expansion rates, these convolutions expand receptive field of the
                            high-level features. And then we design Dual-scale Channel Attention Module (DCAM) to aggregate global context
                            and local information at different scales in high-level features.
                            During training process, DCAM assigns larger weight to the
                            channels that play an important role in MDE. Considering the
                            spatial relationship of visual features, we design Spatial Pyramid Attention Module (SPAM) to fuse the attention of multiscale low-level features. This module improves the detailed local information in the low-level features, which clearer object edge and smoother object surface in prediction depth. Besides, we introduce scale-invariant gradient loss [11] to lead
                            the network to learn more detail of object edges. With the
                            above operations, the proposed PFANet can produce good
                            depth maps (see Fig.1 (d)). In summary, our contributions
                            are as follows:
                            1) We propose a novel Pyramid Feature Attention Network (PFANet) for MDE. For high-level features, we design
                            Dual-scale Channel Attention Module (DCAM) to aggregate
                            global context and local information. For low-level features,
                            we design Spatial Pyramid Attention Module (SPAM) to capture more detailed information.
                            2) We introduce scale-invariant gradient loss to emphasize the depth discontinuity at different object boundaries and
                            enhance smoothness in homogeneous regions.
                            3) The proposed method achieves state-of-the-art results
                            on KITTI dataset."""
    },

    {
       "method_name" :  "VNL", 
        "SILog" : 12.65,
        "arxiv_link" : "https://arxiv.org/abs/1907.12209",
        "title" : """Enforcing geometric constraints of virtual normal for depth prediction""", 
        "abstract" : """Monocular depth prediction plays a crucial role in understanding 3D scene geometry. Although recent methods
                        have achieved impressive progress in evaluation metrics
                        such as the pixel-wise relative error, most methods neglect
                        the geometric constraints in the 3D space. In this work, we
                        show the importance of the high-order 3D geometric constraints for depth prediction. By designing a loss term that
                        enforces one simple type of geometric constraints, namely,
                        virtual normal directions determined by randomly sampled
                        three points in the reconstructed 3D space, we can considerably improve the depth prediction accuracy. Significantly,
                        the byproduct of this predicted depth being sufficiently accurate is that we are now able to recover good 3D structures
                        of the scene such as the point cloud and surface normal directly from the depth, eliminating the necessity of training
                        new sub-models as was previously done. Experiments on
                        two benchmarks: NYU Depth-V2 and KITTI demonstrate
                        the effectiveness of our method and state-of-the-art performance. Code is available at:
                        https://tinyurl.com/virtualnormal""", 
        "introduction" : """Monocular depth prediction aims to predict distances between scene objects and the camera from a single monocular image. It is a critical task for understanding the 3D
                            scene, such as recognizing a 3D object and parsing a 3D
                            scene.
                            Although the monocular depth prediction is an ill-posed
                            problem because many 3D scenes can be projected to the
                            same 2D image, many deep convolutional neural networks
                            (DCNN) based methods [7, 8, 12, 14, 24, 27, 35] have
                            achieved impressive results by using a large amount of labelled data, thus taking advantage of prior knowledge in labelled data to solve the ambiguity.
                            These methods typically formulate the optimization
                            problem as either point-wise regression or classification.
                            That is, with the i.i.d. assumption, the overall loss is summing over all pixels. To improve the performance, some endeavours have been made to employ other constraints besides the pixel-wise term. For example, a continuous conditional random field (CRF) [28] is used for depth prediction, which takes pair-wise information into account. Other
                            high-order geometric relations [9, 31] are also exploited,
                            such as designing a gravity constraint for local regions [9]
                            or incorporating the depth-to-surface-normal mutual transformation inside the optimization pipeline [31]. Note that,
                            for the above methods, almost all the geometric constraints
                            are ‘local’ in the sense that they are extracted from a small
                            neighborhood in either 2D or 3D. Surface normal is ‘local’
                            by nature as it is defined by the local tangent plane. As
                            the ground truth depth maps of most datasets are captured
                            by consumer-level sensors, such as the Kinect, depth values
                            can fluctuate considerably. Such noisy measurement would
                            adversely affect the precision and subsequently the effectiveness of those local constraints inevitably. Moreover, local constraints calculated over a small neighborhood have not fully exploited the structure information of the scene geometry that may be possibly used to boost the performance.
                            To address these limitations, here we propose a more
                            stable geometric constraint from a global perspective to
                            take long-range relations into account for predicting depth,
                            termed virtual normal. A few previous methods already
                            made use of 3D geometric information in depth estimation, almost all of which focus on using surface normal.
                            We instead reconstruct the 3D point cloud from the estimated depth map explicitly. In other words, we generate the 3D scene by lifting each RGB pixel in the 2D image to its corresponding 3D coordinate with the estimated
                            depth map. This 3D point cloud serves as an intermediate
                            representation. With the reconstructed point cloud, we can
                            exploit many kinds of 3D geometry information, not limited to the surface normal. Here we consider the long-range
                            dependency in the 3D space by randomly sampling three
                            non-colinear points with the large distance to form a virtual
                            plane, of which the normal vector is the proposed virtual
                            normal (VN). The direction divergence between groundtruth and predicted VN can serve as a high-order 3D geometry loss. Owing to the long-range sampling of points, the
                            adverse impact caused by noises in depth measurement is
                            much alleviated compared to the computation of the surface
                            normal, making VN significantly more accurate. Moreover,
                            with randomly sampling we can obtain a large number of
                            such constraints, encoding the global 3D geometric. Second, by converting estimated depth maps from images to
                            3D point cloud representations it opens many possibilities
                            of incorporating algorithms for 3D point cloud processing
                            to 2D images and 2.5D depth processing. Here we show
                            one instance of such possibilies.
                            By combining the high-order geometric supervision and
                            the pixel-wise depth supervision, our network can predict
                            not only an accurate depth map but also the high-quality 3D
                            point cloud, subsequently other geometry information such
                            as the surface normal. It is worth noting that we do not use
                            a new model or introduce network branches for estimating
                            the surface normal. Instead it is computed directly from the
                            reconstructed point cloud. The second row of Fig. 1 demonstrates an example of our results. By contrast, although the
                            previously state-of-the-art method [18] predicts the depth
                            with low errors, the reconstructed point cloud is far away
                            from the original shape (see, e.g., left part of ‘sofa’). The
                            surface normal also contains many errors. We are probably
                            the first to achieve high-quality monocular depth and surface normal prediction with a single network.
                            Experimental results on NYUD-v2 [36] and KITTI [13]
                            datasets demonstrate state-of-the-art performance of our
                            method. Besides, when training with the lightweight backbone, MobileNetV2 [34], our framework provides a better
                            trade-off between network parameters and accuracy. Our
                            method outperforms other state-of-the-art real-time systems
                            by up to 29% with a comparable number of network parameters. Furthermore, from the reconstructed point cloud, we
                            directly calculate the surface normal, with a precision being
                            on par with that of specific DCNN based surface normal
                            estimation methods.
                            In summary, our main contributions of this work are as
                            follow.
                            • We demonstrate the effectiveness of enforcing a highorder geometric constraint in the 3D space for the
                            depth prediction task. Such global geometry information is instantiated with a simple yet effective concept
                            termed virtual normal (VN). By enforcing a loss defined on VNs, we demonstrate the importance of 3D
                            geometry information in depth estimation, and design
                            a simple loss to exploit it.
                            • Our method can reconstruct high-quality 3D scene
                            point clouds, from which other 3D geometry features can be calculated, such as the surface normal.
                            In essence, we show that for depth estimation, one
                            should not consider the information represented by
                            depth only. Instead, converting depth into 3D point
                            clouds and exploiting 3D geometry is likely to improve
                            many tasks including depth estimation.
                            • Experimental results on NYUD-V2 and KITTI illustrate that our method achieves state-of-the-art performance."""
    },


    {
       "method_name" :  "VGG16-UNet", 
        "SILog" : 13.41,
        "arxiv_link" : "https://arxiv.org/abs/1808.06586", 
        "title" : """Learning Monocular Depth by Distilling Cross-domain Stereo Networks""", 
        "abstract" : """Monocular depth estimation aims at estimating a pixelwise
                        depth map for a single image, which has wide applications in scene understanding and autonomous driving. Existing supervised and unsupervised
                        methods face great challenges. Supervised methods require large amounts
                        of depth measurement data, which are generally difficult to obtain, while
                        unsupervised methods are usually limited in estimation accuracy. Synthetic data generated by graphics engines provide a possible solution for
                        collecting large amounts of depth data. However, the large domain gaps
                        between synthetic and realistic data make directly training with them
                        challenging. In this paper, we propose to use the stereo matching network
                        as a proxy to learn depth from synthetic data and use predicted stereo
                        disparity maps for supervising the monocular depth estimation network.
                        Cross-domain synthetic data could be fully utilized in this novel framework. Different strategies are proposed to ensure learned depth perception capability well transferred across different domains. Our extensive
                        experiments show state-of-the-art results of monocular depth estimation
                        on KITTI dataset.""", 
        "introduction" : """Depth estimation is an important computer vision task, which is a basis for
                            understanding 3D geometry and could assist other vision tasks including object detection, tracking, and recognition. Depth can be recovered by varieties
                            of methods, such as stereo matching [14,27], structure from motion [40,1,44],
                            SLAM systems [28,7,29], and light field [38]. Recently, monocular depth prediction from a single image [6,9,11] was investigated with deep Convolutional
                            Neural Networks (CNN).
                            Deep CNNs could inherently combine local and global contexts of a single
                            image to learn depth maps. The methods are mainly divided into two categories,
                            supervised and unsupervised methods. For deep CNN based supervised methods [6,5], neural networks are directly trained with ground-truth depths, where conditional random fields (CRF) are optionally used to refine the final results.
                            For unsupervised methods, the photometric loss is used to match pixels between
                            images from different viewpoints by warping-based view synthesis. Some methods [9,11] learn to predict depth maps by matching stereo images, while some
                            other ones [50,47] learn depth and camera poses simultaneously from video frame
                            sequences.
                            There are several challenges for existing monocular depth estimation methods. Supervised learning methods require large amounts of annotated data, and
                            depth annotations need to be carefully aligned and calibrated. Ground truth
                            captured by LIDAR is generally sparse, and structured light depth sensors do
                            not work in strong light. Unsupervised learning methods [9,11] suffer from low
                            texture, repeated pattern, and occlusions. It is hard to recover depth in occlusion regions with only the photometric loss because of the lack of cross-image
                            correspondences at those regions.
                            Learning from synthetic data with accurate depth maps could be a potential
                            way to tackle the above problems, but this requires synthetic data to be similar
                            as realistic data in contents, appearance and viewpoints to ensure the model
                            transferability. Otherwise, it is hard to adapt the model to realistic data due to
                            the large domain gap. For example, a monocular depth estimation network pretrained with indoor synthetic data will have a bad performance in driving scenes,
                            but it will perform better if pretrained with synthetic driving scene datasets like
                            virtual KITTI [8]. As a result, a lot of works are needed to build up corresponding synthetic datasets if the algorithm needs to be deployed in different
                            scenes. On the other hand, we find that for state-of-the-art stereo matching algorithms [27,2], the stereo networks pretrained on cross-domain synthetic stereo
                            images generalize much better to new domains compared with monocular depth
                            networks, because the network learns the concept of matching pixels across stereo
                            images instead of understanding high-level semantic meanings. Recently, stereo
                            matching algorithms have achieved great success with the introduction of deep
                            CNNs and synthetic datasets like Scene Flow datasets [27], which inspires us to
                            use stereo matching as a proxy task to learn depth maps from stereo image pairs,
                            which can better utilize synthetic data and alleviate domain transfer problem
                            compared with directly training monocular depth networks.
                            In this paper, we propose a new pipeline for monocular depth learning with
                            the guidance of stereo matching networks pretrained with cross-domain synthetic datasets. Our pipeline consists of three steps. First, we use a variant of
                            DispNet [27] to predict disparity maps and occlusion masks with synthetic Scene
                            Flow datasets. Then, the stereo matching network is finetuned with realistic data
                            in a supervised or our novel unsupervised way. Finally, the monocular depth estimation network is trained under the supervision of the stereo network.
                            Using stereo matching network as a proxy to learn depth has several advantages. On the one hand, stereo networks could efficiently make full use of crossdomain synthetic data and can be easier adapted to new domains compared to
                            learning monocular depth. The synthetic datasets do not need to be separately
                            designed for different scenes. On the other hand, the input data for stereo networks could be augmented by cropping and resizing to avoid over-fitting, while
                            monocular depth networks usually fail to learn augmented images because it
                            is sensitive to viewpoint changes. The experiment results show that our stereo
                            matching network trained with synthetic data provides strong guidance for training monocular depth network, which could capture sharp boundaries and clear
                            thin structures.
                            Our method achieves state-of-the-art results on the KITTI [10] dataset. Our
                            contributions are as follows. 1) We propose a novel monocular depth learning
                            pipeline, which takes advantages of the power of stereo matching networks and
                            synthetic data. By using stereo matching as a proxy task, the synthetic-torealistic cross-domain problem could be effectively alleviated. 2) A novel unsupervised fine-tuning method is proposed based on the pretrained network to
                            avoid occlusion problem and improve smoothness regularization. Visualization
                            results show shaper boundaries and better occlusion predictions compared with
                            previous unsupervised methods. 3) Our proposed pipeline achieves state-of-theart monocular depth estimation performance in both unsupervised and semisupervised settings."""
    },

    {
       "method_name" :  "MT-SfMLearner", 
        "SILog" : 14.25, 
        "arxiv_link" : "https://arxiv.org/abs/2202.03131", 
        "title" : """Transformers in Self-Supervised Monocular Depth Estimation with Unknown Camera Intrinsics""", 
        "abstract" : """The advent of autonomous driving and advanced driver assistance systems necessitates continuous developments in computer vision for 3D scene understanding. Self-supervised monocular depth estimation, a method
                        for pixel-wise distance estimation of objects from a single camera without the use of ground truth labels, is
                        an important task in 3D scene understanding. However, existing methods for this task are limited to convolutional neural network (CNN) architectures. In contrast with CNNs that use localized linear operations
                        and lose feature resolution across the layers, vision transformers process at constant resolution with a global
                        receptive field at every stage. While recent works have compared transformers against their CNN counterparts for tasks such as image classification, no study exists that investigates the impact of using transformers
                        for self-supervised monocular depth estimation. Here, we first demonstrate how to adapt vision transformers for self-supervised monocular depth estimation. Thereafter, we compare the transformer and CNN-based
                        architectures for their performance on KITTI depth prediction benchmarks, as well as their robustness to
                        natural corruptions and adversarial attacks, including when the camera intrinsics are unknown. Our study
                        demonstrates how transformer-based architecture, though lower in run-time efficiency, achieves comparable
                        performance while being more robust and generalizable.""", 
        "introduction" : """There have been rapid improvements in scene understanding for robotics and advanced driver assistance systems (ADAS) over the past years. This success is attributed to the use of Convolutional Neural
                            Networks (CNNs) within a mostly encoder-decoder
                            paradigm. Convolutions provide spatial locality and
                            translation invariance which has proved useful for image analysis tasks. The encoder, often a convolutional
                            Residual Network (ResNet) (He et al., 2016), learns
                            feature representations from the input and is followed
                            by a decoder which aggregates these features and converts them into final predictions. However, the choice
                            of architecture has a major impact on the performance
                            and generalizability of the task.
                            While CNNs have been the preferred architecture in computer vision, transformers have also recently gained traction (Dosovitskiy et al., 2021) motivated by their success in natural language processing (Vaswani et al., 2017). Notably, they have
                            also outperformed CNNs for object detection (Carion et al., 2020) and semantic segmentation (Zheng
                            et al., 2021). This is also reflected in methods for
                            monocular dense depth estimation, a pertinent task
                            for autonomous planning and navigation, where supervised transformer-based methods (Li et al., 2020;
                            Ranftl et al., 2021) have been proposed as an alternative to supervised CNN-based methods (Lee
                            et al., 2019; Aich et al., 2021). However, supervised methods require extensive RGB-D ground truth
                            collected from costly LiDARs or multi-camera rigs.
                            Instead, self-supervised methods have increasingly
                            utilized concepts of Structure from Motion (SfM)
                            with known camera intrinsics to train monocular
                            depth and ego-motion estimation networks simultaneously (Guizilini et al., 2020; Lyu et al., 2020; Chawla
                            et al., 2021). While transformer ingredients such as
                            attention have been utilized for self-supervised depth
                            estimation (Johnston and Carneiro, 2020), most methods are nevertheless limited to the use of CNNs that
                            have localized linear operations and lose feature resolution during downsampling to increase their limited
                            receptive field (Yang et al., 2021). ductive biases allow for more globally coherent predictions with different layers attending to local and
                            global features simultaneously (Touvron et al., 2021).
                            However, transformers require more training data
                            and can be more computationally demanding (Caron
                            et al., 2021). While multiple studies have compared
                            transformers against CNNs for tasks such as image
                            classification (Raghu et al., 2021; Bhojanapalli et al.,
                            2021), no study exists that evaluates the impact of
                            transformers in self-supervised monocular depth estimation, including when the camera intrinsics may
                            be unknown.
                            In this work, we conduct a comparative study
                            between CNN- and transformer-based architectures
                            for self-supervised monocular depth estimation. Our
                            contributions are as follows:
                            • We demonstrate how to adapt vision transformers for self-supervised monocular depth estimation by implementing a method called MonocularTransformer SfMLearner (MT-SfMLearner).
                            • We compare MT-SfMLearner and CNNs for
                            their performance on the KITTI monocular depth
                            Eigen Zhou split (Eigen et al., 2014) and the online depth prediction benchmark (Geiger et al.,
                            2013).
                            • We investigate the impact of architecture choices
                            for the individual depth and ego-motion networks
                            on performance as well as robustness to natural
                            corruptions and adversarial attacks.
                            • We also introduce a modular method that simultaneously predicts camera focal lengths and principal point from the images themselves and can easily be utilized within both CNN- and transformerbased architectures.
                            • We study the accuracy of intrinsics estimation as
                            well as its impact on the performance and robustness of depth estimation.
                            • Finally, we also compare the run-time computational and energy efficiency of the architectures
                            for depth and intrinsics estimation.
                            MT-SfMLearner provides real-time depth estimates and illustrates how transformer-based architecture, though lower in run-time efficiency, can achieve
                            comparable performance as the CNN-based architectures while being more robust under natural corruptions and adversarial attacks, even when the camera intrinsics are unknown. Thus, our work presents
                            a way to analyze the trade-off between the performance, robustness, and efficiency of transformer- and
                            CNN-based architectures for depth estimation."""
    }, 

    {
       "method_name" : "DiPE",  
        "SILog" : 14.84, 
        "arxiv_link" : "https://arxiv.org/abs/2003.01360", 
        "title" : """DiPE: Deeper into Photometric Errors for Unsupervised Learning of Depth and Ego-motion from Monocular Videos""", 
        "abstract" : """Unsupervised learning of depth and ego-motion
                        from unlabelled monocular videos has recently drawn great
                        attention, which avoids the use of expensive ground truth in the
                        supervised one. It achieves this by using the photometric errors
                        between the target view and the synthesized views from its adjacent source views as the loss. Despite significant progress, the
                        learning still suffers from occlusion and scene dynamics. This
                        paper shows that carefully manipulating photometric errors
                        can tackle these difficulties better. The primary improvement
                        is achieved by a statistical technique that can mask out the
                        invisible or nonstationary pixels in the photometric error map
                        and thus prevents misleading the networks. With this outlier
                        masking approach, the depth of objects moving in the opposite
                        direction to the camera can be estimated more accurately.
                        To the best of our knowledge, such scenarios have not been
                        seriously considered in the previous works, even though they
                        pose a higher risk in applications like autonomous driving.
                        We also propose an efficient weighted multi-scale scheme to
                        reduce the artifacts in the predicted depth maps. Extensive
                        experiments on the KITTI dataset show the effectiveness of
                        the proposed approaches. The overall system achieves state-oftheart performance on both depth and ego-motion estimation.""",
        "introduction" : """The depth and ego-motion estimation is the core problem
                            in Simultaneous Localization And Mapping (SLAM). Recently, Monocular Depth Estimation (MDE) attracts much
                            attention, as it can be flexibly used in many applications,
                            such as autonomous mobile robotics and AR/VR. Tracking
                            the 6-DoF motion for a moving camera is also critical for
                            these applications. Traditional supervised methods require
                            expensively-collected ground truth, resulting in limited ability in generalization. By contrast, unsupervised learning from
                            monocular videos [1] is a much more generalizable solution.
                            The unsupervised learning models usually contain two
                            networks for predicting the depth map of the target view,
                            and the motion between the target view and its temporally
                            adjacent views. With the network output, the target view
                            can be reconstructed by the adjacent source views with
                            image warping, and the resulted photometric loss can be
                            used as the supervisory signal for learning. However, the
                            image reconstruction is usually destroyed by between-view
                            occlusion and scene dynamics, as illustrated in Fig. 1, and the
                            resulting incorrect supervision harms the network learning. The theory of how minimizing between-view reconstruction errors affects the depth estimation of occluded regions
                            and the common forward and backward moving objects is
                            illustrated in Fig. 1. Many methods have been proposed to
                            cope with the occlusion and dynamics, and considerable
                            improvement has been made. For example, the effect of
                            ‘dark holes’ by the co-directionally moving objects has been
                            tackled in the latest work [2], [3], [4]. However, as shown in
                            Fig. 4 the latest models make significant underestimation of
                            the depth for the contra-directionally moving objects. To the
                            best of our knowledge, the inaccuracy of such objects has not
                            been reported in the literature, which may cause trouble in
                            practical applications. For instance, in autonomous driving,
                            if the distance of oncoming cars is underrated, unnecessary
                            braking or avoiding may be executed.
                            This issue can be largely avoided by our proposed outlier
                            masking technique, which helps to exclude the occluded
                            and moving regions, especially the oncoming objects. The
                            technique is driven by our observation that the photometric
                            errors of occluded and dynamic regions are much larger. In
                            theory, the visible background usually dominates the scenes and the invisible or moving pixels are inconsistent with the
                            background, thus making their errors difficult to optimize.
                            Besides, we also propose an efficient weighted multi-scale
                            scheme to reduce artifacts and work with the outlier masking
                            to produce better depth maps.
                            The effectiveness of our two main contributions, as mentioned above, is experimentally proven on the driving KITTI
                            dataset. Together with a simple baseline model and some
                            other masking practices, we build an overall state-of-theart unsupervised monocular depth and ego-motion estimation
                            system, called DiPE."""
    },

    {
       "method_name" :  "SGDepth", 
        "SILog" : 15.30, 
        "arxiv_link" : "https://arxiv.org/abs/2007.06936", 
        "title" : """Self-Supervised Monocular Depth Estimation: Solving the Dynamic Object Problem by Semantic Guidance""", 
        "abstract" : """Self-supervised monocular depth estimation presents a powerful method to obtain 3D scene information from single camera images, which is trainable on arbitrary image sequences without requiring depth labels, e.g., from a LiDAR sensor. 
                        In this work we present a new self-supervised semantically-guided depth estimation (SGDepth) method to deal with moving dynamic-class (DC) objects, such as moving cars and pedestrians, which violate the static-world assumptions typically made during training of such models. 
                        Specifically, we propose (i) mutually beneficial cross-domain training of (supervised) semantic segmentation and self-supervised depth estimation with task-specific network heads, (ii) a semantic masking scheme providing guidance to prevent moving DC objects from contaminating the photometric loss, and (iii) a detection method for frames with non-moving DC objects, from which the depth of DC objects can be learned. 
                        We demonstrate the performance of our method on several benchmarks, in particular on the Eigen split, where we exceed all baselines without test-time refinement.""", 
        "introduction" : """The accurate estimation of depth information from a scene is essential for applications requiring a 3D environment model such as autonomous driving or virtual
                            reality. Therefore, a long-standing research field of computer vision is the prediction of depth maps from camera images. Classical model-based algorithms
                            can predict depth from stereo images [26] or from image sequences (videos)
                            [1], limited by the quality of the model. Deep learning enables the prediction
                            of depth from single monocular images by supervision from LiDAR or RGBD camera measurements [11,12,14]. More recently, self-supervised approaches
                            [16,18] were introduced which solely rely on geometric image projection models
                            and optimize the depth by minimizing photometric errors without the need of
                            any labels. While these self-supervised monocular depth estimation approaches
                            require only a single image as input during inference, they rely either on stereo
                            images [16], or on sequential images from a video [71] during training. For self-supervised monocular depth estimation from video data, the assumptions made during the geometric projections (which are required to calculate
                            the photometric error) impose several problems: Firstly, occlusions can occur
                            inducing artifacts in the photometric error. Secondly, consecutive more or less
                            identical frames caused by a lack of ego-motion present a problem as without
                            any movement between the frames no structure can be inferred. Thirdly, moving
                            dynamic-class (DC) objects such as cars, trucks and pedestrians violate the static
                            world assumption. Early approaches [38,71] did not address these problems. A
                            current state-of-the-art approach by Godard et al. [20] approaches the first two
                            problems by a minimum reprojection loss and an auto-masking technique, which
                            we adopt (same as [5,23,24]). The third problem was left open in [5,20,23,24].
                            Starting to approach this dynamic object problem, we first need to identify
                            dynamic-class (DC) objects pixel-wise by incorporating an image segmentation
                            technique. For this purpose previous approaches either rely on pre-trained segmentation networks [5,6,24,39], which are not available for arbitrary datasets, or
                            an implicit binary segmentation trained as part of the image projection model
                            [37,49,63], thereby coupled and limited to the projection quality. Our solution
                            is somewhat related to Chen et al. [7]: We jointly optimize depth estimation
                            and semantic segmentation, still keeping the depth estimation self-supervised by
                            training the supervised semantic segmentation in a different domain. However,
                            as [7] is limited to training on stereo images and proposes a unified decoder head
                            for both tasks, we transfer it to the monocular case and utilize gradient scaling described by [15] to enable cross-domain training with task-specific decoder
                            heads. This yields optimally learned task-specific weights inside the respective
                            decoders and the possibility to generalize the concept to even more tasks.
                            While we expect the depth estimation to take profit from sharper edges at
                            object boundaries provided by semantic segmentation, the DC objects have to
                            be handled once identified by the segmentation. In contrast to most other approaches [5,37,39,49,63], we do not extend the image projection model to include
                            DC objects, but simply exclude the pixels belonging to DC objects from the loss.
                            However, this alone would lead to a poor performance, as the depth of DC objects
                            would not be learned at all. Therefore, we propose a detection method for frames
                            with non-moving DC objects. From these frames the depth of (non-moving) DC
                            objects can be learned with the normal (valid) image projection model, while
                            in the other frames, the (moving) DC objects are excluded from the loss. Here,
                            our approach presents a significantly simpler, yet powerful method to handle DC
                            objects in self-supervised monocular depth estimation.
                            To sum up, our contribution to the field is threefold. Firstly, we generalize the
                            mutually beneficial cross-domain training of self-supervised depth estimation and
                            supervised semantic segmentation to a more general setting with task-specific
                            network heads. Secondly, we introduce a solution to the dynamic object problem
                            by using a novel semantically-masked photometric loss. Thirdly, we introduce
                            a novel method of detecting moving DC objects, which can then be excluded
                            from the training loss computation, while non-moving DC objects should still
                            contribute. We demonstrate the effectiveness of our approach on the KITTI
                            Eigen split, where we exceed all baselines without test-time refinement, as well
                            as on two further KITTI benchmarks."""
    },

    {
       "method_name" :  "packnSFMHR_RVC", 
        "SILog" : 15.80, 
        "arxiv_link" : "https://arxiv.org/abs/1905.02693", 
        "title" : """3D Packing for Self-Supervised Monocular Depth Estimation""", 
        "abstract" : """Although cameras are ubiquitous, robotic platforms typically rely on active sensors like LiDAR for direct 3D perception. In this work, we propose a novel self-supervised
                        monocular depth estimation method combining geometry
                        with a new deep network, PackNet, learned only from unlabeled monocular videos. Our architecture leverages novel
                        symmetrical packing and unpacking blocks to jointly learn
                        to compress and decompress detail-preserving representations using 3D convolutions. Although self-supervised, our
                        method outperforms other self, semi, and fully supervised
                        methods on the KITTI benchmark. The 3D inductive bias in
                        PackNet enables it to scale with input resolution and number of parameters without overfitting, generalizing better on
                        out-of-domain data such as the NuScenes dataset. Furthermore, it does not require large-scale supervised pretraining
                        on ImageNet and can run in real-time. Finally, we release
                        DDAD (Dense Depth for Automated Driving), a new urban
                        driving dataset with more challenging and accurate depth
                        evaluation, thanks to longer-range and denser ground-truth
                        depth generated from high-density LiDARs mounted on a
                        fleet of self-driving cars operating world-wide.""", 
        "introduction" : """Accurate depth estimation is a key prerequisite in many
                            robotics tasks, including perception, navigation, and planning. Depth from monocular camera configurations can
                            provide useful cues for a wide array of tasks [24, 31, 35,
                            37], producing dense depth maps that could complement
                            or eventually replace expensive range sensors. However,
                            learning monocular depth via direct supervision requires
                            ground-truth information from additional sensors and precise cross-calibration. Self-supervised methods do not suffer from these limitations, as they use geometrical constraints on image sequences as the sole source of supervision. In this work, we address the problem of jointly estimating scene structure and camera motion across RGB image sequences using a self-supervised deep network.
                            While recent works in self-supervised monocular depth estimation have mostly focused on engineering the loss
                            function [6, 34, 48, 54], we show that performance critically depends on the model architecture, in line with the
                            observations of [28] for other self-supervised tasks. Going
                            beyond image classification models like ResNet [21], our
                            main contribution is a new convolutional network architecture, called PackNet, for high-resolution self-supervised
                            monocular depth estimation. We propose new packing and
                            unpacking blocks that jointly leverage 3D convolutions to
                            learn representations that maximally propagate dense appearance and geometric information while still being able
                            to run in real time. Our second contribution is a novel
                            loss that can optionally leverage the camera’s velocity when
                            available (e.g., from cars, robots, mobile phones) to solve
                            the inherent scale ambiguity in monocular vision. Our
                            third contribution is a new dataset: Dense Depth for Automated Driving (DDAD). It leverages diverse logs from a
                            fleet of well-calibrated self-driving cars equipped with cameras and high-accuracy long-range LiDARs. Compared to
                            existing benchmarks, DDAD enables much more accurate
                            depth evaluation at range, which is key for high resolution
                            monocular depth estimation methods. Our experiments on the standard KITTI benchmark [17],
                            the recent NuScenes dataset [5], and our new proposed
                            DDAD benchmark show that our self-supervised monocular approach i) improves on the state of the art, especially at
                            longer ranges; ii) is competitive with fully supervised methods; iii) generalizes better on unseen data; iv) scales better
                            with number of parameters, input resolution, and more unlabeled training data; v) can run in real time at high resolution;
                            and vi) does not require supervised pretraining on ImageNet
                            to achieve state-of-the-art results; or test-time ground-truth
                            scaling if velocity information is available at training time."""
    },

    {
       "method_name" :  "MultiDepth", 
        "SILog" : 16.05, 
        "arxiv_link" : "https://arxiv.org/abs/1907.11111", 
        "title" : """MultiDepth: Single-Image Depth Estimation via Multi-Task Regression and Classification""", 
        "abstract" : """We introduce MultiDepth, a novel training strategy and convolutional neural network (CNN) architecture that allows approaching
                        single-image depth estimation (SIDE) as a multi-task problem. SIDE is an important part of road scene understanding. It, thus,
                        plays a vital role in advanced driver assistance systems and autonomous vehicles. Best results for the SIDE task so far have been
                        achieved using deep CNNs. However, optimization of regression problems, such as estimating depth, is still a challenging task. For
                        the related tasks of image classification and semantic segmentation, numerous CNN-based methods with robust training behavior
                        have been proposed. Hence, in order to overcome the notorious instability and slow convergence of depth value regression during
                        training, MultiDepth makes use of depth interval classification as an auxiliary task. The auxiliary task can be disabled at test-time
                        to predict continuous depth values using the main regression branch more efficiently. We applied MultiDepth to road scenes and
                        present results on the KITTI depth prediction dataset. In experiments, we were able to show that end-to-end multi-task learning with
                        both, regression and classification, is able to considerably improve training and yield more accurate results.""",
        "introduction" : """Depth estimation is an important part of scene understanding
                            in various domains. Traditionally, depth maps are derived from
                            active sensor measurements, such as light detection and ranging
                            (LiDAR) point clouds, or from stereo images. However, in the
                            absence of observations allowing for the explicit reconstruction
                            of pixel-wise depth values for a corresponding image, methods
                            for directly estimating depth from a single monocular image
                            are required. A typical application lies in robotics, most prominently autonomous vehicles, where a high degree of redundancy
                            is of vital importance. Figure 1 exemplarily shows the result of
                            single-image depth estimation (SIDE) for a road scene.
                            Predicting depth from a single image has seen substantial improvements due to the rise of deep learning-based methods.
                            First approaches to SIDE for indoor scenes using deep convolutional neural networks (CNNs) were presented by Eigen et al.
                            [11, 10]. Ever since then, various methods for the prediction
                            [28, 35, 30, 34, 21, 53] and evaluation [25] of depth maps for
                            indoor scenes have been proposed.
                            SIDE in unstructured outdoor environments poses an even
                            greater challenge. Annotated training data is hard to obtain,
                            as RGB-D cameras are not able to provide data at distances
                            of more than 10 m and their low resolution is not able to capture scenes crowded with differently sized objects. Available
                            datasets [14, 9, 22] use measurements from LiDAR sensors to
                            accumulate depth maps as ground-truth, which are, however,
                            naturally sparse. They can serve as an additional input for depth completion [7] or as labels for actual depth prediction.
                            Utilizing stereo image pairs and a photo-consistency loss for
                            semi-supervised training to estimate disparities is another option used in recent approaches [15, 47, 27]. The application
                            of SIDE in advanced driver assistance systems (ADAS) and
                            autonomous vehicles requires high precision and robustness.
                            Approaches to this challenging task have been proposed [13, 23,
                            43, 12, 18, 31, 26, 29, 50], but still require improvement for the
                            application in autonomous driving [43].
                            Estimating continuous depth values is a typical regression task. However, by discretizing depth space into intervals, it can be
                            cast as a classification problem [1, 19, 12]. While this is less
                            intuitive, classification methods have been found to converge
                            faster and more reliably. This was shown by Fu et al. [12], who
                            advanced this approach by taking into account the ordinal characteristic of depth intervals and achieved top-ranking results in
                            the KITTI depth prediction benchmark [45, 14]. Combining the
                            properties of both tasks, i.e., depth regression and classification
                            of depth intervals, in order to exploit their individual advantages
                            yields a multi-task problem.
                            Multi-task learning [3, 2] enables training of CNNs that produce
                            multiple outputs in a single round of inference. In his review
                            article, Ruder [40] gives an extensive overview of CNN-based
                            multi-task learning. Driven by advances in methodology [41,
                            23, 36, 17, 52], multi-task learning has become increasingly
                            popular in computer vision. It has successfully been applied
                            to numerous applications. In road scene understanding, problems that have been tackled using multi-task learning include
                            object detection and bounding box regression [44, 6, 4], as
                            well as SIDE in combination with surface normal estimation or
                            semantic segmentation [37, 49, 10, 38, 47, 23].
                            A different approach to employing multi-task learning is the
                            utilization of auxiliary tasks [33, 8, 46] that merely serve as
                            additional supervision to the network during training and are
                            discarded during test-time. This approach can be seen as an
                            extension to comprehensive regularization terms in loss functions as used by Li et al. [32]. It could be shown that by adding
                            auxiliary tasks to a network the performance of the main task
                            increases [33, 8].
                            Considering this prior work, we approach SIDE by posing it
                            as a multi-task problem with a main regression task and an
                            auxiliary classification task. As both tasks use depth measurements as ground-truth, with minor pre-processing applied in
                            order to segment the continuous depth space into intervals for
                            classification, the auxiliary supervisory signal does not require
                            additional annotations. By adding the auxiliary classification
                            task as a regularizer, we expect training to converge faster and
                            yield better results. Closely related to the idea of casting SIDE
                            to a classification task is the deep ordinal regression network
                            (DORN), proposed by Fu et al. [12]. They do, however, use
                            ordinal regression instead of classification and, furthermore, do
                            not treat it as an additional task. Kendall et al. [23] propose
                            uncertainty-based weighting of individual tasks, which we build
                            upon, but do not make use of auxiliary tasks. Auxiliary tasks
                            have been utilized before [33, 8, 46], however not with posing
                            the same task in two different ways. In contrast to Gurram et al.
                            [19], who use depth interval classification as pre-training, we
                            train for regression and classification in an end-to-end manner.
                            The main contribution of this paper is the proposal of MultiDepth, a novel multi-task approach to SIDE which incorporates
                            both regression and classification. This training strategy facilitates fast and robust convergence. We, furthermore, provide
                            an implementation of the proposed approach based on PSPNet
                            [51] and uncertainty-based weighting [23] that we used to show
                            the superiority of training with an auxiliary task as compared to
                            basic regression."""
    }, 

    {
       "method_name" :  "LSIM", 
        "SILog" : 17.92, 
        "arxiv_link" : "https://arxiv.org/abs/1905.00401", 
        "title" : """Learn Stereo, Infer Mono: Siamese Networks for Self-Supervised, Monocular, Depth Estimation""", 
        "abstract" : """The field of self-supervised monocular depth estimation
                        has seen huge advancements in recent years. Most methods
                        assume stereo data is available during training but usually
                        under-utilize it and only treat it as a reference signal. We
                        propose a novel self-supervised approach which uses both
                        left and right images equally during training, but can still
                        be used with a single input image at test time, for monocular depth estimation. Our Siamese network architecture
                        consists of two, twin networks, each learns to predict a disparity map from a single image. At test time, however, only
                        one of these networks is used in order to infer depth. We
                        show state-of-the-art results on the standard KITTI Eigen
                        split benchmark as well as being the highest scoring selfsupervised method on the new KITTI single view benchmark. To demonstrate the ability of our method to generalize to new data sets, we further provide results on the
                        Make3D benchmark, which was not used during training.""", 
        "introduction" : """Single-view depth estimation is a fundamental problem in computer vision with numerous applications in autonomous driving, robotics, computational photography,
                            scene understanding, and many others. Although single image depth estimation is an ill-posed problem [9, 18], humans are remarkably capable of adapting to estimate depth
                            from a single view [22]. Of course, humans can use stereo
                            vision, but when restricted to monocular vision, we can still
                            estimate depth fairly accurately by exploiting motion parallax, familiarity with known objects and their sizes, and
                            perspectives cues.
                            There is a large body of work on monocular depth estimation using classical computer vision methods [4, 8, 43,
                            45], including several recent approaches based on convolutional neural networks (CNN) [9, 35]. These methods, however, are supervised and require large quantities of ground
                            truth data. Obtaining ground truth depth data for realistic
                            scenes, especially in unconstrained viewing settings, is a
                            complicated task and typically involves special equipment
                            such as light detection and ranging (LIDAR) sensors.
                            Several methods recently tried to overcome this limitation, by taking a self-supervised approach. These methods exploit intrinsic geometric properties of the problem to
                            train monocular systems [11, 15]. All these cases, assume
                            that both images are available during training, though only
                            one training image is used as input to the network; the second image is only used as a reference. Godard et al. [15]
                            showed that predicting both the left and the right disparity maps vastly improves accuracy. While predicting the
                            left disparity using the left image is intuitive and straightforward, they also estimate the right disparity using the left
                            image. This process is prone to errors due to occlusions and
                            information missing from the left viewpoint. By comparison, we fully utilize both images when learning to estimate
                            disparity from a single image.
                            We propose a self-supervised approach similar to that of
                            Godard et al. [15]. Unlike them, however, we exploit the
                            symmetry of the disparity problem in order to obtain effective deep models. We observe that a key problem of existing
                            methods is that they try to train a single network to predict both left and right disparity maps using a single image.
                            This does not work well in practice since crucial information available in the right image is often occluded from the
                            left viewpoint due to parallax (and vice versa). Instead, we
                            propose a simple yet effective alternative approach of flipping the images around the vertical axis (vertical mirroring)
                            and using them for training. In this way, the network only
                            learns a left disparity map; right disparity maps are simply
                            obtained by mirroring the right image, estimating the disparity, and then mirroring the result back to get the correct
                            right disparity.
                            Specifically, we use a deep Siamese [5] network that
                            learns to predict a disparity map both from the left image
                            and the flipped right image. By using a Siamese architecture, we learn to predict each disparity map using its corresponding image. By mirroring the right image, prediction
                            of both left and right disparity maps becomes equivalent.
                            We can therefore train both Siamese networks using shared
                            weights. These shared weights have the dual advantage of
                            reducing the computational cost of training and, as evident
                            by our results, resulting in improved networks. A high level
                            overview of our approach is illustrated in Fig. 1.
                            We evaluate our proposed system on the KITTI [13] and
                            Make3D [43] benchmarks and show that, remarkably, in
                            some cases our self-supervised approach outperforms even
                            supervised methods. Importantly, despite the simplicity of
                            our proposed approach and the improved results it offers,
                            we are unaware of previous reports of methods which exploit the symmetry of stereo training in the same manner as
                            we propose to do.
                            To summarize we provide the following contributions:
                            • A novel approach for self-supervised learning of depth
                            (disparity) estimation which trains on pairs of stereo
                            images simultaneously and symmetrically.
                            • We show how a network trained on stereo images can
                            naturally be used for monocular depth estimation at
                            test time.
                            • We report state-of-the-art, monocular disparity estimation results which, in some cases, even outperform supervised systems."""
    }
]
