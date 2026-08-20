# ML_DL

Research and experiment notebooks for machine learning, deep learning, computer vision, explainable AI, adversarial robustness, and literature-survey analytics.

This repository is mainly a collection of Jupyter/Colab/Kaggle notebooks. Most experiments focus on potato leaf disease classification, transfer learning, Grad-CAM/XAI, uncertainty-aware three-way analysis, cross-dataset transfer, agricultural domain shift, and adversarial robustness for semantic segmentation.

## Published And Related Work

- **CRAT adversarial robustness work**: the CRAT notebooks in this repository are associated with a paper published in **ACIVS 2026**.
- **Trusting CNN Predictions through Uncertainty-Aware Three-Way Analysis**: these notebooks explore explainable and uncertainty-aware three-way analysis for CNN-based plant disease recognition.
- **What Evidence Do Vision Models Use for Plant Disease Recognition Under Agricultural Domain Shift?**: these notebooks study model evidence, attribution behavior, and cross-domain robustness under agricultural domain shift.
- **Literature survey experiment**: [`LiteratureSurvey_Model__Experiment.ipynb`](LiteratureSurvey_Model__Experiment.ipynb) is associated with the IEEE publication: [IEEE Xplore document 11393808](https://ieeexplore.ieee.org/document/11393808/).

## Repository Map

| Group | Main notebooks | Purpose |
| --- | --- | --- |
| CRAT / adversarial robustness | `CRAT_Adversarial_Attack!.ipynb`, `crat-adversarial-attack_V2.0.ipynb`, `crat_adversarial_attack_V2_0.ipynb`, `RRS_Final.ipynb` | Semantic segmentation robustness experiments on Cityscapes using DeepLabV3-ResNet50, FGSM attacks, adversarial training, region-aware evaluation, Grad-CAM visualization, and supporting CRAT data preparation. |
| Literature survey analytics | `LiteratureSurvey_Model__Experiment.ipynb` | Cleans literature-survey data, tags CNN-related papers, extracts model/dataset/metric mentions, and produces conference/year/category visualizations. |
| Potato disease classification | `AI Model(Computer_Vision_Model).ipynb`, `Custom CNN.ipynb`, `caiac-efficientnet-b0.ipynb`, `resnet-50.ipynb` | Baseline and transfer-learning classifiers for potato disease images using custom CNNs, EfficientNetB0, MobileNetV2, ResNet50, Grad-CAM, and evaluation reports. |
| Trusting CNN Predictions through Uncertainty-Aware Three-Way Analysis | `icdm-*` notebooks | Multi-seed, 3-class/7-class potato disease experiments with custom CNN, EfficientNetB0, MobileNetV2, Grad-CAM, uncertainty-aware three-way decision regions, and cross-dataset transfer/ablation studies. |
| What Evidence Do Vision Models Use for Plant Disease Recognition Under Agricultural Domain Shift? | `icmla-stage-1-small-cnns.ipynb`, `aciids-resnet.ipynb`, `aciids-stage-2b-vit.ipynb` | Larger Kaggle pipeline experiments using PlantVillage, PLD Pakistan, Irish Potato, and PlantDoc datasets with classical CNNs, ResNet50, ViT, DBSCAN/evidence-region reporting, attribution evidence, and output packaging. |
| Learning and utility notebooks | `Feed_Forward_Algorithm*.ipynb` | Introductory neural-network training examples and small data-processing utilities. |

## Notebook Index

### CRAT / Adversarial Robustness

| Notebook | What it does |
| --- | --- |
| [`CRAT_Adversarial_Attack!.ipynb`](CRAT_Adversarial_Attack!.ipynb) | Original CRAT experiment notebook. Installs Cityscapes tools, builds a `CityscapesDataset`, loads DeepLabV3-ResNet50, implements FGSM attacks for segmentation, trains/evaluates clean and adversarial models, logs mIoU-style results, and generates Grad-CAM comparisons between clean and adversarial inputs. |
| [`crat-adversarial-attack_V2.0.ipynb`](crat-adversarial-attack_V2.0.ipynb) | Updated CRAT notebook with the same core segmentation robustness pipeline plus saved best-model checkpoints, region-aware adversarial evaluation, visualization helpers, and clean-versus-adversarial Grad-CAM analysis. |
| [`crat_adversarial_attack_V2_0.ipynb`](crat_adversarial_attack_V2_0.ipynb) | Duplicate/alternate filename version of the CRAT V2.0 notebook. It contains the same DeepLabV3-ResNet50, BlurPool, FGSM, adversarial training, evaluation, and visualization workflow. |
| [`RRS_Final.ipynb`](RRS_Final.ipynb) | Utility/data-preparation notebook used in the accepted CRAT project. It supports the CRAT workflow rather than serving as a standalone model-training experiment. |

### Literature Survey Analytics

| Notebook | What it does |
| --- | --- |
| [`LiteratureSurvey_Model__Experiment.ipynb`](LiteratureSurvey_Model__Experiment.ipynb) | Processes a CNN literature-survey spreadsheet. It cleans titles/abstracts, filters CNN-related papers, tags application areas, categories, and work type, extracts model/dataset/metric mentions, then creates bar charts, summary tables, heatmaps, keyword frequencies, co-occurrence heatmaps, and word clouds. Associated IEEE publication: [11393808](https://ieeexplore.ieee.org/document/11393808/). |

### Potato Disease Classification And XAI

| Notebook | What it does |
| --- | --- |
| [`AI Model(Computer_Vision_Model).ipynb`](AI%20Model(Computer_Vision_Model).ipynb) | End-to-end potato disease image classifier using the Kaggle potato dataset. It loads images, trains a custom CNN, evaluates with accuracy/loss curves, classification reports and confusion matrices, saves models, predicts new samples, and applies Grad-CAM/LIME plus three-way KMeans clustering and leaf masking for visual explanation. |
| [`Custom CNN.ipynb`](Custom%20CNN.ipynb) | Compares a lightweight custom CNN with transfer-learning models such as MobileNetV2, EfficientNetB0, and VGG16. It trains on potato disease images, reports accuracy/loss, produces classification reports, and includes Grad-CAM visualization helpers. |
| [`caiac-efficientnet-b0.ipynb`](caiac-efficientnet-b0.ipynb) | EfficientNetB0 training experiment for potato disease classification. It downloads/loads the Kaggle potato dataset, prepares train/validation/test splits, applies class weighting, trains EfficientNetB0, saves training history and model artifacts, and evaluates with confusion matrix and classification report. |
| [`resnet-50.ipynb`](resnet-50.ipynb) | PyTorch/Kaggle ResNet50 experiment. It detects the ImageFolder root, creates train/validation/test transforms and splits, builds a ResNet50 classifier, trains/evaluates with loss and accuracy, and produces confusion matrix/classification-report style outputs. |
| [`new_experiment_custom_light_weight_CNN_XAI_Three-way_Clustering.ipynb`](new_experiment_custom_light_weight_CNN_XAI_Three-way_Clustering.ipynb) | Custom lightweight CNN experiment with explainability. It trains a potato disease classifier, generates Grad-CAM for correct and misclassified samples, applies three-way clustering on Grad-CAM heatmaps, creates leaf masks, and visualizes combined prediction, heatmap, clustering, and masked evidence panels. |

### Trusting CNN Predictions through Uncertainty-Aware Three-Way Analysis

| Notebook | What it does |
| --- | --- |
| [`icdm-plantvillage-3class.ipynb`](icdm-plantvillage-3class.ipynb) | Custom lightweight CNN for 3-class PlantVillage/potato disease classification. It loads the dataset, trains across a controlled seed setup, counts model parameters, and reports accuracy, loss, confusion matrix, precision, recall, and F1. |
| [`icdm-plant-village-7-class.ipynb`](icdm-plant-village-7-class.ipynb) | Custom CNN for a 7-class potato leaf disease dataset. It trains and evaluates the model, generates Grad-CAM visual explanations, assigns three-way evidence regions, and exports reports/figures. |
| [`icdm-efficientnet-b0.ipynb`](icdm-efficientnet-b0.ipynb) | Early EfficientNetB0 baseline for potato disease classification using the Kaggle potato dataset, with training curves, saved artifacts, confusion matrix, and classification report. |
| [`icdm-efficientnet-b0-with-class-3.ipynb`](icdm-efficientnet-b0-with-class-3.ipynb) | EfficientNetB0 3-class experiment with parameter counting, seed-controlled training, class weighting, and evaluation using accuracy, precision, recall, F1, confusion matrix, and classification report. |
| [`icdm-efficientnet-b0-with-class-7.ipynb`](icdm-efficientnet-b0-with-class-7.ipynb) | EfficientNetB0 7-class experiment on the potato leaf disease dataset. It follows the same training/evaluation pattern as the 3-class notebook but targets the broader 7-class label space. |
| [`icdm-7-class-efficient-net.ipynb`](icdm-7-class-efficient-net.ipynb) | Extended 7-class EfficientNetB0 experiment. In addition to model training and metrics, it loads images for Grad-CAM, finds convolution layers, overlays heatmaps, assigns three-way explanation regions, and exports results. |
| [`icdm-mobilenet-with-3-class.ipynb`](icdm-mobilenet-with-3-class.ipynb) | MobileNetV2 3-class potato disease experiment. It trains/evaluates MobileNetV2, reports standard classification metrics, generates Grad-CAM heatmaps, applies three-way evidence-region assignment, and exports artifacts. |
| [`icdm-class-7-mobilenetv2.ipynb`](icdm-class-7-mobilenetv2.ipynb) | MobileNetV2 7-class experiment with the same seed-controlled training/evaluation structure plus Grad-CAM and three-way region analysis. |
| [`icdm-plant-village-3-class-efficient-net.ipynb`](icdm-plant-village-3-class-efficient-net.ipynb) | EfficientNetB0 3-class PlantVillage experiment with recursive convolution-layer discovery for Grad-CAM, overlay visualization, three-way region assignment, and exported models/reports. |
| [`icdm-cross-data-transfer-ablation-study.ipynb`](icdm-cross-data-transfer-ablation-study.ipynb) | Cross-dataset transfer and ablation study. It prepares 3-class and 7-class datasets, trains/evaluates custom CNN, MobileNetV2, and EfficientNetB0, compares transfer behavior between datasets, generates prediction CSVs, and packages ablation outputs. |

### What Evidence Do Vision Models Use for Plant Disease Recognition Under Agricultural Domain Shift?

| Notebook | What it does |
| --- | --- |
| [`icmla-stage-1-small-cnns.ipynb`](icmla-stage-1-small-cnns.ipynb) | Stage-1 Kaggle pipeline for small CNN experiments. It prepares standardized dataset folders from PlantVillage, PLD Pakistan, Irish Potato, and PlantDoc, runs the external `icmla26_code` pipeline, and reports dataset summaries, main performance, few-shot adaptation, attribution evidence, lesion/background perturbation, representation geometry, efficiency, statistical significance, and DBSCAN evidence-region tables. |
| [`aciids-resnet.ipynb`](aciids-resnet.ipynb) | Stage-2 ResNet50 run using the same prepared cross-domain potato disease datasets and external pipeline code. It builds the Kaggle config, runs the pipeline for ResNet50, and packages outputs for download. |
| [`aciids-stage-2b-vit.ipynb`](aciids-stage-2b-vit.ipynb) | Stage-2B ViT/foundation-model run. It prepares the same dataset family, patches/uses the pipeline code, executes ViT-related experiments, inspects output folders, and packages results. |

### Learning / Utility Notebooks

| Notebook | What it does |
| --- | --- |
| [`Feed_Forward_Algorithm.ipynb`](Feed_Forward_Algorithm.ipynb) | Introductory feed-forward neural-network notebook. It loads tabular data, splits train/test data, builds a Keras `Sequential` model, trains it, evaluates accuracy/loss, makes predictions, and plots training curves. |
| [`Feed_Forward_Algorithm_(FFNN).ipynb`](Feed_Forward_Algorithm_(FFNN).ipynb) | Similar FFNN learning notebook with additional reporting, including confusion matrix and classification report visualizations. |

## Common Methods And Tools

- **Languages/frameworks**: Python, Jupyter, Google Colab, Kaggle notebooks
- **Deep learning**: TensorFlow/Keras, PyTorch, torchvision
- **Computer vision models**: custom CNN, EfficientNetB0, MobileNetV2, ResNet50, ViT, DeepLabV3-ResNet50
- **Explainability**: Grad-CAM, LIME, heatmap overlays, leaf masking, three-way clustering/evidence regions
- **Evaluation**: accuracy, loss curves, precision, recall, F1-score, classification report, confusion matrix, cross-dataset transfer, ablation study, adversarial robustness
- **Datasets referenced in notebooks**: potato disease datasets from Kaggle, PlantVillage, PLD Pakistan, Irish Potato, PlantDoc, and Cityscapes for segmentation robustness
