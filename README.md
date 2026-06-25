# AI Brand-Influencer Matchmaking using Heterogeneous Graph Neural Networks

## Overview

Finding the right influencer for a marketing campaign is not as simple as comparing follower counts or engagement rates. A successful collaboration depends on multiple factors such as budget, audience demographics, content category, language preference, engagement quality, and campaign objectives. Traditional filtering methods often fail to capture these relationships effectively.

This project approaches the problem as a recommendation task by representing brands and influencers as a heterogeneous graph, where both entities and their relationships are learned together instead of being evaluated independently. Along with numerical features, semantic information from profiles and descriptions is converted into vector embeddings to capture similarities that are difficult to represent using structured data alone.

The recommendation model is built using a Graph Neural Network based on GraphSAGE, allowing information to propagate through the graph and learn meaningful representations for both brands and influencers. To understand whether these graph-based embeddings can also benefit traditional machine learning models, the learned representations are further integrated with XGBoost and LightGBM, and their performance is compared against the standalone GNN.

The complete workflow covers data preprocessing, feature engineering, graph construction, model training, recommendation generation, detailed evaluation, and comparative analysis. The objective of the project is not only to recommend suitable influencer-brand collaborations but also to study how graph-based learning compares with hybrid machine learning approaches for this recommendation problem.


## Features

* Models the complete brand–influencer ecosystem as a heterogeneous graph with dedicated Brand and Influencer node types and multiple relationship types.
* Combines business metrics such as engagement rate, content quality, ROI, pricing, audience demographics, campaign budget, and growth metrics with semantic profile embeddings.
* Builds compatibility-driven relationships using multiple matching factors instead of relying on simple rule-based filtering.
* Creates similarity connections between brands and influencers to capture shared interests, content categories, and campaign objectives.
* Uses a GraphSAGE-based graph learning pipeline to learn meaningful representations and predict brand–influencer compatibility.
* Implements negative sampling to improve the model's ability to distinguish relevant collaborations from unrelated pairs.
* Includes a complete data preparation pipeline with missing value handling, feature scaling, ROI engineering, Winsorization, Modified Z-Score analysis, and normalization.
* Evaluates model performance using Accuracy, Precision, Recall, F1-Score, AUC-ROC, Average Precision, confusion matrices, ROC curves, and Precision–Recall curves.
* Optimizes prediction thresholds and includes detailed debugging utilities for performance analysis and model validation.
* Benchmarks graph-learned embeddings with hybrid GNN + XGBoost and GNN + LightGBM pipelines for comparative evaluation.
* Generates ranked influencer recommendations for every brand and exports the final recommendations for further analysis and integration.


## Project Workflow

Raw Influencer & Brand Datasets
                │
                ▼
Data Cleaning & Missing Value Handling
                │
                ▼
Feature Engineering
(ROI Engineering, Normalization, Outlier Analysis,
Winsorization, Modified Z-Score)
                │
                ▼
Semantic Text Embeddings
(Sentence Transformers - all-MiniLM-L6-v2)
                │
                ▼
Complete Node Feature Preparation
(Numerical + Categorical + Text Embeddings)
                │
                ▼
Heterogeneous Graph Construction
(Brand ↔ Influencer,
Influencer ↔ Influencer,
Brand ↔ Brand)
                │
                ▼
Compatibility-Based Edge Generation
(Budget, Engagement, Content Quality,
Language & Audience Matching)
                │
                ▼
GraphSAGE Representation Learning
                │
                ▼
Negative Sampling & Model Training
                │
                ▼
Model Evaluation
(AUC, AP, Accuracy, Precision, Recall,
F1-Score, ROC, PR Curve, Threshold Optimization)
                │
                ▼
Hybrid Model Comparison
(GNN vs GNN + XGBoost vs GNN + LightGBM)
                │
                ▼
Ranked Influencer Recommendations
                │
                ▼
Recommendation Export & Performance Reports


## Project Structure

AI-Brand-Influencer-Matchmaking/ │ ├── data/ │ ├── influencers_dataset.csv │ └── brands_dataset.csv │ ├── src/ │ ├── preprocessing.py │ ├── embeddings.py │ ├── features.py │ ├── graph_builder.py │ ├── gnn_model.py │ ├── training.py │ ├── recommendations.py │ ├── evaluation.py │ ├── model_comparison.py │ ├── model_diagnostics.py │ └── advanced_model.py │ ├── images/ │ ├── results/ │ ├── requirements.txt ├── README.md └── .gitignore

## Installation

echnology	Usage in Project
Python	Complete project implementation
Pandas	Dataset loading, preprocessing, feature engineering and data manipulation
NumPy	Numerical operations and feature preparation
PyTorch	Model training, tensor operations and optimization
PyTorch Geometric	Heterogeneous graph creation, GraphSAGE implementation and graph message passing
Sentence Transformers (all-MiniLM-L6-v2)	Converting profile, bio and category information into vector embeddings
Scikit-learn	Feature scaling, label encoding, evaluation metrics and train-validation-test splitting
XGBoost	Hybrid model trained on graph-generated embeddings for comparison
LightGBM	Hybrid gradient boosting model for performance benchmarking
Matplotlib	Training statistics and evaluation visualizations
Seaborn	Correlation analysis, confusion matrices and performance plots

## Usage

## Results

## Future Improvements

## Technologies Used

## Author
