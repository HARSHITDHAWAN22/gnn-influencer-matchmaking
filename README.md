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

## Project Structure

## Installation

## Usage

## Results

## Future Improvements

## Technologies Used

## Author
