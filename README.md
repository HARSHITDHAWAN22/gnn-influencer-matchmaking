# AI Brand-Influencer Matchmaking using Heterogeneous Graph Neural Networks

## Overview

Finding the right influencer for a marketing campaign is not as simple as comparing follower counts or engagement rates. A successful collaboration depends on multiple factors such as budget, audience demographics, content category, language preference, engagement quality, and campaign objectives. Traditional filtering methods often fail to capture these relationships effectively.

This project approaches the problem as a recommendation task by representing brands and influencers as a heterogeneous graph, where both entities and their relationships are learned together instead of being evaluated independently. Along with numerical features, semantic information from profiles and descriptions is converted into vector embeddings to capture similarities that are difficult to represent using structured data alone.

The recommendation model is built using a Graph Neural Network based on GraphSAGE, allowing information to propagate through the graph and learn meaningful representations for both brands and influencers. To understand whether these graph-based embeddings can also benefit traditional machine learning models, the learned representations are further integrated with XGBoost and LightGBM, and their performance is compared against the standalone GNN.

The complete workflow covers data preprocessing, feature engineering, graph construction, model training, recommendation generation, detailed evaluation, and comparative analysis. The objective of the project is not only to recommend suitable influencer-brand collaborations but also to study how graph-based learning compares with hybrid machine learning approaches for this recommendation problem.


## Features

## Project Workflow

## Project Structure

## Installation

## Usage

## Results

## Future Improvements

## Technologies Used

## Author
