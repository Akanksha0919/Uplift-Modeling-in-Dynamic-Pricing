# Uplift Modeling in Dynamic Pricing: Effect of Optimal Pricing strategy on Purchase Probability using Continuous DID (Quasi Experimental approach)

Type: Master's Thesis 

Author: Akanksha Saxena

1st Examiner: Prof. Dr. Stefan Lessmann

2nd Examiner: Prof. Dr. Benjamin Fabian

Master Thesis submitted to the School of Business and Economics of Humboldt-Universität zu Berlin for the degree M.Sc. Economics and Management Science.

![image](https://github.com/Akanksha0919/Uplift-Modeling-in-Dynamic-Pricing/assets/65400521/812aa420-168d-4155-b1c6-6f818f742829)
This image shows the result of causal analysis of optimal pricing strategy on consumer purchase probability

## Table of Content

## Summary 

Adopting the dataset from a Brazilian e-commerce company containing information on daily product demand and consumer purchase behavior, this paper utilizes the continuous difference-in-difference estimator (DID) model to incorporate consumer purchase behavior in dynamic pricing model development. This paper proposes a novel approach to a dynamic pricing algorithm that comprises the following components: First, the dataset is divided into two groups one is the control group which contains the base price, and the other is the Treatment group which will contain optimal prices recommended from the dynamic pricing algorithm. Next, a dynamic pricing model is implemented to derive optimal prices for the treatment group. Finally, the causal impact of optimal prices on consumer purchase probability is examined for before and after dynamic pricing intervention. The empirical results that emerged from the incorporation of continuous DID in a dynamic pricing framework show that dynamic pricing intervention has a statistically significant and positive effect on consumer purchase probability. The results show that implementing dynamic pricing on elastic products is likely to increase the consumer's purchase probability, whereas no effect is accumulated for inelastic products. This paper implies that incorporating the causal analysis in dynamic pricing strategy based on historical data can provide significant improvements in pricing strategy by mitigating risks of negative consumer reaction to variation in and proximity of dynamic pricing.

Keywords: Dynamic Pricing, Consumer Purchase Probability,Continous Treatment Effect,Continous Difference-in-Difference,Quasi=Experimental approach

## Working with Repo

### Dependencies
Python 3.9.12

### Setup
1. Clone this repository
2. Change the path of datasets downloaded from the repo in notebooks. Data is read in notebooks using "pd.read_csv"
3. Change the path of intermediate datasets generated by each notebook. Intermediate data is stored in csv format using "pd.to_csv"

## Reproducing results

### Usage 
When executing the Python files, the following is the correct order:
1. Part 1- Olist Product data preparation - This notebook requires following datasets: olist_products_dataset,olist_order_items_dataset, and olist_orders_dataset. The output of this notebook is "product_charateristics_df_developed_in_easy_way" which is input for part 2 Notebook
2. Part 2-Dynamic Pricing Estimation model: The input is "product_charateristics_df_developed_in_easy_way" obtained from Part 1. The output of this file: post_pre_intervention_df
3. Part 3- Olist Consumer data Preparation : The data necessary for this notebook is : post_pre_intervention_df(obtained from PART 2), SIDRA datasets,olist_orders_dataset, olist_order_items_dataset,and olist_customers_dataset. The output file is : consumer_df_final which is used in Part 4
4. Part 4-Consumer Purchase Probability Estimation: the only input needed in this scenario is : consumer_df_final obtained from part 3. This notebooks has 5 output files: purhcase_proba_all_products_resampled, purhcase_proba_elastic_products, purhcase_proba_veblen_products, purhcase_proba_neg_PED_elastic_prods_products and purhcase_proba_inelastic_products
5. Part 5-Causal analysis with Continous DID estimator using TWFE: the input to this file is the output from Part 4 and product_category_name_translation. this notebook gives final estimation results

Part 1 -->> Part 2 -->> Part 3 -->> Part 4 -->> Part 5 (Please follow this order while running the notebooks)

### Data Preparation code:
1. Part 1- Olist Product data preparation: Olist datasets are aggregated at product level to get demands for each product - each day
2. Part 4-Consumer Purchase Probability Estimation: Datasets from Olist, SIDRA and optimal pricing strategy are integrated for further use in purchase probability estiamtion


### Training code
Following are the notebooks containing model training for different concepts:
1. Part 2-Dynamic Pricing Estimation model:
   1. Price Elasticity of Demand: Logistic Regression
   2. Product Grouping: K-means clustering
   3. Dynamic Pricing:
      1. Random Forest
      2. Linear Regression
      3. Lasso Regression
      4. Ridge Regression
  4. Linear Programming optimization
This notebook contains Data engineering for optimal pricing model.
2. Part 4-Consumer Purchase Probability Estimation:
   1. Logistic Regression
   2. Adaboost
  Hyperparamter tuning is performed using GridsearchCV. All training notebooks have model evaluations in the same notebook

### Result Code:
"Part 5-Causal analysis with Continous DID estimator using TWFE" contains results of continous DID.

## Project Structure
