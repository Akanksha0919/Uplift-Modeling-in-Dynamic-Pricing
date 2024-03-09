# Uplift Modeling in Dynamic Pricing: Effect of Optimal Pricing strategy on Purchase Probability using Continuous DID (Quasi Experimental approach)

Type: Master's Thesis 

Author: Akanksha Saxena

1st Examiner: Prof. Dr. Stefan Lessmann

2nd Examiner: Prof. Dr. Benjamin Fabian

![image](https://github.com/Akanksha0919/Uplift-Modeling-in-Dynamic-Pricing/assets/65400521/812aa420-168d-4155-b1c6-6f818f742829)

Master Thesis submitted to the School of Business and Economics of Humboldt-Universität zu Berlin for the degree M.Sc. Economics and Management Science.

## Abstract

Adopting the dataset from a Brazilian e-commerce company containing information on daily product demand and consumer purchase behavior, this paper utilizes the continuous difference-in-difference estimator (DID) model to incorporate consumer purchase behavior in dynamic pricing model development. This paper proposes a novel approach to a dynamic pricing algorithm that comprises the following components: First, the dataset is divided into two groups one is the control group which contains the base price, and the other is the Treatment group which will contain optimal prices recommended from the dynamic pricing algorithm. Next, a dynamic pricing model is implemented to derive optimal prices for the treatment group. Finally, the causal impact of optimal prices on consumer purchase probability is examined for before and after dynamic pricing intervention. The empirical results that emerged from the incorporation of continuous DID in a dynamic pricing framework show that dynamic pricing intervention has a statistically significant and positive effect on consumer purchase probability. The results show that implementing dynamic pricing on elastic products is likely to increase the consumer's purchase probability, whereas no effect is accumulated for inelastic products. This paper implies that incorporating the causal analysis in dynamic pricing strategy based on historical data can provide significant improvements in pricing strategy by mitigating risks of negative consumer reaction to variation in and proximity of dynamic pricing.

When executing the Python files, the following is the correct order:
1. Part 1- Olist Product data preparation - This notebook requires following datasets: olist_products_dataset,olist_order_items_dataset, and olist_orders_dataset. The output of this notebook is "product_charateristics_df_developed_in_easy_way" which is input for part 2 Notebook
2. Part 2-Dynamic Pricing Estimation model: The input is "product_charateristics_df_developed_in_easy_way" obtained from Part 1. The output of this file: post_pre_intervention_df
3. Part 3- Olist Consumer data Preparation : The data necessary for this notebook is : post_pre_intervention_df(obtained from PART 2), SIDRA datasets,olist_orders_dataset, olist_order_items_dataset,and olist_customers_dataset. The output file is : consumer_df_final which is used in Part 4
4. Part 4-Consumer Purchase Probability Estimation: the only input needed in this scenario is : consumer_df_final obtained from part 3. This notebooks has 5 output files: purhcase_proba_all_products_resampled, purhcase_proba_elastic_products, purhcase_proba_veblen_products, purhcase_proba_neg_PED_elastic_prods_products and purhcase_proba_inelastic_products
5. Part 5-Causal analysis with Continous DID estimator using TWFE: the input to this file is the output from Part 4 and product_category_name_translation. this notebook gives final estimation results 
