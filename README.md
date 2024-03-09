# Uplift Modeling in Dynamic Pricing: Effect of Optimal Pricing strategy on Purchase Probability using Continuous DID (Quasi Experimental approach)

Master Thesis submitted to the School of Business and Economics of Humboldt-Universität zu Berlin for the degree M.Sc. Economics and Management Science.

## Abstract

Adopting the dataset from a Brazilian e-commerce company containing information on daily product demand and consumer purchase behavior, this paper utilizes the continuous difference-in-difference estimator (DID) model to incorporate consumer purchase behavior in dynamic pricing model development. This paper proposes a novel approach to a dynamic pricing algorithm that comprises the following components: First, the dataset is divided into two groups one is the control group which contains the base price, and the other is the Treatment group which will contain optimal prices recommended from the dynamic pricing algorithm. Next, a dynamic pricing model is implemented to derive optimal prices for the treatment group. Finally, the causal impact of optimal prices on consumer purchase probability is examined for before and after dynamic pricing intervention. The empirical results that emerged from the incorporation of continuous DID in a dynamic pricing framework show that dynamic pricing intervention has a statistically significant and positive effect on consumer purchase probability. The results show that implementing dynamic pricing on elastic products is likely to increase the consumer's purchase probability, whereas no effect is accumulated for inelastic products. This paper implies that incorporating the causal analysis in dynamic pricing strategy based on historical data can provide significant improvements in pricing strategy by mitigating risks of negative consumer reaction to variation in and proximity of dynamic pricing.

When executing the Python files, the following is the correct order:
1. Part 1- Olist Product data preparation
2. Part 2-Dynamic Pricing Estimation model
3. Part 3- Olist Consumer data Preparation
4. Part 4-Consumer Purchase Probability Estimation
5. Part 5-Causal analysis with Continous DID estimator using TWFE
