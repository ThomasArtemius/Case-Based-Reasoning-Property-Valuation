# Case-Based-Reasoning-Property-Valuation
Predicting price per square meter property using Case-Based Reasoning. Try it [here](https://case-based-reasoning-property-valuation-jup53fgrz76mzpoecenuxe.streamlit.app/)
# About the project
This is just a simple project using entirely on dummy data. We use Gower distance to measure dissimilarity (distance) of new case with the case base
### Gower Distance Formula
$$
D_{\text{Gower}}(x, y) = \frac{\sum_{j=1}^{p} w_j \cdot d_j(x_j, y_j)}{\sum_{j=1}^{p} w_j}
$$

#### Numeric Features
$$
d_j(x_j, y_j) = \frac{|x_j - y_j|}{\text{range}_j}
$$

#### Categorical Features
$$
d_j(x_j, y_j) = \begin{cases} 
0 & \text{if } x_j = y_j, \\
1 & \text{if } x_j \neq y_j.
\end{cases}
$$
