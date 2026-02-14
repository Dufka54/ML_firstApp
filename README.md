Project Overview: Student Placement Predictor
I developed a custom Batch Gradient Descent Linear Regression model from scratch using numpy to predict student placement outcomes. 
The model was trained on a dataset sourced from Kaggle, which includes features such as the number of internships, CGPA, and history of backlogs.

Methodology
- Model Architecture: A custom-built Linear Regression class was implemented where gradients are computed using the entire dataset ($X$ and $y$) simultaneously during each iteration.
- Data Partitioning: To ensure robust evaluation, the original dataset was divided into three distinct subsets: Training (35%), Validation (35%), and Testing (30%).
- Hyperparameter Tuning: I conducted a grid search across various combinations of learning rates and epochs (iterations). Accuracy scores were calculated for each pair to identify the most effective configuration.
- Decision Logic: Since the task is a binary classification (Placed vs. Not Placed), a threshold of 0.5 was applied to the continuous output; values exceeding this threshold are classified as 1 (Placed), while all others are classified as 0 (Not Placed).

Results
- Performace: After optimizing the hyperparameters, the model achieved a peak test accuracy of approximately 79.25%.
- Final Implementation: The best-performing model (utilizing a learning rate of 0.01 and 1,000 iterations) was selected for final predictions and exported as a pickle file for deployment.
