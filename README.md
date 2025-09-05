# **CodeSentinel Internship — ArtificialIntelligence🚀**

## **Task 1: Linear Regression Model**
Objective: Predict house prices using numerical features (e.g., area, rooms).
Dataset Used: California Housing Dataset (fetch_california_housing from scikit-learn).
Approach:
Preprocessed features with StandardScaler.
Split data into training (80%) and testing (20%) sets.
Trained a Linear Regression model on scaled data.
Results & Evaluation:
Computed Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.
Visualized predictions vs. actual values using Matplotlib.
Key Insight: Linear Regression provided a simple yet effective baseline for regression tasks.

## **Task 2: Classification Model**
Objective: Predict survival on the Titanic dataset using classification models.
Dataset Used: Titanic dataset.
Approach:
Preprocessed data with missing value imputation, label encoding, and feature scaling.
Trained multiple models:
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Compared accuracy of different models.
Results & Evaluation:
Evaluated using Accuracy Score, Confusion Matrix, and Classification Report.
Logistic Regression and Decision Tree achieved strong performance, with KNN providing comparative insights.
Key Insight: Feature preprocessing significantly improved classification accuracy.

## **Task 3: Neural Network**
Objective: Build and train a deep learning model for handwritten digit classification.
Dataset Used: MNIST (70,000 grayscale images of digits 0–9).
Approach:
Built a Sequential Neural Network with TensorFlow/Keras.
Included hidden layers with ReLU activation, Dropout for regularization, and Softmax output layer.
Trained using Adam optimizer and categorical crossentropy loss.
Applied EarlyStopping to prevent overfitting.
Results & Evaluation:
Achieved high test accuracy on unseen data.
Visualized training history (loss & accuracy curves).
Generated a classification report to measure per-class performance.
Key Insight: Neural networks excel at image classification tasks when combined with regularization techniques.
