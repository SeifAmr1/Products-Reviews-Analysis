

## IMDB Sentiment Analysis with LSTM & Optuna Hyperparameter Optimization

This project builds a **sentiment analysis model** using **LSTM (Long Short-Term Memory)** neural networks to classify movie reviews from the IMDB dataset as **positive** or **negative**. To achieve optimal model performance, **Optuna** is used for **automated hyperparameter tuning**. The model is also tested on a set of **custom reviews**, and predictions are saved to a CSV file.

---

### Key Steps:

#### 1. **Data Loading & Preprocessing**

* IMDB dataset loaded in chunks using `pandas` to handle large file size.
* Reviews tokenized with `Tokenizer` (limited to top 5000 words).
* Text converted to padded sequences of length 200.
* Sentiments encoded (`positive` = 1, `negative` = 0).

#### 2. **Model Definition & Tuning with Optuna**

* Built a **Keras Sequential model** with:

  * `Embedding` layer
  * One **LSTM** layer (units tuned)
  * `Dropout` layer (rate tuned)
  * Final `Dense` layer with sigmoid activation
* Optuna optimized:

  * `units` in LSTM (50–100)
  * `dropout_rate` (0.2–0.5)
  * `optimizer`: `adam` or `rmsprop`
* Objective: maximize validation accuracy on the test set

#### 3. **Final Model Training & Evaluation**

* Trained the best LSTM model found by Optuna for 5 epochs.
* Evaluated on the test set and reported final **test accuracy**.

#### 4. **Custom Review Predictions**

* A list of 10 diverse custom reviews was created (positive and negative).
* Preprocessed using the same tokenizer and padding.
* Model predicted sentiment for each review.
* Results were saved in `custom_reviews_with_predictions.csv`.

---

### Output Includes:

* Best hyperparameters from Optuna
* Final model accuracy on unseen test data
* Sentiment predictions for custom reviews

---



### 📌 Dependencies:

* `pandas`
* `scikit-learn`
* `tensorflow` / `keras`
* `optuna`

---

### 🔧 Notes:

* Uses only 2 Optuna trials for quick demonstration — **increase `n_trials`** for better tuning in production.
* LSTM is ideal for handling sequential data like text due to its ability to retain contextual memory.
* Can be extended to support multi-class sentiment classification or deployed in real-time apps.


