# Vietnamese Sports News Classification & Recommendation System

A Natural Language Processing project for classifying Vietnamese sports news articles and recommending related articles based on textual similarity. The project focuses on Vietnamese text preprocessing, multi-class news classification, and content-based recommendation.

> Team project – NLP at HUST
> Members: Nguyễn Diệu My, Hà Huyền Thu, Hồng Minh Khang,

---

## Project Overview

This project solves two main NLP tasks using Vietnamese sports news data collected from VnExpress:

1. **Sports News Classification**  
   Classify Vietnamese sports articles into 6 categories:
   - Bóng đá
   - Cờ vua
   - Golf
   - Quần vợt
   - Điền kinh
   - Đua xe

2. **Article Recommendation System**  
   Recommend the top related articles based on the input article content using TF-IDF and Cosine Similarity.

---

## Key Results

### Classification Model Performance

The classification model uses **Bi-LSTM + Custom Attention** and achieved strong validation performance:

| Metric | Score |
|---|---:|
| Validation Accuracy | **98.88%** |
| Macro Precision | **98.88%** |
| Macro Recall | **98.86%** |
| Macro F1-score | **98.86%** |
| Weighted F1-score | **98.87%** |

### Per-class Performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Bóng đá | 0.9828 | 0.9744 | 0.9785 | 117 |
| Cờ vua | 1.0000 | 1.0000 | 1.0000 | 120 |
| Golf | 0.9836 | 1.0000 | 0.9917 | 120 |
| Quần vợt | 0.9917 | 1.0000 | 0.9959 | 120 |
| Điền kinh | 0.9832 | 0.9915 | 0.9873 | 118 |
| Đua xe | 0.9912 | 0.9658 | 0.9784 | 117 |

The model reached its best validation loss at epoch 4 and used EarlyStopping to prevent overfitting.

---

## Dataset

### Classification Dataset

- Source: Vietnamese sports articles from VnExpress
- Number of original records: **3,574 articles**
- Number of usable records after cleaning missing content: **3,558 articles**
- Number of classes: **6**
- Input text: `title + description + content`

Class distribution is relatively balanced:

| Category | Number of Articles |
|---|---:|
| Golf | 600 |
| Quần vợt | 599 |
| Cờ vua | 599 |
| Đua xe | 599 |
| Điền kinh | 592 |
| Bóng đá | 585 |

### Recommendation Dataset

- Number of articles: **3,742**
- Fields used: `title`, `description`, `content`, `standard_tags`, `url`
- TF-IDF feature size: **10,000 features**

---

## Methodology

### 1. Vietnamese Text Preprocessing

The project applies a complete Vietnamese text preprocessing pipeline:

- Unicode normalization using `unicodedata`
- Lowercasing
- URL removal
- Punctuation and special character removal
- Number removal
- Vietnamese word segmentation using `ViTokenizer`
- Stopword removal
- Sports abbreviation normalization, for example:
  - `hlv` → `huấn luyện viên`
  - `vđv` → `vận động viên`
  - `clb` → `câu lạc bộ`
  - `mu` → `manchester united`

---

## Model Architecture

### Bi-LSTM + Custom Attention Classifier

The classification model is built from scratch using TensorFlow/Keras Functional API.

Architecture:

```text
Input text
   ↓
Text cleaning + Vietnamese tokenization
   ↓
Keras Tokenizer + Padding, max length = 250
   ↓
Embedding layer, dimension = 200
   ↓
Bidirectional LSTM, 64 units each direction
   ↓
Custom Attention Layer
   ↓
Dropout, rate = 0.5
   ↓
Dense Softmax Layer, 6 classes
```

Main training setup:

| Component | Value |
|---|---|
| Vocabulary size | 30,000 max tokens |
| Max sequence length | 250 |
| Embedding dimension | 200 |
| Bi-LSTM units | 64 per direction |
| Dropout | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Sparse Categorical Crossentropy |
| Batch size | 64 |
| Epochs | Up to 10 |
| Callback | EarlyStopping + ModelCheckpoint |

---

## Recommendation System

The recommendation module is a content-based recommendation system.

Pipeline:

```text
Input article
   ↓
Vietnamese text cleaning
   ↓
TF-IDF vectorization, unigram + bigram
   ↓
Cosine Similarity with all articles in database
   ↓
Top-N similar articles
   ↓
Suggested tags from most frequent tags among similar articles
```

Main features:

- Recommends the **Top 5 most related articles**
- Returns article title, URL, tags, and similarity score
- Suggests relevant tags based on similar articles
- Uses `TfidfVectorizer` with `ngram_range=(1, 2)` and `max_features=10000`

Example recommendation output:

```text
Input topic: Công Phượng trở lại thi đấu cho Đồng Nai

Suggested tags:
nguyễn công phượng, đồng nai, lương xuân trường, bình phước, hoà bình

Top related article:
Công Phượng, Xuân Trường hứa đưa Đồng Nai lên V-League
Similarity: 58.15%
```

---

## Tech Stack

- **Language:** Python
- **Notebook:** Google Colab / Jupyter Notebook
- **NLP:** PyVi, ViTokenizer, Regex, Unicode normalization
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **Recommendation:** TF-IDF, Cosine Similarity
- **Visualization:** Matplotlib, Seaborn
- **Data Handling:** Pandas, NumPy

---

## Project Highlights

- Built an end-to-end Vietnamese NLP pipeline for real-world news classification.
- Designed a custom Attention layer instead of relying only on built-in Keras layers.
- Achieved **98.88% validation accuracy** and **98.86% macro F1-score** on a 6-class classification task.
- Implemented a content-based recommendation system using TF-IDF and Cosine Similarity.
- Combined classification and recommendation into one practical news analysis workflow.
- Handled Vietnamese-specific preprocessing challenges such as word segmentation, abbreviation normalization, and Unicode normalization.

---

## Limitations

- The classification model was evaluated on a validation split from the collected dataset; a larger external test set would provide a stronger generalization check.
- The recommendation system is content-based, so it does not use user behavior data such as clicks, reading history, or ratings.
- The model is trained from scratch and does not use pretrained Vietnamese language models such as PhoBERT.

---

## Future Improvements

- Fine-tune Vietnamese pretrained models such as PhoBERT or viBERT for classification.
- Add an independent test set from a different time period or source.
- Build a hybrid recommendation system combining content similarity and user interaction data.
- Deploy the model as a simple web demo using Streamlit or FastAPI.
- Add explainability by visualizing important words from the Attention layer.

---

## How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn pyvi chardet
```

2. Open the notebook:

```bash
jupyter notebook NLP_NewsClassificationAndRecommendation_Group2_2025_1.ipynb
```

3. Run all cells to:

- Load and clean the dataset
- Train the Bi-LSTM + Attention classifier
- Evaluate the model
- Run the recommendation function


---

## CV Summary

Built a Vietnamese sports news classification and recommendation system using TensorFlow/Keras, Bi-LSTM with Custom Attention, TF-IDF, and Cosine Similarity. The classifier achieved **98.88% validation accuracy** and **98.86% macro F1-score** across 6 VnExpress sports categories, while the recommendation module suggests related articles and tags from a 3,742-article database.
