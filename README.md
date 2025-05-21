# 📊 Stack Overflow in the Age of AI: Sentiment, User Engagement, and Language Trends

![Final Project Poster](docs/final_poster.svg)

This project analyzes how large language models (LLMs) like ChatGPT have impacted Stack Overflow — from posting volume to sentiment to programming language trends. We use a mix of statistical testing, NLP, and data visualization to understand how developer help-seeking behavior has shifted in the AI era.

---

## 📁 Project Structure

📁 scripts/ → Python analysis scripts for each hypothesis<br/>
📁 sql/ → SQL queries used to filter and transform raw data<br/>
📁 notebooks/ → Jupyter notebook summarizing exploratory work<br/>
📁 figures/ → Visual outputs from each analysis<br/>
📁 docs/ → Abstract, data spec, tech report, analysis, and final poster<br/>
📁 preprocessing/ → Utility scripts for formatting datasets

---

## 🧪 Methodology Overview

We used four Kaggle datasets (2008–2024) to assess Stack Overflow usage before and after the release of ChatGPT (Nov 2022). Our three main analyses:

1. **Sentiment Analysis**  
   Using HuggingFace’s DistilBERT, we compared frustration levels in questions involving high-level vs. low-level programming languages.

2. **AI-Related Content Trends**  
   Keyword detection showed an 11.6% increase in AI-related posts after ChatGPT's release (p < 0.001).

3. **Usage Decline**  
   Regression analysis demonstrated a statistically significant drop in posting frequency post-ChatGPT, aligning with external reports of a ~50% traffic drop.

---

## 🚀 How to Run the Code

> Requires: `Python 3.8+`, `pandas`, `numpy`, `scikit-learn`, `transformers`, `matplotlib`

You can run each script individually. For example:

python scripts/hypothesis2.py

To reproduce sentiment analysis:
python scripts/hypothesis1/hypothesis1.py

Jupyter notebook for interactive exploration:
jupyter notebook notebooks/analyze.ipynb


## 📂 Data Access
⚠️ Due to file size and privacy concerns, datasets are not tracked in Git.

We used public datasets from Kaggle. You can recreate the database by downloading:

Stack Overflow Questions 2008–2022

60k Stack Overflow Questions with Ratings

Most Popular Programming Languages

Then use the preprocessing scripts in preprocessing/ to prepare the data.

## 📜 Documentation
docs/final_abstract.pdf: Summary of our goals and findings

docs/mid-term_analysis.pdf: Technical deep dive into methods and results

docs/final_poster.pdf: Visual summary for academic presentation

docs/visualizations_overview.pdf: Collection of key plots

## 🧠 Reflection
This project offers a data-driven look into how AI tools like ChatGPT are reshaping online technical communities. We found clear evidence of reduced engagement, increased AI-related activity, and an evolving developer support landscape — all in under two years.

## 👩‍💻 Authors
Nina Py Brozovich<br/>
Charles Clynes<br/>
Colin Pascual<br/>
Andrew Mao
