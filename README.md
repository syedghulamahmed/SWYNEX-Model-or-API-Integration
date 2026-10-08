# SWYNEX Model or API Integration

## Task 2 — Student Feedback Classification

This repository contains a working prototype for SWYNEX Technologies Task 2. It integrates a machine-learning model into the Student Feedback Classification problem defined in Task 1.

### Integrated model

TF-IDF Vectorizer + Logistic Regression using scikit-learn.

The prototype accepts short student/internship feedback and predicts:
- Positive Feedback
- Negative Feedback
- Question
- Suggestion
- Complaint

No external API key is required.

### Structure

    data/sample_feedback.csv
    docs/TASK2.md
    examples/example_outputs.json
    src/feedback_classifier.py
    requirements.txt

### Run

Install Python 3.10+ and dependencies:

    pip install -r requirements.txt

Run the prototype:

    python src/feedback_classifier.py

Classify custom text:

    python src/feedback_classifier.py --text "Can I change my task submission?"

The script trains on the included demonstration dataset, reports accuracy and macro-F1 on a held-out split, then prints example predictions and confidence scores.

### Example

Input: The dashboard is easy to use and the instructions are very clear.

Expected category: Positive Feedback

Input: Could you explain how I submit my weekly task?

Expected category: Question

### Evaluation

The prototype reports accuracy, macro-F1, precision, recall, and per-class support. The included dataset is intentionally small because this is a prototype; production use would require a larger reviewed dataset.

### Security

No API keys, tokens, passwords, or other secrets are stored in this repository.

See docs/TASK2.md for the complete implementation and evaluation notes.
