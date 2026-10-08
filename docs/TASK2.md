# SWYNEX Task 2 — Model or API Integration

## Objective

Build a small working prototype that integrates an AI/ML model into the Student Feedback Classification problem defined in Task 1.

## Integrated model

The prototype uses two scikit-learn components:

1. TfidfVectorizer converts short feedback text into numerical features.
2. LogisticRegression learns the relationship between those features and the five feedback categories.

They are combined in one scikit-learn Pipeline so preprocessing and inference remain consistent.

## Input and output

Input: one short English-language feedback message.

Output:
- predicted category
- highest class probability as a confidence score

Categories:
- Positive Feedback
- Negative Feedback
- Question
- Suggestion
- Complaint

## Example inputs

| Input | Expected category |
|---|---|
| The dashboard is easy to use and the instructions are very clear. | Positive Feedback |
| Could you explain how I submit my weekly task? | Question |
| Please add a calendar showing all upcoming deadlines. | Suggestion |
| My assignment was marked late even though I submitted it on time. | Complaint |
| The portal is confusing and keeps loading when I open my tasks. | Negative Feedback |

The same examples are stored in examples/example_outputs.json.

## Dataset

The prototype includes a small synthetic/manual sample dataset in data/sample_feedback.csv. Each row contains text and label. It does not contain real student personal information.

## Evaluation

The script uses a stratified 75/25 train/test split and reports:
- Accuracy
- Macro-F1
- Precision
- Recall
- Per-class support

A fixed random seed is used for reproducibility.

## Constraints and responsible use

- English short text is the initial scope.
- The sample dataset is small and is not suitable for production deployment.
- Ambiguous feedback can be misclassified.
- Confidence is not a guarantee of correctness.
- Human review remains necessary for important decisions.
- No sensitive personal attributes are inferred.

## Security

No API keys or secrets are needed. Dependencies are public Python packages and the included dataset is demonstration data.

## Run

From the repository root:

    pip install -r requirements.txt
    python src/feedback_classifier.py

For custom text:

    python src/feedback_classifier.py --text "Can I change my task submission?"

## Learning outcome

This task demonstrates practical model integration: raw feedback is transformed into model features, classified into a defined label set, and returned with an interpretable confidence value.

## Next step

A future version can use a larger reviewed dataset, compare multiple models, expose the classifier through a REST API, and add monitoring for data drift.
