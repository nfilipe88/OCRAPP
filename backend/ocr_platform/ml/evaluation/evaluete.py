from jiwer import wer

def evaluate(predictions, references):
    error = wer(references, predictions)
    print(f"Word Error Rate: {error}")
