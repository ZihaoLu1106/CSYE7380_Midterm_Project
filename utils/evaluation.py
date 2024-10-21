from nltk.translate.bleu_score import sentence_bleu

def evaluate_bleu(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    return sentence_bleu(reference, candidate)
