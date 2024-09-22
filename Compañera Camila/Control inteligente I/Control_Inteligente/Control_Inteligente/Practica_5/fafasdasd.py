import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def fitness_function_classification(a, b, c, d, true_labels):
    # Aplicar la función sigmoide a a, b, c, y d
    a = sigmoid(a)
    b = sigmoid(b)
    c = sigmoid(c)
    d = sigmoid(d)

    predicted_labels = [a, b, c, d]
    correct_predictions = sum(round(pred) == true for pred, true in zip(predicted_labels, true_labels))
    total_predictions = len(predicted_labels)
    accuracy = correct_predictions / total_predictions
    print(f"Valor Normalizado: {accuracy}")
    return accuracy

def fitness_function_classification(a,b,c,d, true_labels):
    predicted_labels = [a,b,c,d]
    correct_predictions = sum(pred == true for pred, true in zip(predicted_labels, true_labels))
    total_predictions = len(predicted_labels)
    accuracy = (correct_predictions / total_predictions)
    print(f"Valor Normalizado: {accuracy}")
    return accuracy

def fitness_function_classification(a, b, c, d, true_labels):
    predicted_labels = [a, b, c, d]
    correct_predictions = sum(pred == true for pred, true in zip(predicted_labels, true_labels))
    total_predictions = len(predicted_labels)
    accuracy = correct_predictions / total_predictions
    accuracy_rounded = round(accuracy, 2)  # Redondear a 2 decimales
    print(f"Valor Normalizado: {accuracy_rounded:.2f}")
    return accuracy_rounded

# Ejemplo de uso
a = 1
b = 4
c = 1
d = 0
true_labels = [1, 4, 1, 0]
accuracy = fitness_function_classification(a, b, c, d, true_labels)
print(f"Precisión de la clasificación: {accuracy}")





