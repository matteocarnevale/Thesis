import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, datasets
from PIL import Image
import csv
import os

# Importa il tuo modello Sequencer2D (assicurati di aver importato tutte le dipendenze)
from sequencer.models.two_dim_sequencer import Sequencer2D

# Carica GloVe Embeddings
def load_glove_embeddings(file_path):
    embeddings_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings_dict[word] = vector
    return embeddings_dict

# Carica il modello Sequencer2D e preparalo per l'estrazione delle feature
def load_custom_feature_extractor(model_path):
    # Inizializza il tuo modello Sequencer2D
    model = Sequencer2D(num_classes=1000)  # Specifica i parametri necessari per il tuo modello

    # Carica i pesi addestrati
    model.load_state_dict(torch.load(model_path))
    
    # Rimuovi l'ultimo layer per estrarre le feature
    # Qui si assume che `with_fc` indichi se è presente un fully connected layer finale
    if model.with_fc:
        model.with_fc = False

    model.eval()  # Imposta il modello in modalità di valutazione
    return model

# Estrai feature da un'immagine
def extract_image_features(feature_extractor, input_tensor):
    with torch.no_grad():
        # Passa l'immagine attraverso il modello per estrarre le feature
        image_features = feature_extractor(input_tensor)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
    return image_features

# Calcola la somiglianza e predice la classe
def predict_class(image_features, class_vectors, unseen_classes):
    class_vectors = torch.tensor(class_vectors, dtype=torch.float32)
    similarity_scores = F.cosine_similarity(image_features, class_vectors)
    predicted_class_idx = torch.argmax(similarity_scores)
    return unseen_classes[predicted_class_idx], similarity_scores[predicted_class_idx].item()

# Salva i risultati in un file CSV
def save_results_to_csv(log_file_path, results):
    # Crea il file CSV con intestazioni se non esiste già
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Image Path', 'Dataset', 'Predicted Class', 'Similarity Score'])

    # Aggiungi i risultati al CSV
    with open(log_file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for result in results:
            writer.writerow([result['image_path'], result['dataset'], result['predicted_class'], result['similarity_score']])

# Main Script
if __name__ == "__main__":
    # Percorso ai dati
    glove_path = 'path/to/glove.6B.300d.txt'
    model_path = 'path/to/sequencer2d_pretrained.pth'
    generic_image_folder = 'path/to/images'  # Cartella contenente immagini generiche
    log_file_path = 'zero_shot_classification_log.csv'

    # Carica GloVe embeddings e ottieni i vettori delle classi
    glove_embeddings = load_glove_embeddings(glove_path)

    # Carica il dataset CIFAR-100
    transform_cifar100 = transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensiona le immagini CIFAR-100 a 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    cifar100_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_cifar100)

    # Ottieni i nomi delle classi di CIFAR-100 e i loro embeddings
    unseen_classes_cifar100 = cifar100_dataset.classes
    class_vectors_cifar100 = []
    for class_name in unseen_classes_cifar100:
        if class_name in glove_embeddings:
            class_vectors_cifar100.append(glove_embeddings[class_name])
        else:
            # Se l'embedding non è disponibile, crea un vettore casuale per evitare errori
            class_vectors_cifar100.append(np.random.rand(300).astype(np.float32))

    class_vectors_cifar100 = np.array(class_vectors_cifar100)

    # Carica il modello di feature extraction (Sequencer2D)
    feature_extractor = load_custom_feature_extractor(model_path)

    # Itera su CIFAR-100 e calcola le predizioni
    results = []
    for idx in range(len(cifar100_dataset)):
        image, label = cifar100_dataset[idx]

        # Aggiungi una dimensione batch al tensor
        input_tensor = image.unsqueeze(0)

        # Estrai feature dall'immagine
        image_features = extract_image_features(feature_extractor, input_tensor)

        # Predici la classe
        predicted_class, similarity_score = predict_class(image_features, class_vectors_cifar100, unseen_classes_cifar100)

        # Log del risultato
        result = {
            'image_path': f'CIFAR100_{idx}',
            'dataset': 'CIFAR-100',
            'predicted_class': predicted_class,
            'similarity_score': similarity_score
        }
        results.append(result)

        # Stampa del risultato corrente per monitorare il progresso
        print(f"[CIFAR-100] Immagine {idx} -> Predetta: {predicted_class} (Somiglianza: {similarity_score:.4f})")

    # Itera sulle immagini generiche e calcola le predizioni
    for image_name in os.listdir(generic_image_folder):
        image_path = os.path.join(generic_image_folder, image_name)

        # Carica e trasforma l'immagine
        transform_generic = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = Image.open(image_path)
        input_tensor = transform_generic(image).unsqueeze(0)

        # Estrai feature dall'immagine
        image_features = extract_image_features(feature_extractor, input_tensor)

        # Predici la classe
        predicted_class, similarity_score = predict_class(image_features, class_vectors_cifar100, unseen_classes_cifar100)

        # Log del risultato
        result = {
            'image_path': image_path,
            'dataset': 'Generic Images',
            'predicted_class': predicted_class,
            'similarity_score': similarity_score
        }
        results.append(result)

        # Stampa del risultato corrente per monitorare il progresso
        print(f"[Immagine Generica] {image_path} -> Predetta: {predicted_class} (Somiglianza: {similarity_score:.4f})")

    # Salva tutti i risultati nel file CSV
    save_results_to_csv(log_file_path, results)
    print(f"Tutti i risultati salvati in {log_file_path}")
