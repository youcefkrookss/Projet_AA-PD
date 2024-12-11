document.getElementById('imageUpload').addEventListener('change', (event) => {
    const file = event.target.files[0];
    const preview = document.getElementById('previewImage');
    if (file) {
        preview.src = URL.createObjectURL(file);
        preview.classList.remove('hidden');
    }
});

document.getElementById('predictBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    const resultDiv = document.getElementById('result');
    const loadingDiv = document.getElementById('loading');

    resultDiv.textContent = '';
    loadingDiv.classList.add('hidden');

    // Vérifier si un fichier est sélectionné
    if (!fileInput.files || fileInput.files.length === 0) {
        resultDiv.textContent = 'Veuillez sélectionner une image avant de classer.';
        resultDiv.classList.add('text-red-500');
        return;
    }

    loadingDiv.classList.remove('hidden');

    const formData = new FormData();
    formData.append('image', fileInput.files[0]);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        loadingDiv.classList.add('hidden');

        if (data.error) {
            resultDiv.textContent = `Erreur : ${data.error}`;
            resultDiv.classList.add('text-red-500');
        } else {
            resultDiv.textContent = `Classe prédite : ${data.label}`;
            resultDiv.classList.add('text-green-500');
        }
    } catch (error) {
        loadingDiv.classList.add('hidden');
        resultDiv.textContent = 'Erreur lors de la prédiction.';
        resultDiv.classList.add('text-red-500');
    }
});

