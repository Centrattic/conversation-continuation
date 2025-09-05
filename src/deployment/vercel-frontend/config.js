window.APP_CONFIG = {
    // This will be auto-updated by the Python app when ngrok starts
    apiBase: 'https://27fc2c7bdd2b.ngrok-free.app',
    riyaName: 'Riya',
    owenName: 'Owen',
    maxNewTokens: 50,
    // Adapter paths
    adapters: [
        {
            value: 'mistral-7b/mistral-results-7-27-25/lora_train/checkpoint-11796',
            label: 'Mistral 7B (7/27/25)'
        },
        {
            value: 'mistral-7b/mistral-results-7-6-25/checkpoint-8103',
            label: 'Mistral 7B (7/6/25)'
        },
        {
            value: 'gemma-3-27b-it/gemma-3-27b-it_20250903_122252/training_output/checkpoint-276',
            label: 'Gemma 3 27B IT (9/3/25)'
        }
    ]
};


