window.APP_CONFIG = {
    // This will be auto-updated by the Python app when ngrok starts
    apiBase: 'https://7805e234ec22.ngrok-free.app',
    riyaName: 'Riya',
    owenName: 'Owen',
    maxNewTokens: 150,
    // Adapter paths
    adapters: [
        {
            value: 'mistral-7b/mistral-results-7-27-25/lora_train/lora_adapter',
            label: 'Mistral 7B (7/27/25)'
        },
        {
            value: 'mistral-7b/mistral-results-7-6-25/lora_adapter',
            label: 'Mistral 7B (7/6/25)'
        },
        {
            value: 'gemma-3-27b-it/gemma-3-27b-it_20250903_122252/lora_adapter',
            label: 'Gemma 3 27B IT (9/3/25)'
        }
    ]
};


