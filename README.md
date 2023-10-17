This Repository contains the code used for the MSc Thesis:

# Automatically Identifying Materializations of Emerging Risks 

## Is it possible?

This code is to be used in combination with a polybox folder accesss to which can be provided upon request.

The manuscript can be found at: https://github.com/felanders/MSc-Thesis-Emerging-Risks (request access from the author if neccessary).

Some relevant data and models can be found at:  https://huggingface.co/collections/felanders/thesis-emerging-risks-65256dfafb1678bd6c42ab96.
---

## Folder Structure 
further descriptions of contents in the respective readme's

- `/code`: contains code and skripts
- `/config`: configurations files among them
    - `requirements.txt`: use this to create a virtual environment (using pyhton 3.10.10)
    - This folder also contains a file called `.env` (which is excluded from git as it contains private information) containing:
        ```
        BASE_PATH="/home/your/path/to/the/project/folder"
        HF_TOKEN="hf_your_hugging_face_token"
        OPEN_AI_TOKEN="your_openai_token"
        OPEN_AI_ORG="your_openai_org"
        COHERE_API_KEY="your_cohere_api_key"````
- `/Dashboard`: The code powering the dashboard at: https://dashboard.thesis-emerging-risks.live/topic-overview
- `/data`: not in git but avaialble in the polybox folder
- `/models`: not in git but avaialble in the polybox folder
- `/notebooks`: the jupyter notebooks used for different steps of analysis and data processing
- `/notes`: (not in git but avaialble in the polybox folder) contains an Obsidian vault (ie mostly markdown files) which was used to collect some notes.
- `/writing`: not in git but tracked here:  https://github.com/felanders/MSc-Thesis-Emerging-Risks and also present in the polybox folder some notebooks save figures and tables into this folder or its subfolders.
