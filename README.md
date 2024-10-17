## Directory Structure
```bash
project_root/
├── app.py
├── requirements.txt
├── gunicorn_config.py
├── chat_memory/
│   └── chat_{chat_id}.json
├── documents/
│   ├── doc1.pdf
│   └── doc2.pdf
├── embeddings/
│   ├── index.faiss
│   └── index.pkl
├── utils/
│   ├── pdf_processor.py
│   └── memory_manager.py
└── README.md

```



## Run command
```bash
gunicorn -c gunicorn_config.py app:app

```