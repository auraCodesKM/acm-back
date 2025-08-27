# 🏛️ Constitution RAG Backend

A sophisticated AI-powered Retrieval-Augmented Generation (RAG) system for constitutional law queries, built with FastAPI, Google Gemini, and advanced document processing capabilities.

## 🚀 Features

### Core Capabilities
- **Constitutional Law Expert AI**: Specialized prompts for constitutional analysis
- **Multi-Document Support**: Process PDFs, web pages, and documents from URLs
- **Advanced RAG Pipeline**: Semantic search + BM25 hybrid retrieval with reranking
- **Real-time Processing**: Efficient document chunking and vector indexing
- **Smart Caching**: TTL/LRU caching for optimal performance
- **Production Ready**: Comprehensive logging, error handling, and monitoring

### AI & ML Components
- **Google Gemini 2.0 Flash**: Primary language model for response generation
- **Sentence Transformers**: all-MiniLM-L6-v2 for embeddings
- **Cross-Encoder Reranking**: ms-marco-MiniLM-L-6-v2 for precision
- **FAISS Vector Store**: High-performance similarity search
- **BM25 Keyword Search**: Complementary lexical matching

## 📋 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Document       │    │   AI Models     │
│   Web Server    │◄──►│   Processing     │◄──►│   & Embeddings  │
│                 │    │                  │    │                 │
│ • REST API      │    │ • PDF Loader     │    │ • Gemini 2.0    │
│ • CORS Support  │    │ • Text Chunking  │    │ • SentenceT5    │
│ • Auth Optional │    │ • URL Fetching   │    │ • Cross-Encoder │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Caching       │    │   Vector Store   │    │   Response      │
│   System        │    │   & Search       │    │   Generation    │
│                 │    │                  │    │                 │
│ • TTL Cache     │    │ • FAISS Index    │    │ • Prompt Eng.   │
│ • LRU Cache     │    │ • BM25 Search    │    │ • Context Aware │
│ • Memory Mgmt   │    │ • Hybrid Fusion  │    │ • Constitutional │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Gemini API Key
- 4GB+ RAM recommended

### Setup

1. **Clone Repository**
```bash
git clone <repository-url>
cd constitution-rag
```

2. **Install Dependencies**
```bash
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
```env
GEMINI_API_KEY=your_gemini_api_key_here
PORT=8000
ENVIRONMENT=production
```

4. **Run Server**
```bash
python main.py
```

## 📡 API Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "online",
  "service": "Constitution RAG System with Google Gemini",
  "version": "2.1.0",
  "timestamp": "2025-08-27T08:30:00.000Z"
}
```

#### Query Constitution
```http
POST /api/v1/query
```

**Request:**
```json
{
  "questions": ["What is Article 21 of the Constitution?"],
  "documents": "https://example.com/constitution.pdf"
}
```

**Response:**
```json
{
  "answers": [
    "Article 21 of the Constitution guarantees the Right to Life and Personal Liberty..."
  ]
}
```

#### Cache Statistics
```http
GET /cache-stats
```

**Response:**
```json
{
  "cache_hits": 150,
  "cache_misses": 25,
  "hit_ratio": 0.857,
  "total_entries": 45
}
```

## ⚙️ Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 800 | Document chunk size for processing |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `SEMANTIC_SEARCH_K` | 20 | Number of semantic search results |
| `CONTEXT_DOCS` | 8 | Documents used for context |
| `CONFIDENCE_THRESHOLD` | 0.3 | Minimum confidence for responses |

### Performance Tuning

**Memory Optimization:**
- Adjust `CHUNK_SIZE` based on available RAM
- Configure cache TTL for your usage patterns
- Monitor vector store size

**Response Quality:**
- Increase `CONTEXT_DOCS` for comprehensive answers
- Adjust `CONFIDENCE_THRESHOLD` for precision vs recall
- Fine-tune reranking parameters

## 🚀 Deployment

### Railway Deployment

1. **Prepare for Railway**
```bash
# Railway will auto-detect Python and install requirements
echo "web: python main.py" > Procfile
```

2. **Environment Variables**
Set in Railway dashboard:
- `GEMINI_API_KEY`
- `PORT` (Railway auto-assigns)
- `ENVIRONMENT=production`

3. **Deploy**
```bash
railway up
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 📊 Monitoring & Logging

### Log Levels
- **INFO**: General operation logs
- **WARNING**: Non-critical issues
- **ERROR**: Critical failures
- **DEBUG**: Detailed debugging (development only)

### Key Metrics
- Response time per query
- Cache hit/miss ratios
- Document processing time
- Memory usage patterns
- API endpoint performance

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/

# Monitor cache performance
curl http://localhost:8000/cache-stats
```

## 🔧 Development

### Code Structure
```
constitution-rag/
├── main.py                 # FastAPI application & RAG system
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
├── README.md              # This documentation
└── deployment/
    ├── Dockerfile         # Container configuration
    └── railway.json       # Railway deployment config
```

### Key Classes
- `FastRAGSystem`: Main RAG pipeline implementation
- `CacheManager`: Intelligent caching system
- `CognitiveRouter`: Query classification and routing

### Adding New Features

1. **New Document Types**
```python
# Add to document loaders in main.py
def load_custom_format(url: str):
    # Implementation here
    pass
```

2. **Custom Prompts**
```python
# Extend _get_dynamic_prompt method
prompts['custom_type'] = f"""Your custom prompt here"""
```

## 🧪 Testing

### Manual Testing
```bash
# Test basic functionality
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"questions": ["Test question"], "documents": "test-url"}'
```

### Load Testing
```bash
# Use Apache Bench for load testing
ab -n 100 -c 10 -T application/json -p test-data.json http://localhost:8000/api/v1/query
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the troubleshooting section below

### Troubleshooting

**Common Issues:**

1. **"Model not found" error**
   - Verify GEMINI_API_KEY is set correctly
   - Check internet connectivity

2. **High memory usage**
   - Reduce CHUNK_SIZE and CONTEXT_DOCS
   - Clear cache periodically

3. **Slow responses**
   - Check document size and complexity
   - Monitor cache hit ratios
   - Consider increasing timeout values

---

Built with ❤️ for constitutional law research and education.
