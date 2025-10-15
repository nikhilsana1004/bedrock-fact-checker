 Amazon Bedrock Fact Checker

AI-powered fact-checking system combining Amazon Bedrock LLMs, AWS Comprehend toxicity detection, and Google Custom Search API for real-time claim verification.


 Architecture
```
User Query  Bedrock Titan (Generation)  Comprehend (Toxicity)  Google Search  Claude v2 (Verification)  Fact-Checked Response
```

Components:
- Text Generation: Amazon Bedrock Titan Text Lite v1
- Fact Verification: Amazon Bedrock Claude v2
- Toxicity Filter: AWS Comprehend Toxic Content Detection
- Web Search: Google Custom Search JSON API
- Frontend: Streamlit with session state management

 Key Features

-  Multi-model LLM pipeline (Titan  Claude)
-  Real-time toxicity scoring with configurable thresholds
-  Automated web scraping and source attribution
-  Pandas-based toxicity visualization
-  Stateful session management

  Quick Start
```bash
 Clone and setup
git clone https://github.com/nikhilsana1004/bedrock-fact-checker.git
cd bedrock-fact-checker
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

 Configure
cp .env.example .env
 Add your AWS credentials and Google API keys

 Run
streamlit run app.py
```

  Configuration

Create `.env`:
```env
AWS_REGION=us-west-2
GOOGLE_API_KEY=your_api_key
GOOGLE_CSE_ID=your_cse_id
```

AWS IAM Permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "comprehend:DetectToxicContent"
    ],
    "Resource": ""
  }]
}
```

  Technical Implementation

 1. Text Generation Pipeline
```python
 Uses Amazon Titan with JSON payload
bedrock_runtime.invoke_model(
    modelId="amazon.titan-text-lite-v1",
    body=json.dumps({"inputText": prompt})
)
```

 2. Toxicity Detection
```python
 Comprehend API with 0.5 threshold
comprehend.detect_toxic_content(
    TextSegments=[{"Text": text}],
    LanguageCode='en'
)
 Filters: PROFANITY, HATE_SPEECH, INSULT, HARASSMENT
```

 3. Fact-Checking Workflow

Step 1: Query Google Custom Search API
```python
params = {"key": api_key, "cx": cse_id, "q": query, "num": 5}
```

Step 2: Extract and aggregate snippets
```python
snippets = [item['snippet'] for item in search_results['items']]
```

Step 3: Summarize with Claude v2
```python
prompt = f"Summarize: {combined_snippets}"
bedrock_runtime.invoke_model(modelId='anthropic.claude-v2')
```

Step 4: Cross-verify against original claim
```python
prompt = f"Fact-check '{statement}' using: {summary}"
```

 4. State Management

Uses Streamlit session state for multi-step interactions:
```python
st.session_state.response_text = response
st.session_state.toxicity_details = toxicity_data
```

  Model Specifications

| Component | Model | Purpose | Max Tokens |
|-----------|-------|---------|------------|
| Generation | Titan Text Lite v1 | Initial response | Default |
| Summarization | Claude v2 | Snippet aggregation | 200 |
| Verification | Claude v2 | Fact-checking | 200 |
| Toxicity | Comprehend | Content moderation | N/A |

  Performance Considerations

- Latency: ~2-4s (Bedrock) + ~1-2s (Comprehend) + ~1s (Google)
- Rate Limits: 
  - Bedrock: Region-specific TPS limits
  - Google CSE: 100 queries/day (free tier)
- Cost Optimization: 
  - Use Titan Lite for generation (cheaper)
  - Cache Google search results
  - Batch Comprehend requests where possible

  Tech Stack
```
Python 3.8+
 streamlit (UI framework)
 boto3 (AWS SDK)
 pandas (data manipulation)
 requests (HTTP client)
 python-dotenv (config management)
```

  Security Best Practices

1. API Key Management: Store in `.env`, never commit
2. IAM Least Privilege: Scope permissions to specific models
3. Input Validation: Sanitize user queries before API calls
4. Rate Limiting: Implement request throttling
5. Error Handling: Catch credential errors gracefully

  Troubleshooting

AWS Credentials Error:
```bash
aws configure
 Or set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
```

Bedrock Model Access:
```bash
 Request model access in AWS Console
 Bedrock > Model access > Request access
```

Google API Quota:
```python
 Implement exponential backoff
time.sleep(2  attempt)
```

  Example Request/Response

Input:
```
"Is coffee good for health?"
```

Pipeline:
1. Titan generates: "Coffee has antioxidants and may reduce disease risk..."
2. Comprehend scores: Toxicity=0.02 (PASS)
3. Google finds 5 sources from health journals
4. Claude summarizes research consensus
5. Claude fact-checks: "VERIFIED with caveats..."

Output:
```
Fact-Check Summary: The claim is largely supported...
Sources:
- [Mayo Clinic] https://...
- [Harvard Health] https://...
```

  API Documentation

- [Bedrock API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/)
- [Comprehend Toxicity](https://docs.aws.amazon.com/comprehend/latest/dg/how-toxicity.html)
- [Google CSE API](https://developers.google.com/custom-search/v1/overview)

  License

MIT License - see [LICENSE](LICENSE)

  Contact

Nikhil Sana  
GitHub: [@nikhilsana1004](https://github.com/nikhilsana1004)  
Project: [bedrock-fact-checker](https://github.com/nikhilsana1004/bedrock-fact-checker)

---

 Built with AWS Bedrock | Powered by Claude & Titan
