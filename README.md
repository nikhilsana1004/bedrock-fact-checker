🧠 Amazon Bedrock Fact Checker

AI-powered fact-checking system using Amazon Bedrock LLMs, AWS Comprehend, and Google Custom Search API for real-time claim verification.

🚀 Architecture

Flow:
User Query → Titan (Text Gen) → Comprehend (Toxicity) → Google Search → Claude v2 (Verification) → ✅ Fact-Checked Response

Core Components:

Text Generation: Amazon Titan Text Lite v1

Fact Verification: Claude v2 via Amazon Bedrock

Toxicity Detection: AWS Comprehend

Web Search: Google Custom Search API

Frontend: Streamlit (session-managed UI)

⚙️ Setup
git clone https://github.com/nikhilsana1004/bedrock-fact-checker.git
cd bedrock-fact-checker
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt


Create .env:

AWS_REGION=us-west-2
GOOGLE_API_KEY=your_key
GOOGLE_CSE_ID=your_id


Run:

streamlit run app.py

🔐 IAM Permissions
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel", "comprehend:DetectToxicContent"],
  "Resource": "*"
}

🧩 Pipeline Overview

Titan (Text Gen):

bedrock.invoke_model("amazon.titan-text-lite-v1", {"inputText": prompt})


Comprehend (Toxicity, threshold=0.5):

comprehend.detect_toxic_content(TextSegments=[{"Text": text}])


Google Search (Top 5 results)

Claude v2 (Summarize + Verify)
Summarizes sources and cross-verifies claim.

📊 Features

Multi-model LLM pipeline (Titan + Claude)

Real-time toxicity scoring

Automated source aggregation

Pandas-based result visualization

Streamlit state management

⚡ Performance & Cost

Latency: 3–7s avg (Titan + Comprehend + Google)

Rate Limits: Bedrock (regional) | Google (100/day free)

Optimize: Cache searches, batch Comprehend calls.

🛡️ Security

.env for API keys (never commit)

Least-privilege IAM

Input sanitization & rate limiting

Graceful credential error handling

🧪 Example

Input: “Is coffee good for health?”
Output:
✅ Largely supported — coffee may reduce disease risk (Harvard, Mayo Clinic).

📚 Tech Stack

Python · Streamlit · Boto3 · Pandas · Requests · Dotenv

📄 License

MIT License — see LICENSE

Author: Nikhil Sana

Built with AWS Bedrock, powered by Titan & Claude