# AWS RAG Setup Guide
## Amazon Bedrock + Amazon OpenSearch Serverless

This guide covers creating all AWS resources needed to run `aws_rag_ai.py`.

---

## Prerequisites

- AWS account with billing enabled
- AWS CLI installed and configured (`aws configure`)
- Python 3.10+

---

## 1. Authentication — Choose Your Method

There are three ways to authenticate. Pick one based on your use case.

---

### Option A — Bedrock API Key ✅ Recommended for development (NEW, July 2025)

Amazon Bedrock now has its own native API keys — no IAM user setup needed.
Two types are available:

**Short-term key (up to 12 hours):**
1. Open the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock)
2. In the left sidebar → **API keys**
3. **Short-term API keys** tab → click **Generate short-term API key**
4. Copy the key — it expires when your console session expires

**Long-term key (custom expiry):**
1. Open the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock)
2. In the left sidebar → **API keys**
3. **Long-term API keys** tab → click **Generate long-term API key**
4. Set expiry duration → the `AmazonBedrockLimitedAccess` managed policy is auto-attached
5. Copy and store the key securely

> **Note**: Bedrock API keys only authenticate Bedrock calls. For OpenSearch Serverless you still need IAM credentials (Option B or C).

---

### Option B — IAM User (local development, full access)

1. Go to **IAM → Users → Create user**
2. Name it (e.g. `rag-dev-user`)
3. Select **Attach policies directly** and attach the inline policy below
4. Go to **Security credentials → Create access key** → choose **Local code**
5. Save the `Access Key ID` and `Secret Access Key`

### Option C — IAM Role (EC2 / Lambda / SageMaker, no keys needed)

1. Go to **IAM → Roles → Create role**
2. Choose the trusted entity (EC2, Lambda, etc.)
3. Attach the inline policy below
4. Assign the role to your compute resource — credentials are injected automatically

### IAM Policy (required for Options B and C)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    },
    {
      "Sid": "OpenSearchServerless",
      "Effect": "Allow",
      "Action": [
        "aoss:APIAccessAll"
      ],
      "Resource": "*"
    }
  ]
}
```

---

## 2. Amazon Bedrock — Model Access

> **As of September 29, 2025, the Model Access page is retired.**
> All serverless foundation models are automatically enabled for every AWS account.
> No manual activation is required.

The models used by this project:

| Provider  | Model ID                                        | Used for   |
|-----------|-------------------------------------------------|------------|
| Amazon    | `amazon.titan-embed-text-v2:0`                  | Embeddings |
| Anthropic | `us.anthropic.claude-3-5-sonnet-20241022-v2:0`  | Chat / RAG |

Access is controlled exclusively through **IAM policies** (see Step 1).
To restrict or allow specific models, use `bedrock:InvokeModel` with a scoped `Resource` ARN in your IAM policy.

> **Region note**: The `us.` prefix uses cross-region inference, automatically routing across `us-east-1`, `us-east-2`, and `us-west-2` for higher throughput and availability.

---

## 3. Amazon OpenSearch Serverless — Create Collection

> Console path: **Amazon OpenSearch Service → Serverless → Collections**

### 3a. Create Encryption Policy (required before collection)

1. Go to **Serverless → Security → Encryption policies → Create**
2. Name: `rag-encryption-policy`
3. Resource pattern: `rag-*`
4. Key: **AWS owned key** (default, free)
5. Click **Create**

### 3b. Create Network Policy

1. Go to **Serverless → Security → Network policies → Create**
2. Name: `rag-network-policy`
3. Resource type: **Collection**
4. Resource: `rag-*`
5. Access type: **Public** (for development) or **VPC** (for production)
6. Click **Create**

### 3c. Create the Collection

1. Go to **Serverless → Collections → Create collection**
2. Fill in:
   - **Name**: `rag-collection`
   - **Type**: `Vector search`
   - **Encryption**: select `rag-encryption-policy`
   - **Network**: select `rag-network-policy`
3. Click **Create** — takes ~2 minutes
4. Once **Active**, copy the **OpenSearch Endpoint** URL
   (format: `https://<id>.us-east-1.aoss.amazonaws.com`)
   → paste this into `OPENSEARCH_ENDPOINT` in `aws_rag_ai.py`

### 3d. Create Data Access Policy

This grants your IAM user/role permission to read/write the index.

1. Go to **Serverless → Security → Data access policies → Create**
2. Name: `rag-data-access`
3. Add principal: paste your IAM user ARN or role ARN
   (format: `arn:aws:iam::<account-id>:user/rag-dev-user`)
4. Grant permissions:
   - **Collections**: `rag-collection` → check **Read, Write**
   - **Indexes**: `rag-collection/*` → check **Read, Write, Create**
5. Click **Create**

---

## 4. Create the Vector Index

The index is auto-created by `OpenSearchVectorSearch.from_documents()` on first run with the correct kNN mapping for Titan Embeddings V2 (1024 dimensions).

If you prefer to create it manually via the OpenSearch Dashboards:

```json
PUT /rag-index
{
  "settings": {
    "index": {
      "knn": true
    }
  },
  "mappings": {
    "properties": {
      "vector_field": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "engine": "faiss",
          "space_type": "l2"
        }
      },
      "text": { "type": "text" },
      "metadata": { "type": "object" }
    }
  }
}
```

---

## 5. Configure the Script

Set these values in `aws_rag_ai.py`:

```python
DATA_MODE             = "pdf"          # or "excel"
AWS_REGION            = "us-east-1"
AWS_ACCESS_KEY_ID     = "AKIA..."      # leave empty to use env vars / IAM role
AWS_SECRET_ACCESS_KEY = "..."
OPENSEARCH_ENDPOINT   = "https://<id>.us-east-1.aoss.amazonaws.com"
OPENSEARCH_INDEX      = "rag-index"
```

Or export as environment variables (recommended — keeps keys out of code):

```bash
# Option B — IAM User keys
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Option A — Bedrock API key (only covers Bedrock calls, not OpenSearch)
export AWS_BEDROCK_API_KEY=brk-...
```

> If using an IAM Role (Option C), leave all key fields empty — boto3 picks up the role credentials automatically from the instance metadata.

---

## 6. Install Dependencies

```bash
pip install langchain-aws langchain-community opensearch-py requests-aws4auth boto3
```

Or add to your existing `requirements.txt` (already included).

---

## 7. Run

```bash
# First run — index your documents (uncomment index_documents in __main__)
python aws_rag_ai.py

# Subsequent runs — load existing index
python aws_rag_ai.py
```

---

## AWS Service Cost Reference

| Service                        | Free tier / pricing |
|-------------------------------|---------------------|
| Amazon Bedrock (Titan Embed V2) | Pay per 1K tokens (~$0.00002) |
| Amazon Bedrock (Claude 3.5 Sonnet) | ~$3 input / $15 output per 1M tokens |
| OpenSearch Serverless          | Min 2 OCUs (~$0.24/hr), no free tier |

> Tip: Delete the collection when not in use to avoid OpenSearch Serverless OCU charges.
