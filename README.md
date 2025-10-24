## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:

Extracting specific, nuanced information from a collection of dense academic papers is a slow and inefficient manual process. Standard search tools rely on exact keywords and fail to understand the conceptual context of a user's question. This program aims to build an AI agent that can intelligently query multiple documents to synthesize precise answers to complex questions.

### DESIGN STEPS:

#### STEP 1:
Load PDF documents and create specialized search and summary tools for each paper.

#### STEP 2:
Initialize an AI agent with an OpenAI model, giving it access to all the created tools.

#### STEP 3:
Query the agent with a specific question about one paper to get a detailed answer from its content.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
```
```
import nest_asyncio
nest_asyncio.apply()
```
```
urls = [
    "https://openreview.net/pdf?id=u31BLudWQr",
    "https://openreview.net/pdf?id=H1lnZlHYDS",
    "https://openreview.net/pdf?id=gx7TEqAogg8",
]

papers = [
    "1_Virtual_Personalized_Fashion.pdf",
    "2151_provable_convergence_and_globa.pdf",
    "99_challenging_america_digitized_.pdf",
]
```
```
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
```
```
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
```
```
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
```
```
len(initial_tools)
```
```
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
```
```
response = agent.query(
    "Tell me about fashion from the paper 1_Virtual_Personalized_Fashion, "
    "and then tell me about different styles"
)
```
### OUTPUT:

<img width="1319" height="699" alt="image" src="https://github.com/user-attachments/assets/a493e561-619f-42b6-9911-ae9996a6c291" />

### RESULT:

The system successfully retrieves and synthesizes relevant information from multiple documents, providing concise and relevant answers to the user's query. Performance is evaluated based on the accuracy, relevance, and coherence of the responses.    
