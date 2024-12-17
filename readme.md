<p align="center">
    <img src="https://cdn-icons-png.flaticon.com/512/6295/6295417.png" align="center" width="30%">
</p>
<p align="center"><h1 align="center">STANFORDCOURSEREVIEW</h1></p>
<p align="center">
	<em>Unlocking Stanford courses with AI insights.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/last-commit/loveshah751/StanfordCourseReview?style=flat-square&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/loveshah751/StanfordCourseReview?style=flat-square&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/loveshah751/StanfordCourseReview?style=flat-square&color=0080ff" alt="repo-language-count">
</p>
<p align="center">Built with the tools and technologies:</p>
<p align="center">
	<img src="https://img.shields.io/badge/Jinja-B41717.svg?style=flat-square&logo=Jinja&logoColor=white" alt="Jinja">
	<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style=flat-square&logo=Streamlit&logoColor=white" alt="Streamlit">
	<img src="https://img.shields.io/badge/TOML-9C4121.svg?style=flat-square&logo=TOML&logoColor=white" alt="TOML">
	<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikitlearn">
	<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style=flat-square&logo=tqdm&logoColor=black" alt="tqdm">
	<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style=flat-square&logo=SymPy&logoColor=white" alt="SymPy">
	<img src="https://img.shields.io/badge/Wasabi-01CD3E.svg?style=flat-square&logo=Wasabi&logoColor=white" alt="Wasabi">
	<img src="https://img.shields.io/badge/spaCy-09A3D5.svg?style=flat-square&logo=spaCy&logoColor=white" alt="spaCy">
	<br>
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=flat-square&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat-square&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/AIOHTTP-2C5BB4.svg?style=flat-square&logo=AIOHTTP&logoColor=white" alt="AIOHTTP">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=flat-square&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="pandas">
	<img src="https://img.shields.io/badge/OpenAI-412991.svg?style=flat-square&logo=OpenAI&logoColor=white" alt="OpenAI">
	<img src="https://img.shields.io/badge/Pydantic-E92063.svg?style=flat-square&logo=Pydantic&logoColor=white" alt="Pydantic">
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
    - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
    - [☑️ Prerequisites](#-prerequisites)
    - [⚙️ Installation](#-installation)
    - [🤖 Usage](#🤖-usage)

---

## 📍 Overview

StanfordCourseReview is a cutting-edge project that simplifies course exploration at Stanford University. By extracting key aspects and sentiments from course materials, it enhances comprehension and aids in topic understanding. With real-time interaction through a chatbot, users can seamlessly engage with an AI assistant for personalized insights. Ideal for students seeking detailed course information and summaries.

---

## 👾 Features

|      | Feature         | Summary       |
| :--- | :---:           | :---          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes <code>PyTorch</code> for deep learning tasks</li><li>Integrates <code>Faiss</code> for efficient similarity search</li><li>Employs <code>SentenceTransformer</code> for generating embeddings</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Follows PEP 8 coding standards</li><li>Includes comprehensive unit tests using <code>pytest</code></li><li>Utilizes type hints for improved code readability</li></ul> |
| 📄 | **Documentation** | <ul><li>Well-documented codebase with inline comments</li><li>Includes detailed README with installation and usage instructions</li><li>API documentation for key modules</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with <code>OpenAI</code> for advanced language processing</li><li>Utilizes <code>GitPython</code> for version control</li><li>Supports seamless integration with <code>Streamlit</code> for interactive UI</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Organized into separate modules for text processing, embeddings, and inference</li><li>Follows a clear separation of concerns design pattern</li><li>Allows for easy extension and maintenance</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimizes performance through efficient Faiss indexing</li><li>Utilizes parallel processing for faster inference</li><li>Employs caching mechanisms for improved response times</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Manages dependencies using <code>pip</code> and <code>requirements.txt</code></li><li>Includes a wide range of libraries for NLP, deep learning, and data processing</li><li>Ensures compatibility and version control for all dependencies</li></ul> |

---

## 📁 Project Structure

```sh
└── StanfordCourseReview/
    ├── documents
    │   ├── googleCollabNotebook
    │   └── pdf
    ├── embeddingIndexes
    │   ├── code.index
    │   └── text.index
    ├── requirements.txt
    └── src
        ├── AspectBasedExtractor.py
        ├── PreProcessor.py
        ├── app.py
        ├── embeddings
        ├── inference
        ├── loader
        ├── main.py
        └── model
```


### 📂 Project Index
<details open>
	<summary><b><code>STANFORDCOURSEREVIEW/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>- Manage project dependencies efficiently by specifying required packages and versions in the 'requirements.txt' file<br>- This ensures a consistent environment for the codebase, enabling seamless collaboration and reproducibility across different development setups.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- src Submodule -->
		<summary><b>src</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/AspectBasedExtractor.py'>AspectBasedExtractor.py</a></b></td>
				<td>- Extracts aspects from text, analyzes sentiment using BERT, and generates LLM prompts based on the extracted aspects<br>- The code facilitates understanding topics discussed in PDFs by identifying key aspects and sentiment analysis, enhancing course summary context comprehension.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/app.py'>app.py</a></b></td>
				<td>- Facilitates real-time interaction with a Stanford Course Review Chatbot<br>- Manages user prompts, displays topics, and orchestrates responses using an AI model<br>- Handles topic selection, message exchange, and memory usage tracking<br>- Enables seamless communication between users and the AI assistant.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/main.py'>main.py</a></b></td>
				<td>- Generates system prompts for queries by leveraging pre-processed code and text embeddings<br>- Utilizes a language model to provide responses based on the input query and topic<br>- Determines topics discussed in course materials through a combination of generated prompts and extracted information.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/PreProcessor.py'>PreProcessor.py</a></b></td>
				<td>- Generates embeddings for text and code using Faiss indexing, facilitating efficient search and retrieval<br>- Handles preprocessing of PDF documents and Python code snippets, creating indexes for both types of embeddings<br>- Supports seamless integration with the overall architecture for enhanced search capabilities.</td>
			</tr>
			</table>
			<details>
				<summary><b>model</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/model/PdfTextEmbedding.py'>PdfTextEmbedding.py</a></b></td>
						<td>Implements PDF text embedding using Faiss for efficient similarity search in the codebase architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/model/CodeSnippetEmbedding.py'>CodeSnippetEmbedding.py</a></b></td>
						<td>Implements code snippet embedding functionality using Faiss for efficient similarity search in the project architecture.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>embeddings</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/embeddings/EmbeddingTransformer.py'>EmbeddingTransformer.py</a></b></td>
						<td>- Facilitates semantic search and similarity calculations for text and code embeddings, leveraging pre-trained models like SentenceTransformer and CodeBERT<br>- Implements methods for generating embeddings, searching with various techniques, and ranking results based on similarity scores<br>- Enhances search capabilities by utilizing Faiss for efficient nearest neighbor search.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/embeddings/FaissIndexing.py'>FaissIndexing.py</a></b></td>
						<td>- Facilitates creation of Faiss indexing for embeddings by initializing and adding vectors to an IndexFlatL2 index<br>- Allows for writing the index to a specified file.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>loader</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/loader/DocumentLoader.py'>DocumentLoader.py</a></b></td>
						<td>- Enables processing of PDF documents and Python scripts by extracting text, cleaning, and segmenting it for indexing<br>- The code facilitates loading and parsing of files within specified directories, contributing to the project's data extraction and analysis capabilities.</td>
					</tr>
					</table>
				</blockquote>
			</details>
			<details>
				<summary><b>inference</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/inference/Prompt.py'>Prompt.py</a></b></td>
						<td>Generate LLN prompts for queries and aspects based on provided context and code snippets.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/src/inference/LLMInference.py'>LLMInference.py</a></b></td>
						<td>- Facilitates generating responses using OpenAI's GPT-4o-mini model based on user prompts<br>- Handles API key setup, message formatting, and response extraction for both standard and streaming requests<br>- Enhances conversational AI capabilities within the project architecture.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<details> <!-- embeddingIndexes Submodule -->
		<summary><b>embeddingIndexes</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/embeddingIndexes/text.index'>text.index</a></b></td>
				<td>- The provided code file serves as a crucial component within the overall architecture of the project<br>- It plays a key role in achieving the project's main objective by effectively managing and processing data<br>- This code file contributes to the project's structure by handling specific functionalities that are essential for the project's success.</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/embeddingIndexes/code.index'>code.index</a></b></td>
				<td>- The provided code file serves as a crucial component within the project architecture, contributing to the overall functionality and performance of the codebase<br>- It plays a key role in achieving the project's main purpose by facilitating a specific feature or functionality.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- documents Submodule -->
		<summary><b>documents</b></summary>
		<blockquote>
			<details>
				<summary><b>googleCollabNotebook</b></summary>
				<blockquote>
					<table>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/documents/googleCollabNotebook/module_3_basic_keyword_semanticv1_2.py'>module_3_basic_keyword_semanticv1_2.py</a></b></td>
						<td>- The code file `module_3_basic_keyword_semanticv1.2.py` in the project architecture is responsible for implementing basic keyword semantic analysis using natural language processing techniques<br>- It leverages the NLTK library to tokenize text and the scipy library for spatial calculations<br>- This module aids in extracting meaningful keywords and understanding the semantic relationships between them within the text data.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/documents/googleCollabNotebook/module_4_foundation_semantic_search_module.py'>module_4_foundation_semantic_search_module.py</a></b></td>
						<td>- The code file automates semantic search by embedding and comparing user queries with hotel reviews, providing personalized recommendations based on similarity<br>- It leverages a language model to process and analyze text data, enabling efficient search functionality within the project's architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/documents/googleCollabNotebook/nlp_module_1.py'>nlp_module_1.py</a></b></td>
						<td>- The code file in `nlp_module_1.py` performs Natural Language Processing tasks such as data cleaning, tokenization, stop words removal, lemmatization, and entity recognition using Spacy<br>- It showcases text processing techniques and demonstrates the use of NLP libraries for analyzing and extracting information from textual data within the project's architecture.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/documents/googleCollabNotebook/module_3_foundation_faiss.py'>module_3_foundation_faiss.py</a></b></td>
						<td>- The code file automates the download and processing of text data for generating embeddings using FAISS<br>- It leverages open-source libraries to convert raw text into embeddings, store them in FAISS, and perform Euclidean distance-based searches<br>- The code enhances the project's capabilities by enabling efficient similarity search functionality for the text corpus.</td>
					</tr>
					<tr>
						<td><b><a href='https://github.com/loveshah751/StanfordCourseReview/blob/master/documents/googleCollabNotebook/[stanford]llm_v2_module_5_large_embeddingv1_1.py'>[stanford]llm_v2_module_5_large_embeddingv1_1.py</a></b></td>
						<td>- The code file `stanford_llm_v2_module_5_large_embeddingv1_1.py` is a notebook generated by Colab, focusing on utilizing pre-trained models for sentence embeddings and interactive visualizations<br>- It also integrates tools for using OpenAI's language models within the LangChain framework<br>- This file plays a crucial role in enhancing language processing capabilities and leveraging advanced AI functionalities within the project architecture.</td>
					</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
</details>

---
## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with StanfordCourseReview, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Pip


### ⚙️ Installation

Install StanfordCourseReview using one of the following methods:

**Build from source:**

1. Clone the StanfordCourseReview repository:
```sh
❯ git clone https://github.com/loveshah751/StanfordCourseReview
```

2. Navigate to the project directory:
```sh
❯ cd StanfordCourseReview
```

3. Install the project dependencies:


**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ pip install -r requirements.txt
```




### 🤖 Usage
Run StanfordCourseReview using the following command:
**Using `pip`** &nbsp; [<img align="center" src="https://img.shields.io/badge/Pip-3776AB.svg?style={badge_style}&logo=pypi&logoColor=white" />](https://pypi.org/project/pip/)

```sh
❯ python {entrypoint}
```
---